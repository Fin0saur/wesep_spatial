import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.nbc2 import NBC2
from wesep.modules.common.deep_update import deep_update

class TSE_NBC2_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. Basic Configs ---
        self.win = config.get("win",512)
        self.stride = config.get("stride",256)
        
        # [优化] 使用 register_buffer 自动管理设备，无需在 forward 中 .to(device)
        self.window = torch.hann_window(self.win)
        
        freq_bins = self.win // 2 + 1
        
        # --- 2. Spatial Configs ---
        spatial_configs = {
            "geometry": {
                # [关键] 确保这里的 n_fft 与 STFT 实际使用的参数一致
                "n_fft": self.win,              
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.03333333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [
                [0, 1], [1, 2], [2, 3], [0, 3]
            ],
            "features": {
                "ipd": {"enabled": True},
                "cdf": {"enabled": True},
                "sdf": {"enabled": True},
                "delta_stft": {"enabled": True},
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        
        # --- 3. Dynamic Input Size Calculation ---
        spec_feat_dim = 2 
        
        n_pairs = len(self.spatial_configs['pairs'])
        feat_cfg = self.spatial_configs['features']
        spatial_dim=0
        if feat_cfg.get('ipd', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): spatial_dim += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False): spatial_dim += 2*n_pairs
        
        total_input_size = spec_feat_dim + spatial_dim
        # print(f"Dynamic Input Size: {total_input_size}") # Debug用

        # --- 4. Backbone Configs ---
        block_kwargs = {
            'n_heads': 2,
            # 'dropout': 0.0,
            'conv_kernel_size': 3,
            'n_conv_groups': 8,
            'norms': ("LN", "GBN", "GBN"),
            'group_batch_norm_kwargs': {
                'group_size': freq_bins,
                'share_along_sequence_dim': False,
            },
        }
        
        sep_configs = dict(
            input_size=total_input_size, # 使用动态计算的值
            output_size=2, # 假设 NBC2 内部处理这里代表输出 mask 或 complex
            n_layers=8,
            dim_hidden=96,
            dim_ffn=96*2,
            block_kwargs=block_kwargs
        )
        self.sep_configs = deep_update(sep_configs, config.get('separator', {}))
        
        # --- 5. Instantiate Modules ---
        self.sep_model = NBC2(**self.sep_configs)
        
        # 通常 SpatialFrontend 在 init 时就需要 config，forward 时不应该再传 config
        # 如果你的 SpatialFrontend forward 需要 config，保持原样即可
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
    def forward(self, mix,cue):
        # input shape: (B, C, T)
        # self.window 已经在正确的 device 上了
        spatial_cue=cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]        
        B, M, T_wav = mix.shape
        self.window = self.window.to(mix.device)
        # print(f"mix_shape:{mix.shape}")
        # print(f"azimuth:{azi_rad},elerad:{ele_rad}")
        mix_reshape = mix.view(B * M, T_wav)
        
        spec = torch.stft(
            mix_reshape,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            return_complex=True
        )
        
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim)
        
        # --- 2. A-Norm ---
        Y_ref = Y[:, 0]
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)
        
        # Spectral: (B, 2, F, T)
        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1)
        
        # Spatial: (B, 16, F, T)
        spatial_feat_dict = self.spatial_ft.compute_all(Y_norm,azi_rad, ele_rad)
        spatial_feat_list = []
        for name, feat in spatial_feat_dict.items():
            if name in self.spatial_configs["features"]:
                spatial_feat_list.append(feat)
        if len(spatial_feat_list) == 0:
            raise RuntimeError("No spatial features enabled or computed!")
        
        spatial_feat = torch.cat(spatial_feat_list, dim=1)
        # --- Fusion ---
        features = torch.cat([spec_feat, spatial_feat], dim=1)
        
        # --- Backbone ---
        est_spec_feat_raw = self.sep_model(features)
        
        est_spec_feat = est_spec_feat_raw.view(B, F_dim, T_dim, 2).permute(0, 3, 1, 2)
        # --- Reconstruction ---
        est_spec = torch.complex(est_spec_feat[:, 0], est_spec_feat[:, 1])
        
        # Inverse Normalization
        est_spec = est_spec * ref_mag_mean
        
        # --- iSTFT ---
        est_wav = torch.istft(
            est_spec,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window,
            length=T_wav
        )
        
        est_wav=est_wav.unsqueeze(1) # [B 1 T]
        
        return est_wav