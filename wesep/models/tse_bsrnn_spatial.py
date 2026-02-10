import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.bsrnn import (
    BandSplit, 
    SubbandNorm, 
    BSRNN_Separator, 
    BandMasker
)
from wesep.modules.common.deep_update import deep_update

class TSE_BSRNN_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sr = 16000
        self.win = config.get("win", 512)
        self.stride = config.get("stride", 256)
        self.register_buffer("window", torch.hann_window(self.win))
        
        spatial_configs = {
            "geometry": {
                "n_fft": self.win,              
                "fs": self.sr,
                "c": 343.0,
                "mic_spacing": 0.03333333,
                "mic_coords": [[-0.05, 0, 0], [-0.0166, 0, 0], [0.0166, 0, 0], [0.05, 0, 0]],
            },
            "pairs": [[0, 1], [1, 2], [2, 3], [0, 3]],
            "features": {
                "ipd": {"enabled": True},
                "cdf": {"enabled": True},
                "sdf": {"enabled": True},
                "delta_stft": {"enabled": True},
                "cyc_doaemb": {"enabled": False}
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
        
        n_pairs = len(self.spatial_configs['pairs'])
        feat_cfg = self.spatial_configs['features']
        self.spatial_dim = 0
        if feat_cfg.get('ipd', {}).get('enabled', False): self.spatial_dim += n_pairs
        if feat_cfg.get('cdf', {}).get('enabled', False): self.spatial_dim += n_pairs
        if feat_cfg.get('sdf', {}).get('enabled', False): self.spatial_dim += n_pairs
        if feat_cfg.get('delta_stft', {}).get('enabled', False): self.spatial_dim += 2 * n_pairs
        
        sep_cfg = config.get('separator', {})
        feature_dim = sep_cfg.get('feature_dim', 128)
        num_repeat = sep_cfg.get('num_repeat', 6)
        causal = sep_cfg.get('causal', False)
        norm_type = "cLN" if causal else "GN"
        
        enc_dim = self.win // 2 + 1
        bandwidth_100 = int(np.floor(100 / (self.sr / 2.0) * enc_dim))
        bandwidth_200 = int(np.floor(200 / (self.sr / 2.0) * enc_dim))
        bandwidth_500 = int(np.floor(500 / (self.sr / 2.0) * enc_dim))
        bandwidth_2k = int(np.floor(2000 / (self.sr / 2.0) * enc_dim))

        band_width = [bandwidth_100] * 15
        band_width += [bandwidth_200] * 10
        band_width += [bandwidth_500] * 5
        band_width += [bandwidth_2k] * 1
        band_width.append(enc_dim - int(np.sum(band_width)))
        
        self.nband = len(band_width)
        
        self.band_split = BandSplit(band_width)
        self.spec_norm = SubbandNorm(
            band_width=band_width,
            spec_dim=2,
            nband=self.nband,
            feature_dim=feature_dim,
            norm_type=norm_type
        )
        
        if self.spatial_dim > 0:
            self.spatial_norm = SubbandNorm(
                band_width=band_width,
                spec_dim=self.spatial_dim,
                nband=self.nband,
                feature_dim=feature_dim,
                norm_type=norm_type
            )
        else:
            self.spatial_norm = None
            self.fusion_layer = None

        self.separator = BSRNN_Separator(
            nband=self.nband,
            num_repeat=num_repeat,
            feature_dim=feature_dim,
            causal=causal,
            norm_type=norm_type
        )
        
        self.band_masker = BandMasker(
            band_width=band_width,
            nband=self.nband,
            feature_dim=feature_dim,
            norm_type=norm_type,
            nspk=1
        )

    def forward(self, mix, cue):
        B, M, T_wav = mix.shape
        spatial_cue = cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]
        
        mix_reshape = mix.view(B * M, T_wav)
        spec = torch.stft(
            mix_reshape,
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window.to(mix.device),
            return_complex=True
        )
        _, F_dim, T_dim = spec.shape
        Y = spec.view(B, M, F_dim, T_dim) # (B, M, F, T) Complex
        
        # --- 2. Reference & Norm ---
        Y_ref = Y[:, 0] 
        ref_mag_mean = torch.abs(Y_ref).mean(dim=(1, 2), keepdim=True) + 1e-8
        Y_norm = Y / ref_mag_mean.unsqueeze(1)
        
        # --- 3. Feature Preparation (Dual Stream) ---
        
        # Stream 1: Spectral (B, 2, F, T)
        spec_feat = torch.stack([Y_norm[:, 0].real, Y_norm[:, 0].imag], dim=1)
        subband_spec = self.band_split(spec_feat)
        # Projection: (B, Nband, Dim, T)
        spec_emb = self.spec_norm(subband_spec)
        
        # Stream 2: Spatial (B, SpatialDim, F, T)
        if self.spatial_dim > 0:
            spatial_feat_dict = self.spatial_ft.compute_all(Y_norm, azi_rad, ele_rad)
            spatial_feat_list = []
            for name, feat in spatial_feat_dict.items():
                if name in self.spatial_configs["features"]:
                    spatial_feat_list.append(feat)
            
            spatial_feat = torch.cat(spatial_feat_list, dim=1)
            subband_spatial = self.band_split(spatial_feat)
            # Projection: (B, Nband, Dim, T)
            spatial_emb = self.spatial_norm(subband_spatial)
            input_emb = spec_emb + spatial_emb
        else:
            # Fallback for no spatial features
            input_emb = spec_emb
            
        # --- 5. Backbone Separation ---
        sep_output = self.separator(input_emb)
        
        # --- 6. Masking ---
        subband_mix_spec = self.band_split(Y[:, 0]) 
        
        est_spec_RI = self.band_masker(sep_output, subband_mix_spec)
        
        # --- 7. Reconstruction ---
        est_complex = torch.complex(est_spec_RI[:, 0], est_spec_RI[:, 1])
        est_complex = est_complex * ref_mag_mean.unsqueeze(1)
        
        est_wav = torch.istft(
            est_complex.squeeze(1),
            n_fft=self.win,
            hop_length=self.stride,
            window=self.window.to(mix.device),
            length=T_wav
        )
        
        return est_wav.unsqueeze(1)