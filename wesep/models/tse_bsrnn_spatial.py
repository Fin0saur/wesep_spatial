import torch
import torch.nn as nn
import numpy as np
from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update

class TSE_BSRNN_SPATIAL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # --- 1. top model setting ---
        self.full_input = config.get("full_input",True)
        
        # --- 2. Merge Configs ---
        sep_configs = dict(
            sr=16000,
            win=512,
            stride=128,
            feature_dim=128,
            num_repeat=6,
            causal=False,
            nspk=1,  # For Separation (multiple output)
            spec_dim=2,  # For TSE feature, used in self.subband_norm
        )
        sep_configs = {**sep_configs, **config['separator']}
        spatial_configs = {
            "geometry": {
                "n_fft": 512,              
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
                "ipd": {"enabled": False},
                "cdf": {"enabled": False},
                "sdf": {"enabled": False},
                "delta_stft": {"enabled": False},
                "cyc_doaemb":{
                    "enabled": False,
                    "cyc_alpha": 20,
                    "cyc_dimension": 40,
                    "use_ele": True,
                    "out_channel": 1, # only use when concat
                    "fusion_type": "multiply" # concat or multiply
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        # ===== Separator Loading =====
        n_pairs = len(self.spatial_configs['pairs'])
        if self.full_input:
            sep_configs["spec_dim"] = 2 * len(self.spatial_configs['geometry']['mic_coords'])
        if self.spatial_configs["features"]["ipd"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs
        if self.spatial_configs["features"]["cdf"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs
        if self.spatial_configs["features"]["sdf"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs
        if self.spatial_configs["features"]["delta_stft"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs
        if self.spatial_configs["features"]["cyc_doaemb"]["enabled"]:
            self.spatial_configs['features']['cyc_doaemb']['encoder_kwargs']['out_channel'] = sep_configs["feature_dim"] # dim_hidden    
            self.spatial_configs['features']['cyc_doaemb']['num_encoder'] = 1
            
        self.sep_model = BSRNN(**sep_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)
    def forward(self, mix, cue):
        """
        mix: (B, M, T_wav)
        cue: (B, 2) -> [azimuth, elevation] (弧度)
        """
        spatial_cue = cue[0]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1] 
        
        spec = self.sep_model.stft(mix)[-1]
        
        if self.full_input:
            spec_feat = torch.cat([spec.real, spec.imag], dim=1)
        else :    
            spec_feat = torch.stack([spec[:, 0].real, spec[:, 0].imag], dim=1)
        
        # spatial_feat_dict = self.spatial_ft.compute_all(spec, azi_rad, ele_rad)
        
        #################################################################
        # Spatio-temporal Features
        if self.spatial_configs['features']['ipd']['enabled'] :
            ipd_feature = self.spatial_ft.features['ipd'].compute(spec)
            spec_feat = self.spatial_ft.features['ipd'].post(spec_feat,ipd_feature)
            # spec_feat=self.spatial_ft.features['ipd'].post(spec_feat,spatial_feat_dict['ipd']) # if use compute_all
        
        if self.spatial_configs['features']['cdf']['enabled'] :
            cdf_feature = self.spatial_ft.features['cdf'].compute(spec,azi_rad,ele_rad)
            spec_feat = self.spatial_ft.features['cdf'].post(spec_feat,cdf_feature)
            # spec_feat=self.spatial_ft.features['cdf'].post(spec_feat,spatial_feat_dict['cdf'])
        
        if self.spatial_configs['features']['sdf']['enabled']:
            sdf_feature = self.spatial_ft.features['sdf'].compute(spec,azi_rad,ele_rad)
            spec_feat = self.spatial_ft.features['sdf'].post(spec_feat,sdf_feature)
            # spec_feat=self.spatial_ft.features['sdf'].post(spec_feat,spatial_feat_dict['sdf'])
        
        if self.spatial_configs['features']['delta_stft']['enabled']:
            dstft_feature = self.spatial_ft.features['delta_stft'].compute(spec)
            spec_feat = self.spatial_ft.features['delta_stft'].post(spec_feat,dstft_feature)
            # spec_feat=self.spatial_ft.features['delta_stft'].post(spec_feat,spatial_feat_dict['delta_stft'])
        
        subband_spec = self.sep_model.band_split(spec_feat)
        
        subband_mix_spec = self.sep_model.band_split(spec[:,0])
        
        subband_feature = self.sep_model.subband_norm(subband_spec)
               
        if self.spatial_configs['features']['cyc_doaemb']['enabled']:
            cyc_doaemb = self.spatial_ft.features['cyc_doaemb'].compute(azi_rad,ele_rad)
            subband_feature=self.spatial_ft.features['cyc_doaemb'].post(subband_feature.permute(0,2,1,3),cyc_doaemb).permute(0,2,1,3)     
        
        sep_out=self.sep_model.separator(subband_feature) 
        
        est_spec_RI = self.sep_model.band_masker(sep_out, subband_mix_spec)
        
        est_complex = torch.complex(est_spec_RI[:, 0], est_spec_RI[:, 1])
        # est_complex = est_complex * ref_mag_mean.unsqueeze(1)
        est_wav = self.sep_model.istft(est_complex)
        return est_wav