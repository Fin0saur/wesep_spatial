# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from wesep.modules.speaker.spk_frontend import SpeakerFrontend
from wesep.modules.spatial.spatial_frontend import SpatialFrontend
from wesep.modules.separator.bsrnn import BSRNN
from wesep.modules.common.deep_update import deep_update


class TSE_BSRNN_SPK_SPATIAL(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.full_input = config.get("full_input",True)
        
        # ===== Merge configs =====
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
                "hop_length": 128,
                "win_length": 512,
                "fs": 16000,
                "c": 343.0,
                "mic_spacing": 0.033333,
                "mic_coords": [
                    [-0.05,        0.0, 0.0],  # Mic 0
                    [-0.01666667,  0.0, 0.0],  # Mic 1
                    [ 0.01666667,  0.0, 0.0],  # Mic 2
                    [ 0.05,        0.0, 0.0],  # Mic 3
                ],
            },
            "pairs": [[0, 1], [1, 2], [2, 3], [0, 3]], 
            "features": {
                "ipd": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "cdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "sdf": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "delta_stft": {
                    "enabled": False, 
                    "num_encoder": 1
                },
                "Multiply_emb": {
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "cyc",
                        "cyc_alpha": 20,
                        "cyc_dimension": 40,
                    },
                    "use_ele": True,
                    "out_channel": 1
                },
                "InitStates_emb": {  
                    "enabled": False,
                    "num_encoder": 1,
                    "encoding_config":{
                        "encoding": "oh",
                        "emb_dim": 180,
                    },
                    "hidden_size_f": 256,
                    "hidden_size_t": 256,
                    "use_ele" : True
                }
            }
        }
        self.spatial_configs = deep_update(spatial_configs, config.get('spatial', {}))
        spk_configs = {
            "features": {
                "listen": {
                    "enabled": False,
                    "win": sep_configs["win"],
                    "hop": sep_configs["stride"],
                },
                "usef": {
                    "enabled": False,
                    "causal": sep_configs["causal"],
                    "enc_dim": sep_configs["win"] // 2 + 1,
                    "emb_dim": sep_configs["feature_dim"] // 2,
                },
                "tfmap": {
                    "enabled": False
                },
                "context": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"],
                    "atten_dim": sep_configs["feature_dim"]
                },
                "spkemb": {
                    "enabled": False,
                    "mix_dim": sep_configs["feature_dim"]
                },
            },
            "speaker_model": {
                "fbank": {
                    "sample_rate": sep_configs["sr"]
                },
            },
        }
        self.spk_configs = deep_update(spk_configs, config['speaker'])
        if self.full_input:
            sep_configs["spec_dim"] = 2 * len(self.spatial_configs['geometry']['mic_coords'])
        # ===== Separator Loading =====
        if self.spk_configs["features"]["usef"]["enabled"]:
            sep_configs["spec_dim"] += self.spk_configs["features"]["usef"][
                "emb_dim"] * 2
        if self.spk_configs["features"]["tfmap"]["enabled"]:
            sep_configs["spec_dim"] = sep_configs["spec_dim"] + 1  #
        n_pairs = len(self.spatial_configs['pairs'])
        if self.spatial_configs["features"]["ipd"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs * self.spatial_configs["features"]["ipd"]["num_encoder"]
        if self.spatial_configs["features"]["cdf"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs * self.spatial_configs["features"]["cdf"]["num_encoder"]
        if self.spatial_configs["features"]["sdf"]["enabled"]:
            sep_configs["spec_dim"] += n_pairs * self.spatial_configs["features"]["sdf"]["num_encoder"]
        if self.spatial_configs["features"]["delta_stft"]["enabled"]:
            sep_configs["spec_dim"] += 2 * n_pairs * self.spatial_configs["features"]["delta_stft"]["num_encoder"]
        if self.spatial_configs["features"]["Multiply_emb"]["enabled"]:
            self.spatial_configs['features']['Multiply_emb']['out_channel'] = sep_configs["feature_dim"] # dim_hidden    
            self.spatial_configs['features']['Multiply_emb']['num_encoder'] = 1
        if self.spatial_configs["features"]["InitStates_emb"]["enabled"]:
            self.spatial_configs["features"]["InitStates_emb"]["hidden_size_f"] = sep_configs["feature_dim"] * 2
            self.spatial_configs["features"]["InitStates_emb"]["hidden_size_t"] = sep_configs["feature_dim"] * 2
            self.spatial_configs["features"]["InitStates_emb"]["num_encoder"] = sep_configs["num_repeat"]
        self.sep_model = BSRNN(**sep_configs)
        # ===== Speaker Loading =====
        if self.spk_configs["features"]["context"]["enabled"]:
            self.spk_configs["features"]["context"][
                "band"] = self.sep_model.nband  #
        self.spk_ft = SpeakerFrontend(self.spk_configs)
        self.spatial_ft = SpatialFrontend(self.spatial_configs)

    def forward(self, mix, cue):
        """
        Args:
            mix:  Tensor [B, M, T]
            cue: list[Tensor]
                 cue[0]: enroll Tensor [B, 1, T]
                 cue[1]: spatial_cue Tensor [B, 2] -> [azimuth, elevation] (弧度)
        """
        enroll = cue[0].squeeze(1)
        spatial_cue = cue[1]
        azi_rad = spatial_cue[:, 0]
        ele_rad = spatial_cue[:, 1]
        
        ##### Cue of the target speaker
        wav_enroll = enroll[..., 0] # only ref channel
        
        ###### Extraction with speaker cue
        wav_mix = mix
        
        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            B, M, T = wav_mix.shape
            processed_channels = []
            for m in range(M):
                processed_ch = self.spk_ft.listen.compute(wav_enroll, wav_mix[:, m, :])
                processed_channels.append(processed_ch)
                
            wav_mix = torch.stack(processed_channels, dim=1)
            
        ###########################################################
        # S1. Convert into frequency-domain
        spec = self.sep_model.stft(wav_mix)[-1]
        
        # S2. Concat real and imag, split to subbands
        spec_RI = None
        spec_RI_single = torch.stack([spec[:, 0].real, spec[:, 0].imag], 1)
        
        if self.full_input:
            spec_RI = torch.cat([spec.real, spec.imag], dim=1)
        else :
            spec_RI = spec_RI_single  # (B, 2, F, T)
        
        ###########################################################
        # C1. Feature: usef
        if self.spk_configs['features']['usef']['enabled']:
            enroll_spec = self.sep_model.stft(wav_enroll)[-1]  # (B, F, T_e) complex
            enroll_spec = torch.stack([enroll_spec.real, enroll_spec.imag], 1)  # (B, 2, F, T)
            enroll_usef, mix_usef = self.spk_ft.usef.compute(enroll_spec, spec_RI_single)  
            usef_feat = self.spk_ft.usef.post(mix_usef, enroll_usef) 
            spec_RI = torch.cat([spec_RI, usef_feat], dim=1)
            
        # C2. Feature: tfmap
        if self.spk_configs['features']['tfmap']['enabled']:
            enroll_mag = self.sep_model.stft(wav_enroll)[0]  
            enroll_tfmap = self.spk_ft.tfmap.compute(enroll_mag, torch.abs(spec[:,0])) 
            spec_RI = self.spk_ft.tfmap.post(spec_RI, enroll_tfmap.unsqueeze(1)) 
            
        ###########################################################
        # Early Spatial Features
        if self.spatial_configs['features']['ipd']['enabled']:
            ipd_feature = self.spatial_ft.features['ipd'].compute(Y=spec)
            spec_RI = self.spatial_ft.features['ipd'].post(spec_RI, ipd_feature)
        
        if self.spatial_configs['features']['cdf']['enabled']:
            cdf_feature = self.spatial_ft.features['cdf'].compute(Y=spec, azi=azi_rad, ele=ele_rad)
            spec_RI = self.spatial_ft.features['cdf'].post(spec_RI, cdf_feature)
        
        if self.spatial_configs['features']['sdf']['enabled']:
            sdf_feature = self.spatial_ft.features['sdf'].compute(Y=spec, azi=azi_rad, ele=ele_rad)
            spec_RI = self.spatial_ft.features['sdf'].post(spec_RI, sdf_feature)
        
        if self.spatial_configs['features']['delta_stft']['enabled']:
            dstft_feature = self.spatial_ft.features['delta_stft'].compute(Y=spec)
            spec_RI = self.spatial_ft.features['delta_stft'].post(spec_RI, dstft_feature)
        ###########################################################
        subband_spec = self.sep_model.band_split(
            spec_RI)  # list of (B, 2/3/2*usef.emb_dim, BW, T)
        subband_mix_spec = self.sep_model.band_split(spec[:, 0])
        # S3. Normalization and bottleneck
        subband_feature = self.sep_model.subband_norm(
            subband_spec)  # (B, nband, feat, T)
        ###########################################################
        # C3. Feature: context
        if self.spk_configs['features']['context']['enabled']:
            # C3.1 Generate the frame-level speaker embeddings
            enroll_context = self.spk_ft.context.compute(
                wav_enroll)  # (B, F_e, T_e)
            # C3.2 Fuse the frame-level speaker embeddings into the mix_repr
            subband_feature = self.spk_ft.context.post(
                subband_feature, enroll_context)  # (B, nband, feat, T)
        # C4. Feature: spkemb
        if self.spk_configs['features']['spkemb']['enabled']:
            # C4.1 Generate the speaker embedding
            enroll_emb = self.spk_ft.spkemb.compute(wav_enroll)  # (B, F_e)
            # C4.2 Fuse the speaker embeeding into the mix_repr
            enroll_emb = enroll_emb.unsqueeze(1).unsqueeze(3)  # (B, 1, F_e, 1)
            subband_feature = self.spk_ft.spkemb.post(
                subband_feature, enroll_emb)  # (B, nband, feat, T)
            
        if self.spatial_configs['features']['Multiply_emb']['enabled']:
            cyc_doaemb = self.spatial_ft.features['Multiply_emb'].compute(azi=azi_rad,ele=ele_rad)
            subband_feature=self.spatial_ft.features['Multiply_emb'].post(subband_feature.permute(0,2,1,3),cyc_doaemb).permute(0,2,1,3)     
        
        ch_result = None
        
        if self.spatial_configs['features']['InitStates_emb']['enabled']:
            ch_result = self.spatial_ft.features['InitStates_emb'].compute(azi=azi_rad,ele=ele_rad)   
        ###########################################################
        # S4. Separation
        sep_output = self.sep_model.separator(
            subband_feature, ch_result)  # (B, nband, feat, T)
        # S5. Complex Mask
        est_spec_RI = self.sep_model.band_masker(
            sep_output, subband_mix_spec)  # (B, 2, S, F, T)
        est_complex = torch.complex(est_spec_RI[:, 0],
                                    est_spec_RI[:, 1])  # (B, S, F, T)
        # S6. Back into waveform
        s = self.sep_model.istft(est_complex)  # (B, S, T)
        ###########################################################
        # C0. Feature: listen
        if self.spk_configs['features']['listen']['enabled']:
            # C0.2 Prepend the enroll to the mix in the beginning
            s = self.spk_ft.listen.post(s)  # (B, T)
        ###########################################################
        return s

