# Copyright (c) 2025 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
#
# Description: wesep v2 network component.

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.deep_update import deep_update
from wesep.modules.fusion.speech import SpeakerFuseLayer
from wesep.modules.visual.muse_visual_frontend import Muse_LipROIProcessor, Muse_VisualFrontend, Muse_load_model, Muse_VisualConv1D


class BaseVisualFeature(nn.Module):

    def compute(self, video, mix=None):
        raise NotImplementedError

    def post(self, mix_repr, feat_repr):
        return mix_repr


class MuseVisualFeature(BaseVisualFeature):

    def __init__(self, config):
        super().__init__()

        self.roi = Muse_LipROIProcessor()
        self.visual_frontend = Muse_VisualFrontend()

        if config["vf_pretrained"] is not None:
            self.visual_frontend = Muse_load_model(
                self.visual_frontend,
                config["vf_pretrained"],
            )
            for key, param in self.visual_frontend.named_parameters():
                param.requires_grad = False

        ve_blocks = []
        for x in range(config.get("vtcn_layers", 5)):
            ve_blocks += [Muse_VisualConv1D(channels=config["vtcn_channels"])]
        self.vtcn = nn.Sequential(*ve_blocks)

        self.upsample_to_audio = config.get("upsample", False)

        self.fusionLayer = SpeakerFuseLayer(
            embed_dim=config["vtcn_channels"],
            feat_dim=config["mix_dim"],
            fuse_type=config["fusion"],
        )

    def compute(self, video, mix=None):
        """
        video: (B, H, W, 3, T_v)
        return:
            muse_visual: (B, 512, T_v)
        """

        feat = self.roi(video)  # # (B, T, 112,112)
        feat = feat.transpose(0, 1)
        feat = feat.unsqueeze(2)  # (T, B, 1,112,112)
        with torch.no_grad():
            feat = self.visual_frontend(feat)  # (B, 512, T_v)
        feat = self.vtcn(feat)  # (B, 512, T_v)

        if self.upsample_to_audio and mix is not None:
            T_audio = mix.shape[-1]
            feat = F.interpolate(feat, size=T_audio,
                                 mode="linear")  # (B, 512, T)

        return feat

    def post(self, mix_repr, feat_repr):
        return self.fusionLayer(mix_repr, feat_repr)  # or concat, configurable


class VisualFrontend(nn.Module):

    def __init__(self, config):
        super().__init__()

        DEFAULT_CONFIG = {
            "features": {
                "muse_visual": {
                    "enabled": True,
                    "vf_pretrained": "./pretrain_networks/visual_frontend.pt",
                    "vtcn_channels": 512,
                    "vtcn_layers": 5,
                    "upsample": False,
                    "mix_dim": 128,
                    "fusion": "concat",
                }
            }
        }

        self.config = deep_update(DEFAULT_CONFIG, config)
        feats = self.config["features"]

        if feats["muse_visual"]["enabled"]:
            self.muse_visual = MuseVisualFeature(feats["muse_visual"])

    def compute_all(self, enroll, mix=None):
        out = {}
        for name, module in self.features.items():
            out[name] = module.compute(enroll, mix)
        return out
