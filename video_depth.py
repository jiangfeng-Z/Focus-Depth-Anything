# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from Video_Depth_Anything_FT.video_depth_anything.dinov2 import DINOv2
from Video_Depth_Anything_FT.video_depth_anything.dpt_temporal import DPTHeadTemporal
# from util.transform import Resize, NormalizeImage, PrepareForNet
# from Video_Depth_Anything_FT.f_utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward_focus_dists(self, depth, focus_dists):
        depth = F.softplus(depth) + 1e-6
        depth = depth / depth.sum(axis=1, keepdim=True)
        depth = torch.sum(focus_dists * depth, dim=1)
        return depth

    def forward(self, x, focus_dists):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)

        depth = self.head(features, patch_h, patch_w, T)


        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = depth.squeeze(1).unflatten(0, (B, T))
        depth = self.forward_focus_dists(depth, focus_dists)
        return depth


