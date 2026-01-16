import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Label-free-CBM'))
from data_utils import get_target_model


class ResNet50Backbone(nn.Module):
    def __init__(self, device="cuda", pretrained=True):
        super().__init__()
        target_model, preprocess = get_target_model("resnet50", device=device)
        
        if isinstance(target_model, nn.Sequential):
            self.backbone = target_model
        else:
            self.backbone = nn.Sequential(*list(target_model.children())[:-1])
        
        self.preprocess = preprocess
        self.output_dim = 2048
        self.eval()
        
    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        return x


class CLIPViTBackbone(nn.Module):
    def __init__(self, clip_name="ViT-B/16", device="cuda", use_penultimate=False):
        super().__init__()
        import clip
        
        model, preprocess = clip.load(clip_name, device=device)
        self.preprocess = preprocess
        
        if use_penultimate:
            model = model.visual
            N = model.attnpool.c_proj.in_features
            identity = nn.Linear(N, N, dtype=torch.float16, device=device)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            model.attnpool.c_proj = identity
            self.output_dim = N
        else:
            model = model.visual
            self.output_dim = model.attnpool.c_proj.out_features
            
        self.backbone = model.float()
        self.eval()
        
    def forward(self, x):
        output = self.backbone(x).float()
        return output
