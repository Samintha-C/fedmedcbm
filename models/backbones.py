import os
import sys
import torch
import torch.nn as nn

lf_cbm_path = os.path.join(os.path.dirname(__file__), '../../Label-free-CBM')
if os.path.exists(lf_cbm_path):
    sys.path.insert(0, lf_cbm_path)
    from data_utils import get_target_model
else:
    raise ImportError(f"Label-free-CBM not found at {lf_cbm_path}. Please ensure Label-free-CBM is in the parent directory.")


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
