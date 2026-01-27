import torch
import torch.nn as nn
from .backbones import ResNet50Backbone, ResNet18CUBBackbone, CLIPViTBackbone


class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_hidden=0, bias=False, device="cuda"):
        super().__init__()
        layers = []
        if num_hidden == 0:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
        else:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            for _ in range(num_hidden):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(out_features, out_features, bias=bias))
        
        self.model = nn.Sequential(*layers).to(device)
        
    def forward(self, x):
        return self.model(x)


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std, device="cuda"):
        super().__init__()
        self.register_buffer('mean', mean.to(device))
        self.register_buffer('std', std.to(device))
        
    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)


class FinalLayer(nn.Module):
    def __init__(self, in_features, out_features, device="cuda"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True).to(device)
        
    def forward(self, x):
        return self.linear(x)


class FedLFC_CBM(nn.Module):
    def __init__(
        self,
        backbone_type="resnet50",
        clip_name="ViT-B/16",
        num_concepts=100,
        num_classes=2,
        backbone_output_dim=None,
        use_clip_penultimate=False,
        proj_hidden_layers=0,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        
        if backbone_type == "resnet50":
            self.backbone = ResNet50Backbone(device=device)
            backbone_dim = self.backbone.output_dim
        elif backbone_type == "resnet18_cub":
            self.backbone = ResNet18CUBBackbone(device=device)
            backbone_dim = self.backbone.output_dim
        elif backbone_type.startswith("clip_"):
            clip_model_name = backbone_type[5:] if backbone_type.startswith("clip_") else clip_name
            self.backbone = CLIPViTBackbone(
                clip_name=clip_model_name,
                device=device,
                use_penultimate=use_clip_penultimate
            )
            backbone_dim = self.backbone.output_dim
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        if backbone_output_dim is not None:
            backbone_dim = backbone_output_dim
            
        self.projection = ProjectionLayer(
            backbone_dim,
            num_concepts,
            num_hidden=proj_hidden_layers,
            bias=False,
            device=device
        )
        
        self.normalization = None
        
        self.final_layer = FinalLayer(num_concepts, num_classes, device=device)
        
    def set_normalization(self, mean, std):
        self.normalization = NormalizationLayer(mean, std, device=self.device)
        
    def forward(self, x, return_concepts=False):
        features = self.backbone(x)
        concept_logits = self.projection(features)
        
        if self.normalization is not None:
            concept_features = self.normalization(concept_logits)
        else:
            concept_features = concept_logits
            
        class_logits = self.final_layer(concept_features)
        
        if return_concepts:
            return class_logits, concept_features
        return class_logits
    
    def get_projection_params(self):
        return list(self.projection.parameters())
    
    def get_final_params(self):
        return list(self.final_layer.parameters())
    
    def get_all_params(self):
        return list(self.parameters())
