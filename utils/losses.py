import torch
import torch.nn as nn


def cosine_similarity_loss(pred, target):
    pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    return -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))


def cosine_similarity_cubed_loss(pred, target):
    pred_centered = pred - torch.mean(pred, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    
    pred_cubed = pred_centered ** 3
    target_cubed = target_centered ** 3
    
    pred_norm = pred_cubed / (pred_cubed.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm = target_cubed / (target_cubed.norm(dim=-1, keepdim=True) + 1e-8)
    
    return -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))


class ElasticNetLoss(nn.Module):
    def __init__(self, alpha=0.5, l1_ratio=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1.0 - l1_ratio
        
    def forward(self, pred, target, model_params):
        mse_loss = nn.functional.mse_loss(pred, target)
        
        l1_reg = sum(torch.sum(torch.abs(p)) for p in model_params)
        l2_reg = sum(torch.sum(p ** 2) for p in model_params)
        
        elastic_net = self.alpha * (self.l1_ratio * l1_reg + self.l2_ratio * l2_reg)
        return mse_loss + elastic_net


class L1SparsityLoss(nn.Module):
    def __init__(self, lambda_l1=1e-4):
        super().__init__()
        self.lambda_l1 = lambda_l1
        
    def forward(self, pred, target, model_params):
        base_loss = nn.functional.mse_loss(pred, target)
        l1_reg = sum(torch.sum(torch.abs(p)) for p in model_params)
        return base_loss + self.lambda_l1 * l1_reg
