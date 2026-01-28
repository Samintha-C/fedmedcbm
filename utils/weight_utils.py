import torch


def weight_truncation(weight: torch.Tensor, sparsity: float):
    numel = weight.numel()
    num_zeros = int((1 - sparsity) * numel)
    num_zeros = min(max(0, num_zeros), numel - 1)
    threshold = torch.sort(weight.flatten().abs())[0][num_zeros]
    sparse_weight = weight.clone().detach()
    sparse_weight[weight.abs() < threshold] = 0
    return sparse_weight
