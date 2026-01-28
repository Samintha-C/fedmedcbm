import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from torch.utils.data import DataLoader, TensorDataset

from glm_saga.elasticnet import glm_saga
from utils.weight_utils import weight_truncation

MAX_GLM_STEP = 150
GLM_STEP_SIZE = 2 ** 0.1


def measure_acc(
    num_concepts,
    num_classes,
    num_samples,
    train_loader,
    val_loader,
    test_concept_loader,
    saga_step_size=0.1,
    saga_n_iters=500,
    device="cuda",
    max_lam=0.01,
    measure_level=(5, 10, 15, 20, 25, 30),
):
    linear = torch.nn.Linear(num_concepts, num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = max_lam
    max_sparsity = measure_level[-1] / num_concepts
    output_proj = glm_saga(
        linear, train_loader, saga_step_size, saga_n_iters, ALPHA,
        k=MAX_GLM_STEP,
        epsilon=1 / (GLM_STEP_SIZE ** MAX_GLM_STEP),
        val_loader=val_loader,
        test_loader=test_concept_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=num_samples,
        n_classes=num_classes,
        max_sparsity=max_sparsity,
    )
    path = output_proj["path"]
    sparsity_list = [(params["weight"].abs() > 1e-5).float().mean().item() for params in path]

    final_layer = torch.nn.Linear(num_concepts, num_classes)
    accs = []
    weights = []
    for eff_concept_num in measure_level:
        target_sparsity = eff_concept_num / num_concepts
        i = 0
        for i, sparsity in enumerate(sparsity_list):
            if sparsity >= target_sparsity:
                break
        i = min(i, len(path) - 1)
        params = path[i]
        W_g, b_g, sparsity = params["weight"], params["bias"], sparsity_list[i]
        lam = params["lam"]
        print(eff_concept_num, lam, sparsity)
        print(f"Num of effective concept: {eff_concept_num}. Choose lambda={lam:.6f} with sparsity {sparsity:.4f}")
        W_g_trunc = weight_truncation(W_g, target_sparsity)
        weight_contribs = torch.sum(torch.abs(W_g_trunc), dim=0)
        print("Num concepts with outgoing weights:{}/{}".format(torch.sum(weight_contribs > 1e-5).item(), len(weight_contribs)))
        print(target_sparsity, (W_g_trunc.abs() > 0).sum().item())
        final_layer.load_state_dict({"weight": W_g_trunc, "bias": b_g})
        final_layer = final_layer.to(device)
        weights.append((W_g_trunc, b_g))
        correct = []
        for x, y in test_concept_loader:
            x, y = x.to(device), y.to(device)
            pred = final_layer(x).argmax(dim=-1)
            correct.append(pred == y)
        correct = torch.cat(correct)
        accs.append(correct.float().mean().item())
        print(f"Test Acc: {correct.float().mean():.4f}")
    print(f"Average acc: {sum(accs) / len(accs):.4f}")
    return path, {nec: w for nec, w in zip(measure_level, weights)}, accs
