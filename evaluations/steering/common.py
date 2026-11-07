"""Shared pieces of the steering evaluation (Sec. 4.7, App. "Steering Details").

A data root holds everything the scripts read, laid out as pull_inputs.py writes it:
  <data>/<ds>/                 backbone feature caches and DINO label caches, named as on the PVC
  <data>/saved/<ds>/s<seed>/   one seed's run files (cbl.pt, normalization, args.txt, concepts.txt, ...)
                               and its OURS_HEAD_SAVED heads.pt + metrics.json
with ds in c10, c100, places365, imagenet and seed in 42, 31, 7.

CIFAR-10 / CIFAR-100 use each seed's full training split and its validation split. Places365 and
ImageNet keep their full training caches (5-7 GB per seed) on the PVC: the validation portion of the
seed's training split (never test) is split 90/10 into judge train / judge val and also serves for the
calibration.
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "baselines", "fedcbm_reimpl"))

SEEDS = (42, 31, 7)
# train / val split hashes of the training split per seed (CIFAR; the split depends on the seed only)
SPLIT = {42: ("6727ca6a7fab", "cb73e6a158ed"), 31: ("9b5a53e0311a", "382653daa70b"), 7: ("67086b2077af", "7a1f795f5cdb")}
# (feature cache, DINO label cache) of the training split, as named on the PVC
CACHE = {"c10": ("c10/cifar10_train_clip_RN50_feat_s{}.pt", "c10/cifar10_train_conf0.100_c76bd6855f59c_s{}.pt"),
         "c100": ("c100/cifar100_train_resnet50_feat_s{}.pt", "c100/cifar100_train_conf0.100_cca53bfe901a6_s{}.pt"),
         "places365": ("places365/places365_train_resnet50_feat_s{}.pt", "places365/places365_train_conf0.100_ced98e1ece293_s{}.pt"),
         "imagenet": ("imagenet/imagenet_train_resnet50_feat_s{}.pt", "imagenet/imagenet_train_conf0.100_c5abf7981ace8_s{}.pt")}
# the validation portion of each seed's training split, Places365 / ImageNet (checked against the
# run's val_concept_labels.pt in load_seed)
VAL = {"places365": {42: "0ddda77d387b", 31: "3f92329b62d7", 7: "70482a684a29"},
       "imagenet": {42: "6f3ec0c148a9", 31: "6dae12c3f1c9", 7: "c4c1ab2d7d51"}}
TEST = {"c10": "c10/cifar10_val_clip_RN50_feat_s552aae514028.pt",
        "c100": "c100/cifar100_val_resnet50_feat_s552aae514028.pt",
        "places365": "places365/places365_val_resnet50_feat_s67f2a42edd86.pt",
        "imagenet": "imagenet/imagenet_val_resnet50_feat_s2c184805042d.pt"}
# test DINO labels over the full concept list (within_class_auc.py)
TEST_ANN = {"c10": "c10/cifar10_val_conf0.100_c76bd6855f59c_s552aae514028.pt",
            "c100": "c100/cifar100_val_conf0.100_cca53bfe901a6_s552aae514028.pt"}


def nec(W):
    return float((W.abs() > 1e-5).sum()) / W.shape[0]


def split_90_10(n, frac=0.9):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    return perm[:int(frac * n)], perm[int(frac * n):]


def format_concept(s):
    """data_utils.format_concept (copied: importing data_utils pulls in clip)."""
    s = s.lower()
    s = s.replace("-", " ").replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ")
    if s.startswith("a "):
        s = s[2:]
    elif s.startswith("an "):
        s = s[3:]
    return " ".join(s.split())


def kept_by_name(run, n_cols):
    """Columns of a full-concept-list label cache that the run kept, rebuilt from data_utils.get_concepts's
    column order minus removed_concepts.txt (the zero-count filter differs by seed)."""
    args = json.load(open(os.path.join(run, "args.txt")))
    lines = open(os.path.join(REPO, args["concept_file"])).read().splitlines()
    full = list(dict.fromkeys(format_concept(c.strip()) for c in lines if c.strip()))
    removed = {l.strip() for l in open(os.path.join(run, "removed_concepts.txt")) if l.strip()}
    kept = [l.strip() for l in open(os.path.join(run, "concepts.txt")) if l.strip()]
    assert len(full) == n_cols, (len(full), n_cols)
    keep = torch.tensor([c not in removed for c in full])
    assert [c for c, k in zip(full, keep.tolist()) if k] == kept
    return keep


def load_seed(data, ds, seed):
    """One seed's concept layer, saved heads, logged metrics, features (train, val, test) and training
    DINO labels on the kept concepts."""
    run = os.path.join(data, "saved", ds, f"s{seed}")
    t = lambda f: torch.load(os.path.join(run, f), map_location="cpu", weights_only=False)  # noqa: E731
    cbl = t("cbl.pt")
    L = {"Wc": cbl["model.0.weight"].float(), "bc": cbl["model.0.bias"].float(),
         "mean": t("train_concept_features_mean.pt").float(), "std": t("train_concept_features_std.pt").float()}
    heads = t("heads.pt")["dirichlet"][5]
    metrics = json.load(open(os.path.join(run, "metrics.json")))["partitions"]["dirichlet"]["targets"]["5"]
    load = lambda p: torch.load(os.path.join(data, p), map_location="cpu")  # noqa: E731
    c = load(TEST[ds])
    f = {"test": (c["features"].float(), c["targets"].long())}
    if ds in VAL:
        cv = load(CACHE[ds][0].format(VAL[ds][seed]))
        a = load(CACHE[ds][1].format(VAL[ds][seed]))
        keep = kept_by_name(run, a.shape[1])
        # val_concept_labels.pt holds the class label per row, in the seed's split order
        assert torch.equal(cv["targets"].long(), t("val_concept_labels.pt").long()), "cache is not this seed's validation portion"
        tr, va = split_90_10(len(cv["targets"]))
        f["train"] = (cv["features"][tr].float(), cv["targets"][tr].long())
        f["val"] = (cv["features"][va].float(), cv["targets"][va].long())
        lab = a[tr][:, keep]
    else:
        ctr, cva = (load(CACHE[ds][0].format(h)) for h in SPLIT[seed])
        f["train"] = (ctr["features"].float(), ctr["targets"].long())
        f["val"] = (cva["features"].float(), cva["targets"].long())
        a = load(CACHE[ds][1].format(SPLIT[seed][0]))
        lab = a[:, a.sum(0) > 0]
    assert lab.shape == (len(f["train"][1]), L["Wc"].shape[0]), (lab.shape, L["Wc"].shape)
    return L, heads, metrics, f, lab


def layer_concepts(X, L):
    """z-scored concept activations of a linear concept layer L = {Wc, bc, mean, std}."""
    return (X.float() @ L["Wc"].T + L["bc"] - L["mean"]) / L["std"]


def calib(a_tr, lab):
    """mu+ / mu-: mean training activation where the DINO label is present / absent; a concept with
    fewer than five positives takes its 99th percentile as mu+."""
    lab = lab.float()
    npos = lab.sum(0)
    on = torch.where(npos < 5, torch.quantile(a_tr, 0.99, dim=0), (a_tr * lab).sum(0) / npos.clamp(min=1))
    off = (a_tr * (1 - lab)).sum(0) / (1 - lab).sum(0).clamp(min=1)
    return on, off


def edit(h, rows, target_raw):
    """Minimal-norm dh with rows @ (h + dh) = target_raw (rows [k, d], h [n, d], target_raw [n, k])."""
    r = target_raw - h @ rows.T
    return (r @ torch.linalg.pinv(rows @ rows.T)) @ rows


def concept_edit(h, y, yp, L, W, on, off, scale):
    """Signed edit toward target yp: its positive-weight concepts to mu+, its negative-weight concepts
    and the source class's positive-only concepts to mu-, a fraction `scale` of the way."""
    raw = lambda a, idx: a * L["std"][idx] + L["mean"][idx] - L["bc"][idx]  # noqa: E731
    dh = torch.zeros_like(h)
    for k in yp.unique().tolist():
        for src in y[yp == k].unique().tolist():
            sub = ((yp == k) & (y == src)).nonzero().squeeze(1)
            S = (W[k] > 1e-5).nonzero().squeeze(1)
            So = (((W[src] > 1e-5) & (W[k] <= 1e-5)) | (W[k] < -1e-5)).nonzero().squeeze(1)
            cols = torch.cat([S, So])
            if cols.numel() == 0:
                continue
            target = torch.cat([on[S].expand(len(sub), -1), off[So].expand(len(sub), -1)], 1)
            cur = layer_concepts(h[sub], L)[:, cols]
            dh[sub] = edit(h[sub], L["Wc"][cols], raw(cur + scale * (target - cur), cols))
    return dh


def draw(valid, avoid, g):
    """One class per row from valid, different from every column of avoid [n, j]."""
    out = valid[torch.randint(len(valid), (len(avoid),), generator=g)]
    for _ in range(100):
        bad = (out[:, None] == avoid).any(1)
        if not bad.any():
            return out
        out[bad] = valid[torch.randint(len(valid), (int(bad.sum()),), generator=g)]
    raise RuntimeError("could not draw distinct classes")


def train_judge(kind, X, y, Xv, yv, nk, epochs=30, seed=0):
    """The independent judge: a linear or one-hidden-layer MLP probe on backbone features, trained on
    class labels only, best epoch on validation."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    d = X.shape[1]
    net = torch.nn.Linear(d, nk) if kind == "linear" else torch.nn.Sequential(
        torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Linear(512, nk))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_acc = None, -1
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 512):
            ix = perm[s:s + 512]
            loss = torch.nn.functional.cross_entropy(net((X[ix] - mu) / sd), y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = float((net((Xv - mu) / sd).argmax(1) == yv).float().mean())
        if acc > best_acc:
            best_acc, best = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best)
    net.eval()
    judge = lambda F: net((F - mu) / sd)  # noqa: E731
    judge.net, judge.mu, judge.sd = net, mu, sd
    return judge, best_acc


# FedDualAvg as the sanity check ran it: baselines/fedcbm_reimpl/ours_head.py:feddualavg, returning
# (W, b, round) and selecting the round on plain validation accuracy.
def soft(z, t):
    return torch.sign(z) * torch.clamp(z.abs() - t, min=0.0)


def acc(w, b, z, y):
    return float(((z @ w.T + b).argmax(1) == y).float().mean())


def feddualavg(z, y, zv, yv, nk, lam, clients, rounds=200, epochs=3, bs=512, eta_c=0.01, eta_s=1.0):
    dev = z.device
    g = torch.Generator(device="cpu").manual_seed(0)
    n = float(sum(len(c) for c in clients))
    wts = [len(c) / n for c in clients]
    zw, zb = torch.zeros(nk, z.shape[1], device=dev), torch.zeros(nk, device=dev)
    K = epochs * -(-len(clients[0]) // bs)
    best, best_acc, best_nnz = None, -1.0, None
    for r in range(rounds):
        dw, db = torch.zeros_like(zw), torch.zeros_like(zb)
        for ci, ix in enumerate(clients):
            lw, lb, step = zw.clone(), zb.clone(), 0
            for _ in range(epochs):
                perm = ix[torch.randperm(len(ix), generator=g).to(dev)]
                for s in range(0, len(perm), bs):
                    b_ix = perm[s:s + bs]
                    w = soft(lw, eta_s * eta_c * (r * K + step + 1) * lam)
                    zb_ = z[b_ix].float()
                    P = torch.softmax(zb_ @ w.T + lb, 1)
                    P[torch.arange(len(b_ix), device=dev), y[b_ix]] -= 1.0
                    lw -= eta_c * (P.T @ zb_ / len(b_ix))
                    lb -= eta_c * P.mean(0)
                    step += 1
            dw += wts[ci] * (lw - zw)
            db += wts[ci] * (lb - zb)
        zw += eta_s * dw
        zb += eta_s * db
        w_s = soft(zw, eta_s * eta_c * (r + 1) * K * lam)
        a = acc(w_s, zb, zv, yv)
        nnz = int((w_s.abs() > 1e-5).sum())
        if a > best_acc or (a == best_acc and best_nnz is not None and nnz < best_nnz):
            best, best_acc, best_nnz = (w_s.clone(), zb.clone(), r + 1), a, nnz
    return best, best_acc, {}
