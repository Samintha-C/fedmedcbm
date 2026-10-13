"""Check core.py against the F-CM authors' FedCBM (src/fedcbm.py, commit 31cbb14).

Runs their run_fedcbm_baseline end to end on a small synthetic federated
dataset, then core.py on the same data, and compares:
  1. the concept bank W, row by row (Stage A),
  2. the projected scores,
  3. each client's head: test predictions and accuracy (Stage B),
  4. the headline number: mean client accuracy on the global test set.

No GPU or cluster data needed. Usage:
  python baselines/fedcbm_reimpl/selftest.py --fcm_repo ../F-CMs
Exit code 1 if any check fails.
"""

import argparse
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def load_reference(fcm_repo: str):
    """Import their src/fedcbm.py. It imports hydra only for an optional
    backbone path we do not use, so stub hydra if it is not installed."""
    if "hydra" not in sys.modules:
        try:
            import hydra.utils  # noqa: F401
        except ImportError:
            hydra = types.ModuleType("hydra")
            hydra.utils = types.ModuleType("hydra.utils")
            hydra.utils.instantiate = lambda *a, **k: None
            sys.modules["hydra"] = hydra
            sys.modules["hydra.utils"] = hydra.utils
    path = os.path.join(fcm_repo, "src", "fedcbm.py")
    spec = importlib.util.spec_from_file_location("fcm_fedcbm_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_data(seed: int, m: int = 4, d: int = 48, c: int = 9, k: int = 6,
              n_train: int = 1600, n_test: int = 600):
    """Concepts are noisy linear functions of the embedding; classes depend on
    concepts; clients hold skewed class mixes (some miss classes entirely).
    Concept 7 is never present and concept 8 always present, so both must stay
    untrained (w = 0), as in the reference."""
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(k, d))
    a = rng.normal(size=(d, c))

    def sample(n):
        y = rng.integers(0, k, size=n)
        h = np.abs(proto[y] + 0.7 * rng.normal(size=(n, d))).astype(np.float32)
        logits = (h - h.mean(0)) @ a + 0.5 * rng.normal(size=(n, c))
        cc = (logits > 1.0).astype(np.float32)
        cc[:, 7] = 0.0
        cc[:, 8] = 1.0
        return h, cc, y

    h, cc, y = sample(n_train)
    mix = rng.dirichlet([0.4] * m, size=k)  # class -> client shares
    owner = np.array([rng.choice(m, p=mix[t]) for t in y])
    clients = [np.where(owner == i)[0] for i in range(m)]
    ht, ct, yt = sample(n_test)
    return h, cc, y, clients, ht, ct, yt


def batches(h, cc, y, bs=256):
    return [{"x": torch.from_numpy(h[s:s + bs]), "c": torch.from_numpy(cc[s:s + bs]),
             "y": torch.from_numpy(y[s:s + bs].astype(np.float32))}
            for s in range(0, len(y), bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcm_repo", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    cli = ap.parse_args()
    ref = load_reference(cli.fcm_repo)
    settings = {"n_rounds": cli.rounds, "local_epochs": 1, "svm_lr": 0.01,
                "svm_weight_decay": 1e-4, "head_max_iter": cli.rounds,
                "svm_log_every": 0, "feature_source": "input"}
    ok = True

    for seed in cli.seeds:
        h, cc, y, clients, ht, ct, yt = make_data(seed)
        m, k = len(clients), int(y.max()) + 1
        names = [f"c{j}" for j in range(cc.shape[1])]
        cfg = types.SimpleNamespace(
            learning=types.SimpleNamespace(settings=settings),
            trainer=types.SimpleNamespace(max_epochs=cli.rounds),
            seed=seed, engine=types.SimpleNamespace(model=types.SimpleNamespace()))
        train = [batches(h[ix], cc[ix], y[ix]) for ix in clients]
        test = [batches(ht, ct, yt)]

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                rres = ref.run_fedcbm_baseline(
                    cfg=cfg, train_dataloaders=train, val_dataloaders=[None] * m,
                    test_dataloaders=test, active_client_indices=list(range(m)),
                    concept_names=names, essential_concepts=None, device="cpu")
                w_ref = np.load(os.path.join(tmp, "results", "fedcbm_cav_bank.npy"))
            finally:
                os.chdir(cwd)

        # ours, on the rows the clients hold, in client order like the reference
        rows = torch.from_numpy(np.concatenate(clients))
        feats, conc = torch.from_numpy(h), torch.from_numpy(cc)
        stage_a = core.train_cav_bank(feats, conc, rows=rows, rounds=cli.rounds, lr=0.01,
                                      l2=1e-4, device="cpu", chunk=500, log_every=0)
        w = stage_a["W"]
        d_w = float(np.abs(w.numpy() - w_ref).max() / max(np.abs(w_ref).max(), 1e-12))
        untrained_ok = bool((w[7] == 0).all() and (w[8] == 0).all()
                            and not stage_a["trainable"][7] and not stage_a["trainable"][8])

        z_ref = ref._project_embeddings(ht, w_ref)
        z_test = core.project(torch.from_numpy(ht), w)
        d_z = float(np.abs(z_test.numpy() - z_ref).max() / max(np.abs(z_ref).max(), 1e-12))

        agree, accs, ref_accs = [], [], []
        for i, ix in enumerate(clients):
            z_tr = core.project(feats[ix], w)
            head = core.fit_head(z_tr.double(), torch.from_numpy(y[ix]), C=1.0,
                                 max_iter=max(50, cli.rounds), dtype=torch.float64)
            pred = head.predict(z_test.double()).numpy()
            rhead, _ = ref._train_local_head(ref._project_embeddings(h[ix], w_ref), y[ix].astype(np.float32),
                                             seed=seed + i + 1, max_iter=cli.rounds)
            rpred = rhead.predict(z_ref)
            agree.append(float((pred == rpred).mean()))
            accs.append(float((pred == yt).mean()))
            ref_accs.append(float((rpred == yt).mean()))
        ours_mean, ref_mean = float(np.mean(accs)), float(rres["task_accuracy_weighted"])

        # 5e-4, not 1e-4: both are float32 runs of the same update, and a sample
        # whose margin sits within rounding of 1 can be counted active by one and
        # not the other, leaving a ~1e-4 step difference. Checked against a
        # float64 run (2026-10-12): on some seeds ours matches float64 to 1e-6 and
        # the reference is 5e-5 off, on others the reverse. A real bug is O(1).
        c_ok = d_w < 5e-4 and d_z < 5e-4 and untrained_ok
        b_ok = min(agree) >= 0.99 and abs(ours_mean - ref_mean) <= 0.01
        ok &= c_ok and b_ok
        print(f"seed {seed}: clients={[len(ix) for ix in clients]} classes/client="
              f"{[len(np.unique(y[ix])) for ix in clients]}/{k}")
        print(f"  Stage A  max rel |W - W_ref| = {d_w:.2e}   untrained rows zero: {untrained_ok}")
        print(f"  project  max rel |z - z_ref| = {d_z:.2e}")
        print(f"  heads    prediction agreement per client = {[round(a, 4) for a in agree]}")
        print(f"  global-test mean client accuracy: ours {ours_mean:.4f}  reference {ref_mean:.4f}"
              f"  (ref heads re-fit here: {np.mean(ref_accs):.4f})")
        print(f"  -> {'PASS' if c_ok and b_ok else 'FAIL'}")

    ok &= variant_check()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


def variant_check(seed: int = 0) -> bool:
    """The stronger variant must fix the failure the reference shows on our
    data. Synthetic data shaped like ours: non-negative pooled features with a
    large shared mean, concepts 1-6% positive. The bias-free reference then
    scores every image 'absent' (balanced accuracy ~0.5); balanced + bias +
    standardize should not."""
    rng = np.random.default_rng(seed)
    n, d, c = 4000, 64, 10
    a = rng.normal(size=(d, c))
    rates = np.linspace(0.01, 0.06, c)

    def sample(m, cut=None):
        h = np.maximum(1.5 + rng.normal(size=(m, d)), 0.0).astype(np.float32)
        logit = (h - 1.5) @ a
        if cut is None:  # each concept's own positive rate
            cut = np.array([np.quantile(logit[:, j], 1 - rates[j]) for j in range(c)])
        return h, (logit > cut).astype(np.float32), cut

    h, cc, cut = sample(n)
    ht, ct, _ = sample(2000, cut)
    feats, conc = torch.from_numpy(h), torch.from_numpy(cc)
    res = {}
    for name, kw in (("reference", {}),
                     ("variant", {"balanced": True, "bias": True, "stats": core.feature_stats(feats)})):
        out = core.train_cav_bank(feats, conc, rounds=200, lr=0.01, l2=1e-4, device="cpu",
                                  chunk=1000, log_every=0, **kw)
        z = core.project(torch.from_numpy(ht), out["W"], b=out["b"], stats=out["stats"])
        res[name] = core.concept_accuracy(z, torch.from_numpy(ct), out["trainable"])["concept_balanced_accuracy_mean"]
    good = res["reference"] < 0.6 and res["variant"] > 0.8
    print(f"variant check (synthetic, 1-6% positives): concept balanced accuracy "
          f"reference {res['reference']:.3f}, variant {res['variant']:.3f} -> {'PASS' if good else 'FAIL'}")
    return good


if __name__ == "__main__":
    main()
