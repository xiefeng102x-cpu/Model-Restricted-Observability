"""
candidate1_architecture_ladder_mnist.py
=================================================
Candidate 1 (redesigned after the r_hidden/r_dim triviality finding): tests
an escalating ladder of classical architectures against the QML circuit on
REAL MNIST data (n_qubits=5, layer=12, pair=t7_vs_t0, matching the project's
core experiment exactly), using the metric that actually matters --
Mahalanobis-distance collapse under a REAL, unrestricted weight-space
adaptive attack (Phase 1 clean-train + Phase 2 attack, same protocol as
topology_mechanism_sweep.py) -- NOT r_hidden/r_dim (shown to be near-
universal mathematical artifacts, not quantum-specific; see conversation).

Ladder (Level 0/1 reuse existing real numbers, not rerun here):
  Level 0: Unconstrained classical MLP        (existing: memory + prior runs)
  Level 1: Bloch-ball-projected classical MLP (existing: memory + prior runs)
  Level 2: OrthogonalMLP    -- NEW, from classical_architectures.py
  Level 3: RotationCascadeNet -- NEW, from classical_architectures.py
  Reference: QML circuit (n_qubits=5, ring, L=12) -- rerun here for a
             same-script, same-metric, directly comparable number.

All four (Level2, Level3, QML reference, [optionally re-verify Level0/1])
go through the IDENTICAL generic Phase1(clean CE)+Phase2(real Mahalanobis-
minimizing weight attack) harness, so the comparison is apples-to-apples --
same data, same protocol, same metric, only the feature-extractor differs.

Standalone rebuild; classical_architectures.py and topology_mechanism_
sweep.py are imported (read-only) for their validated circuit/architecture
code, not modified.

Self-tests: reuses classical_architectures.py's own self-tests (orthogonality,
RotationCascadeNet-vs-general-circuit equivalence) plus a Mahalanobis/
LedoitWolf sanity check.

Outputs (results/):
  - candidate1_ladder_phase1_history.csv
  - candidate1_ladder_phase2_history.csv
  - candidate1_ladder_summary.csv -- the key comparison table
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
tms = importlib.import_module("topology_mechanism_sweep")
ca = importlib.import_module("classical_architectures")

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/Mnist/detail_single_samle"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
LAYER = 12
PAIR = "t7_vs_t0"
PR = 0.1
N_QUBITS = 5
FEATURE_DIM = 32
RING_PAIRS = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]

PHASE1_EPOCHS = 200
PHASE1_LR = 0.003
PHASE1_BATCH = 64
PHASE2_STEPS = 200
PHASE2_LR = 0.05
PHASE2_EVAL_EVERY = 20


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_classes))

    def forward(self, z):
        return self.net(z.to(torch.float32))


def detection_auc(feats_clean, feats_poison, n=300, seed=0):
    rng = np.random.default_rng(seed)
    nc = min(len(feats_clean), n); np_ = min(len(feats_poison), n)
    ic = rng.choice(len(feats_clean), nc, replace=False)
    ip = rng.choice(len(feats_poison), np_, replace=False)
    fc, fp = feats_clean[ic], feats_poison[ip]
    nn_ = min(nc, np_)
    fc, fp = fc[:nn_], fp[:nn_]
    lw = LedoitWolf().fit(fc)
    diff = np.vstack([fc, fp]) - lw.location_
    scores = np.einsum("ni,ij,nj->n", diff, lw.get_precision(), diff)
    labels = np.array([0] * nn_ + [1] * nn_)
    return float(roc_auc_score(labels, scores))


def load_real_data(seed, layer, pair, pr):
    pr_s = f"{float(pr):.12g}"
    base = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt", map_location="cpu", weights_only=False)
    return meta["target_clean_data"].double(), meta["non_target_clean_data"].double(), meta["poisoned_non_target_data"].double()


# ========================================================================= generic Phase1/Phase2 harness

def get_zfeat_dim(feature_model):
    """Feature models return either XYZ (3n) or XZ (2n); the classifier reads
    only the LAST n_units columns in both cases -- which are always the Z block
    (see xyz_features_from_state / xz_features_from_real_state column ordering)."""
    return feature_model.n_qubits if hasattr(feature_model, "n_qubits") else feature_model.n_units


def phase1_pretrain_generic(feature_model, x_ct, x_cnt, seed, name):
    torch.manual_seed(seed)
    n_units = get_zfeat_dim(feature_model)
    head = ClassifierHead(n_units)
    opt = torch.optim.Adam(list(feature_model.parameters()) + list(head.parameters()), lr=PHASE1_LR)

    x_tr = torch.cat([x_ct, x_cnt], dim=0)
    y_tr = torch.cat([torch.ones(len(x_ct)), torch.zeros(len(x_cnt))]).long()
    rng = np.random.default_rng(seed)

    history = []
    feature_model.train(); head.train()
    for epoch in range(PHASE1_EPOCHS):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), PHASE1_BATCH):
            bi = idx[i:i + PHASE1_BATCH]
            xb, yb = x_tr[bi], y_tr[bi]
            feat = feature_model(xb)
            z = feat[:, -n_units:]
            loss = F.cross_entropy(head(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 50 == 0 or epoch == PHASE1_EPOCHS - 1:
            feature_model.eval(); head.eval()
            with torch.no_grad():
                z_ct = feature_model(x_ct)[:, -n_units:]
                z_cnt = feature_model(x_cnt)[:, -n_units:]
                cacc = 0.5 * ((head(z_ct).argmax(1) == 1).float().mean().item() +
                               (head(z_cnt).argmax(1) == 0).float().mean().item())
            history.append({"arch": name, "seed": seed, "epoch": epoch + 1, "ca": cacc})
            print(f"    [{name}/seed={seed}] phase1 epoch {epoch+1:4d}: CA={cacc:.4f}")
            feature_model.train(); head.train()

    feature_model.eval(); head.eval()
    return head, history


def phase2_attack_generic(feature_model, head, x_ct, x_p, x_cnt, seed, name):
    n_units = get_zfeat_dim(feature_model)
    opt = torch.optim.SGD(feature_model.parameters(), lr=PHASE2_LR, momentum=0.0)

    with torch.no_grad():
        feat_ref = feature_model(x_ct)
    lw_ref = LedoitWolf().fit(feat_ref.detach().numpy())
    mu_ref = torch.tensor(lw_ref.location_, dtype=torch.float32)
    prec_ref = torch.tensor(lw_ref.get_precision(), dtype=torch.float32)

    def mahal_of(feat):
        diff = (feat.to(torch.float32) - mu_ref)
        return (diff @ prec_ref * diff).sum(dim=1)

    def evaluate():
        with torch.no_grad():
            feat_p = feature_model(x_p); feat_ct = feature_model(x_ct); feat_cnt = feature_model(x_cnt)
            z_p, z_ct, z_cnt = feat_p[:, -n_units:], feat_ct[:, -n_units:], feat_cnt[:, -n_units:]
            asr = (head(z_p).argmax(1) == 1).float().mean().item()
            cacc = 0.5 * ((head(z_ct).argmax(1) == 1).float().mean().item() +
                          (head(z_cnt).argmax(1) == 0).float().mean().item())
            auc_full = detection_auc(feat_ct.numpy(), feat_p.numpy())
            mahal_p = mahal_of(feat_p).mean().item()
        return {"asr": asr, "ca": cacc, "auc_full": auc_full, "mahal": mahal_p}

    history = []
    det0 = evaluate(); det0.update({"arch": name, "seed": seed, "step": 0}); history.append(det0)
    print(f"    [{name}/seed={seed}] phase2 step    0: AUC={det0['auc_full']:.4f} ASR={det0['asr']:.4f} "
          f"CA={det0['ca']:.4f} Mahal={det0['mahal']:.3f}")

    for step in range(1, PHASE2_STEPS + 1):
        feat_p = feature_model(x_p)
        loss = mahal_of(feat_p).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % PHASE2_EVAL_EVERY == 0:
            det = evaluate(); det.update({"arch": name, "seed": seed, "step": step}); history.append(det)
            print(f"    [{name}/seed={seed}] phase2 step {step:4d}: AUC={det['auc_full']:.4f} ASR={det['asr']:.4f} "
                  f"CA={det['ca']:.4f} Mahal={det['mahal']:.3f}")

    return history


def run_one_arch(feature_model, x_ct, x_cnt, x_p, seed, name):
    t0 = time.time()
    head, hist1 = phase1_pretrain_generic(feature_model, x_ct, x_cnt, seed, name)
    print(f"  [{name}/seed={seed}] Phase 1 done ({time.time()-t0:.1f}s)")
    t0 = time.time()
    hist2 = phase2_attack_generic(feature_model, head, x_ct, x_p, x_cnt, seed, name)
    print(f"  [{name}/seed={seed}] Phase 2 done ({time.time()-t0:.1f}s)")

    mahal_before, mahal_after = hist2[0]["mahal"], hist2[-1]["mahal"]
    collapse_ratio = mahal_after / (mahal_before + 1e-12)
    summary = {"arch": name, "seed": seed,
               "mahal_before": mahal_before, "mahal_after": mahal_after,
               "collapse_ratio": collapse_ratio,
               "asr_after": hist2[-1]["asr"], "ca_after": hist2[-1]["ca"],
               "n_params": sum(p.numel() for p in feature_model.parameters())}
    print(f"  [{name}/seed={seed}] Mahal {mahal_before:.3f} -> {mahal_after:.3f} "
          f"(collapse_ratio={collapse_ratio:.4f}, lower=more collapsed=easier to evade)")
    return hist1, hist2, summary


def self_test():
    print("Running self-test (candidate1_architecture_ladder_mnist)...")
    ca.self_test()
    # quick LedoitWolf/Mahalanobis sanity
    rng = np.random.RandomState(0)
    feat_clean = rng.randn(50, 6)
    lw = LedoitWolf().fit(feat_clean)
    d_at_mean = np.einsum("i,ij,j->", np.zeros(6), lw.get_precision(), np.zeros(6))
    assert d_at_mean < 1e-8
    print(f"  [check] Mahalanobis distance at fitted mean = {d_at_mean:.2e} (expect ~0)")
    print("Self-test PASSED.\n")


def main(seed_filter=None):
    t_start = time.time()
    self_test()
    seeds_to_run = seed_filter if seed_filter is not None else SEEDS
    suffix = f"_seeds{'-'.join(str(s) for s in seeds_to_run)}" if seed_filter is not None else ""

    all_h1, all_h2, summaries = [], [], []

    for seed in seeds_to_run:
        x_ct, x_cnt, x_p = load_real_data(seed, LAYER, PAIR, PR)
        print(f"\n{'='*70}\nseed={seed}  (target_clean={x_ct.shape}, non_target_clean={x_cnt.shape}, poisoned={x_p.shape})\n{'='*70}")

        archs = {
            "QML_reference": None,  # special-cased below (reuses tms circuit directly)
            "OrthogonalMLP": ca.OrthogonalMLP(FEATURE_DIM, N_QUBITS, n_layers=LAYER // 4).double(),
            "RotationCascade": ca.RotationCascadeNet(N_QUBITS, LAYER, RING_PAIRS),
        }

        for name, model in archs.items():
            if name == "QML_reference":
                torch.manual_seed(seed)
                weights = (torch.randn(LAYER * N_QUBITS * 2, dtype=torch.float64) * 0.1).requires_grad_(True)
                class QMLWrap(nn.Module):
                    def __init__(self, w):
                        super().__init__()
                        self.weights = nn.Parameter(w)
                        self.n_qubits = N_QUBITS
                    def forward(self, x):
                        state = tms.simulate_final_state(x, self.weights, N_QUBITS, LAYER, RING_PAIRS)
                        xyz, _ = tms.xyz_features_from_state(state, N_QUBITS)
                        return xyz
                model = QMLWrap(weights)

            h1, h2, summ = run_one_arch(model, x_ct, x_cnt, x_p, seed, name)
            all_h1.extend(h1); all_h2.extend(h2); summaries.append(summ)

            pd.DataFrame(all_h1).to_csv(OUT_DIR / f"candidate1_ladder_phase1_history{suffix}.csv", index=False)
            pd.DataFrame(all_h2).to_csv(OUT_DIR / f"candidate1_ladder_phase2_history{suffix}.csv", index=False)
            pd.DataFrame(summaries).to_csv(OUT_DIR / f"candidate1_ladder_summary{suffix}.csv", index=False)

    print(f"\n\n=== CANDIDATE 1 LADDER SUMMARY (wall time {time.time()-t_start:.1f}s) ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    if "--self-test-only" in sys.argv:
        self_test()
    elif "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_args = [int(s) for s in sys.argv[idx + 1:] if s.lstrip("-").isdigit()]
        main(seed_filter=seed_args)
    else:
        main()
