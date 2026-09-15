"""
candidate3_architecture_ladder_bloodmnist.py
======================================================
"换数据集测一下" -- runs the full 2x2-factorial architecture ladder
(Bloch, OrthogonalMLP, RotationCascadeNet, PermutationCascadeNet, plus the
QML reference) on REAL BloodMNIST data (n_qubits=8, layer=8, pair=t6_vs_t0),
using the exact same Mahalanobis-collapse-under-real-attack metric as
candidate 1 (real MNIST) and candidate 2 (synthetic scaling). This is the
"different dataset" complement to candidate 1's real-MNIST ladder -- same
architectures, same generic Phase1/Phase2 harness, real data instead of
synthetic, second real dataset instead of relying solely on MNIST.

                norm-preserving   NOT norm-preserving
entangle:  no    OrthogonalMLP         Bloch
entangle: yes  RotationCascadeNet  PermutationCascadeNet
  (+ QML reference, which is norm-preserving AND entangling, i.e. genuinely
   quantum -- the fifth point completing the comparison)

Reuses: candidate1_architecture_ladder_mnist.py's generic Phase1/Phase2
harness (imported, not duplicated) and bloodmnist_confound_controls.py's
real-data loading convention (n_qubits=8, layer=8, pair=t6_vs_t0, pr=0.1,
data/MedMnist/single_detect_20260411/epsilon_0.8/...).

Self-tests: reuses classical_architectures.py's self-tests (orthogonality,
RotationCascadeNet/PermutationCascadeNet correctness) plus a Mahalanobis
sanity check.

Outputs (results/):
  - candidate3_bloodmnist_ladder_phase2_history.csv
  - candidate3_bloodmnist_ladder_summary.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
tms = importlib.import_module("topology_mechanism_sweep")
ca = importlib.import_module("classical_architectures")
c1 = importlib.import_module("candidate1_architecture_ladder_mnist")

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/MedMnist/single_detect_20260411/epsilon_0.8"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
LAYER = 8
PAIR = "t6_vs_t0"
PR = 0.1
N_QUBITS = 8
FEATURE_DIM = 256
RING_PAIRS = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]


def load_real_data(seed, layer, pair, pr):
    pr_s = f"{float(pr):.12g}"
    base = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta_path = base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt"
    if not meta_path.exists():
        # QST fresh-cohort seeds (47-51): this minimal code release does not
        # bundle a cohort_fresh/ data-generation module, so this fallback
        # will raise a clear ModuleNotFoundError for these seeds rather than
        # a confusing FileNotFoundError deep in torch.load -- see README for
        # which seeds this release supports end to end.
        cohort_fresh_dir = ROOT / "cohort_fresh"
        if str(cohort_fresh_dir) not in sys.path:
            sys.path.insert(0, str(cohort_fresh_dir))
        import build_fresh_cohort as _bfc
        x_ct, x_cnt, _, _ = _bfc.clean_pool_for_new_seed("bloodmnist", seed)
        x_p = x_cnt[:60].clone()  # unused placeholder -- never read downstream
        return x_ct, x_cnt, x_p
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    return meta["target_clean_data"].double(), meta["non_target_clean_data"].double(), meta["poisoned_non_target_data"].double()


def self_test():
    print("Running self-test (candidate3_architecture_ladder_bloodmnist)...")
    ca.self_test()
    rng = np.random.RandomState(0)
    feat_clean = rng.randn(50, 6)
    lw = LedoitWolf().fit(feat_clean)
    d_at_mean = np.einsum("i,ij,j->", np.zeros(6), lw.get_precision(), np.zeros(6))
    assert d_at_mean < 1e-8
    print(f"  [check] Mahalanobis distance at fitted mean = {d_at_mean:.2e} (expect ~0)")
    print("Self-test PASSED.\n")


def make_qml(seed):
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

    return QMLWrap(weights)


def make_bloch():
    class BlochWrap(nn.Module):
        def __init__(self, in_dim, nq):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 3 * nq))
            self.n_qubits = nq
        def forward(self, x):
            z = self.net(x.to(torch.float32))
            z3 = z.view(-1, self.n_qubits, 3)
            norms = z3.norm(dim=-1, keepdim=True).clamp(min=1.0)
            z3 = z3 / norms
            return torch.cat([z3[:, :, 0], z3[:, :, 1], z3[:, :, 2]], dim=1).double()
    return BlochWrap(FEATURE_DIM, N_QUBITS)


def main(seed_filter=None):
    t_start = time.time()
    self_test()
    seeds_to_run = seed_filter if seed_filter is not None else SEEDS
    suffix = f"_seeds{'-'.join(str(s) for s in seeds_to_run)}" if seed_filter is not None else ""

    all_h2, summaries = [], []

    for seed in seeds_to_run:
        x_ct, x_cnt, x_p = load_real_data(seed, LAYER, PAIR, PR)
        print(f"\n{'#'*70}\n# seed={seed}: real BloodMNIST, target_clean={x_ct.shape}, "
              f"non_target_clean={x_cnt.shape}, poisoned={x_p.shape}\n{'#'*70}")

        archs = {
            "QML": make_qml(seed),
            "Bloch": make_bloch(),
            "OrthogonalMLP": ca.OrthogonalMLP(FEATURE_DIM, N_QUBITS, n_layers=LAYER // 4).double(),
            "RotationCascade": ca.RotationCascadeNet(N_QUBITS, LAYER, RING_PAIRS),
            "PermutationCascade": ca.PermutationCascadeNet(N_QUBITS, LAYER, RING_PAIRS),
        }

        for name, model in archs.items():
            print(f"\n{'='*70}\narch={name}  seed={seed}\n{'='*70}")
            h1, h2, summ = c1.run_one_arch(model, x_ct, x_cnt, x_p, seed, name)
            all_h2.extend(h2); summaries.append(summ)
            pd.DataFrame(all_h2).to_csv(OUT_DIR / f"candidate3_bloodmnist_ladder_phase2_history{suffix}.csv", index=False)
            pd.DataFrame(summaries).to_csv(OUT_DIR / f"candidate3_bloodmnist_ladder_summary{suffix}.csv", index=False)

    df = pd.DataFrame(summaries)
    print(f"\n\n=== CANDIDATE 3 (BloodMNIST LADDER) SUMMARY (wall time {time.time()-t_start:.1f}s) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    if "--self-test-only" in sys.argv:
        self_test()
    elif "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_args = [int(s) for s in sys.argv[idx + 1:] if s.lstrip("-").isdigit()]
        main(seed_filter=seed_args)
    else:
        main()
