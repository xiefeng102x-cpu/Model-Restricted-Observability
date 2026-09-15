"""Experiment 47 (README Follow-up 58; manuscript Discussion item 6,
"Simulation-centric validation": "the finite-shot noise model has no
gate error, decoherence, or calibration drift" -- P1(b)). Stress-tests
the end-to-end clean-vs-tampered detection score (Experiment 38, Follow-
up 49) against a REALISTIC hardware-noise-like corruption on top of the
existing finite-shot sampling noise, not yet tested anywhere in this
project.

Noise model: a global depolarizing channel applied to the reduced
audit-subsystem state before pilot measurement, rho_noisy = (1-p)*rho +
p*I/d, at p in {0, 0.01, 0.02, 0.05, 0.1} -- the standard first-order
approximation for accumulated gate error/decoherence on a small
subsystem, applied identically to BOTH the clean and tampered deployed
states (a real device's noise does not care whether the state was
tampered with). Reuses Experiment 37/38's own `_pilot_discrepancy` gate
statistic unchanged; only the state it is computed on is corrupted.

Scope: the primary ring-ZZ mechanism (full magnitude) and the Z-only
mechanism, 5 BloodMNIST seeds, N_PILOT=100K (Follow-up 44/48's headline
budget) -- a focused robustness check, not a full re-run of the entire
magnitude sweep under noise.

Usage:
    python run_experiment47_noise_robustness_depolarizing.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import ThetaState, rho_A_and_jacobian, di
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled, apply_single_z_null_unitary
from run_experiment37_gated_adaptive_audit import _r_dc_offdiag, _pilot_discrepancy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1]
N_PILOT = 100_000
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp47_noise_robustness_v1"


def _depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    d = rho.shape[0]
    return (1 - p) * rho + p * np.eye(d, dtype=complex) / d


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    noise_levels = [0.0, 0.05] if smoke else NOISE_LEVELS
    n_test_clean = 4 if smoke else N_TEST_CLEAN
    n_test_tampered = 4 if smoke else N_TEST_TAMPERED

    score_rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]

        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                         ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
        intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs: apply_zz_scaled(
            flat, N_QUBITS, phis, ring_pairs, 1.0)
        rho_after_zz, _ = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)

        z_thetas = list(np.random.RandomState(seed + 90001).uniform(-1.5, 1.5, size=N_QUBITS))
        z_fn = lambda flat, z_thetas=z_thetas: apply_single_z_null_unitary(flat, N_QUBITS, z_thetas)
        rho_after_z, _ = rho_A_and_jacobian(ts, "after", intervention_fn=z_fn)

        for p in noise_levels:
            rho_before_noisy = _depolarize(rho_before, p)
            r_dc_before_offdiag = _r_dc_offdiag(rho_before_noisy, jac_before, pool)

            for rep in range(n_test_clean):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"p{p}_clean_{rep}", "pilot")
                rng = np.random.default_rng(s)
                delta, _, _ = _pilot_discrepancy(rho_before_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, noise_p=p, condition="clean", label=0, delta_pilot=delta))

            for cond_name, rho_after in [("zz_full", rho_after_zz), ("z_only", rho_after_z)]:
                rho_after_noisy = _depolarize(rho_after, p)
                for rep in range(n_test_tampered):
                    s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"p{p}_{cond_name}_{rep}", "pilot")
                    rng = np.random.default_rng(s)
                    delta, _, _ = _pilot_discrepancy(rho_after_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                    score_rows.append(dict(seed=seed, noise_p=p, condition=cond_name, label=1, delta_pilot=delta))

            print(f"seed={seed} noise_p={p}: clean+zz_full+z_only draws done", flush=True)
            _write_csv(RESULTS_DIR / "noise_robustness_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows, noise_levels)


def _summarize(rows, noise_levels):
    print("\n=== Detection AUC vs. depolarizing noise level (pooled across seeds/conditions) ===")
    for p in noise_levels:
        sub = [r for r in rows if r["noise_p"] == p]
        labels = np.array([r["label"] for r in sub])
        scores = np.array([r["delta_pilot"] for r in sub])
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")
        print(f"  p={p:.2f}: pooled AUC={auc:.4f} (n_clean={int((labels==0).sum())}, n_tampered={int((labels==1).sum())})")

        for cond in ["zz_full", "z_only"]:
            clean_scores = np.array([r["delta_pilot"] for r in sub if r["condition"] == "clean"])
            cond_scores = np.array([r["delta_pilot"] for r in sub if r["condition"] == cond])
            y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(cond_scores))])
            s = np.concatenate([clean_scores, cond_scores])
            try:
                cond_auc = roc_auc_score(y, s)
            except ValueError:
                cond_auc = float("nan")
            print(f"      {cond}: AUC={cond_auc:.4f}")


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
