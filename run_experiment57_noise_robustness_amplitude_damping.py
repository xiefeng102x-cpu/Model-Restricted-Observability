"""Experiment 57 (README Follow-up 83; Phase-2 peer-review roadmap,
responding to the Perspective/NISQ-hardware reviewer): Experiment 47's
only tested noise channel is global depolarizing, rho -> (1-p)*rho +
p*I/d -- a scalar multiple of the identity added to rho, hence exactly
isotropic and, structurally, incapable of ever rotating the detector's
own directional signal (the off-diagonal Pauli-coefficient discrepancy
r_DC lives entirely in the traceless part of rho; depolarizing shrinks
that traceless part uniformly toward zero but never rotates it). This
is a real structural blind spot in the paper's only noise robustness
check, independent of whether depolarizing is itself a realistic
channel: no result obtained under depolarizing alone can distinguish
"the detector is robust to noise" from "the detector was never tested
against a channel that could have broken it directionally."

Noise model: single-qubit amplitude damping, applied independently to
each of the n_a audit qubits via the standard 2-Kraus-operator channel
(K0=diag(1,sqrt(1-g)), K1=[[0,sqrt(g)],[0,0]]), tensored across all n_a
qubits (2^n_a joint Kraus operators). Unlike depolarizing, amplitude
damping is NOT a scalar multiple of identity: it biases the state
toward |0...0>, a fixed direction in the Pauli basis (every Z-type and
some off-diagonal coefficients shift asymmetrically, not just shrink
uniformly) -- physically, this is also the more realistic channel for
superconducting/many NISQ qubits, where energy relaxation toward the
ground state dominates over symmetric depolarization.

Same protocol as Experiment 47 otherwise (matched noise applied
identically to clean and tampered deployed states, same damping-
parameter grid values reused as an analogous scale, same primary
ring-ZZ and Z-only mechanisms, 5 BloodMNIST seeds, 100K-shot pilot) --
a focused robustness check under a structurally different channel, not
a full re-run of the magnitude sweep.

Usage:
    python run_experiment57_noise_robustness_amplitude_damping.py [--smoke]
"""
import argparse
import csv
import sys
from itertools import product
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
SEEDS = [42, 43, 44, 45, 46]
DAMPING_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1]
N_PILOT = 100_000
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp57_amplitude_damping_v1"


def _amplitude_damping_kraus(n_qubits: int, gamma: float):
    if gamma <= 0.0:
        return [np.eye(2 ** n_qubits, dtype=complex)]
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    single = [k0, k1]
    kraus_ops = []
    for combo in product(range(2), repeat=n_qubits):
        op = single[combo[0]]
        for idx in combo[1:]:
            op = np.kron(op, single[idx])
        kraus_ops.append(op)
    return kraus_ops


def _apply_channel(rho: np.ndarray, kraus_ops) -> np.ndarray:
    d = rho.shape[0]
    out = np.zeros((d, d), dtype=complex)
    for k in kraus_ops:
        out += k @ rho @ k.conj().T
    return out


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    damping_levels = [0.0, 0.05] if smoke else DAMPING_LEVELS
    n_test_clean = 4 if smoke else N_TEST_CLEAN
    n_test_tampered = 4 if smoke else N_TEST_TAMPERED

    kraus_cache = {g: _amplitude_damping_kraus(N_A, g) for g in damping_levels}

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

        for g in damping_levels:
            kraus = kraus_cache[g]
            rho_before_noisy = _apply_channel(rho_before, kraus)
            r_dc_before_offdiag = _r_dc_offdiag(rho_before_noisy, jac_before, pool)

            for rep in range(n_test_clean):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"g{g}_clean_{rep}", "pilot")
                rng = np.random.default_rng(s)
                delta, _, _ = _pilot_discrepancy(rho_before_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, damping_g=g, condition="clean", label=0, delta_pilot=delta))

            for cond_name, rho_after in [("zz_full", rho_after_zz), ("z_only", rho_after_z)]:
                rho_after_noisy = _apply_channel(rho_after, kraus)
                for rep in range(n_test_tampered):
                    s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"g{g}_{cond_name}_{rep}", "pilot")
                    rng = np.random.default_rng(s)
                    delta, _, _ = _pilot_discrepancy(rho_after_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                    score_rows.append(dict(seed=seed, damping_g=g, condition=cond_name, label=1, delta_pilot=delta))

            print(f"seed={seed} damping_g={g}: clean+zz_full+z_only draws done", flush=True)
            _write_csv(RESULTS_DIR / "noise_robustness_amplitude_damping_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows, damping_levels)


def _summarize(rows, damping_levels):
    print("\n=== Detection AUC vs. amplitude-damping level (pooled across seeds/conditions) ===")
    for g in damping_levels:
        sub = [r for r in rows if r["damping_g"] == g]
        labels = np.array([r["label"] for r in sub])
        scores = np.array([r["delta_pilot"] for r in sub])
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")
        print(f"  g={g:.2f}: pooled AUC={auc:.4f} (n_clean={int((labels==0).sum())}, n_tampered={int((labels==1).sum())})")

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
