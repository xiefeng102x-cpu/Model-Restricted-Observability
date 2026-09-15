"""Experiment 58 (README Follow-up 84). Experiment 57 tested amplitude-
damping robustness only on the full-magnitude ring-ZZ and Z-only
mechanisms -- both already near-perfect detection (AUC>=0.977) even
with ZERO added noise, so modest damping had essentially no competing
signal to degrade (exactly the same gap Experiment 47 had for
depolarizing noise, which Experiment 48 then closed). This experiment
asks the sharper question directly for the directional (non-isotropic)
channel: does detection degrade further, under added amplitude damping,
for the WEAK-magnitude conditions already near or at chance even
WITHOUT any added noise (zz_scale=0.05: AUC=0.51; zz_scale=0.15:
AUC=0.82; zz_scale=0.3: AUC=0.98)?

Same amplitude-damping channel and gate-discrepancy statistic as
Experiment 57, crossed with the same 3 weak/near-threshold ring-ZZ
magnitudes Experiment 48 used for depolarizing x 5 damping levels x 5
seeds -- a full magnitude x noise 2D sweep at the boundary this
project's own magnitude threshold identifies as where detection is
least reliable to begin with, now under a channel that can actually
rotate the detector's directional signal instead of only shrinking it.

Usage:
    python run_experiment58_near_threshold_amplitude_damping.py [--smoke]
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
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled
from run_experiment37_gated_adaptive_audit import _r_dc_offdiag, _pilot_discrepancy
from run_experiment57_noise_robustness_amplitude_damping import _amplitude_damping_kraus, _apply_channel

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DAMPING_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1]
ZZ_SCALES = [0.05, 0.15, 0.3]
N_PILOT = 100_000
N_TEST_CLEAN = 15
N_TEST_TAMPERED = 15
EXPERIMENT_ID = "paper13_exp58_near_threshold_amplitude_damping_v1"


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    damping_levels = [0.0, 0.05] if smoke else DAMPING_LEVELS
    zz_scales = [0.05, 0.3] if smoke else ZZ_SCALES
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

        rho_after_by_scale = {}
        for scale in zz_scales:
            intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs, scale=scale: apply_zz_scaled(
                flat, N_QUBITS, phis, ring_pairs, scale)
            rho_after_by_scale[scale], _ = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)

        for g in damping_levels:
            kraus = kraus_cache[g]
            rho_before_noisy = _apply_channel(rho_before, kraus)
            r_dc_before_offdiag = _r_dc_offdiag(rho_before_noisy, jac_before, pool)

            for rep in range(n_test_clean):
                s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"g{g}_clean_{rep}", "pilot")
                rng = np.random.default_rng(s)
                delta, _, _ = _pilot_discrepancy(rho_before_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                score_rows.append(dict(seed=seed, damping_g=g, zz_scale=None, condition="clean", label=0, delta_pilot=delta))

            for scale in zz_scales:
                rho_after_noisy = _apply_channel(rho_after_by_scale[scale], kraus)
                for rep in range(n_test_tampered):
                    s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"g{g}_zz{scale}_{rep}", "pilot")
                    rng = np.random.default_rng(s)
                    delta, _, _ = _pilot_discrepancy(rho_after_noisy, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
                    score_rows.append(dict(seed=seed, damping_g=g, zz_scale=scale, condition=f"zz_scale_{scale}",
                                            label=1, delta_pilot=delta))

            print(f"seed={seed} damping_g={g}: clean + {len(zz_scales)} zz-scale conditions done", flush=True)
            _write_csv(RESULTS_DIR / "near_threshold_amplitude_damping_scores.csv", score_rows)

    print(f"\nDone collecting {len(score_rows)} scored trials.")
    _summarize(score_rows, damping_levels, zz_scales)


def _summarize(rows, damping_levels, zz_scales):
    print("\n=== Detection AUC vs. (magnitude, damping level), near-threshold conditions ===")
    header = "  scale\\g  " + "  ".join(f"{g:6.2f}" for g in damping_levels)
    print(header)
    for scale in zz_scales:
        line = f"  {scale:<8}"
        for g in damping_levels:
            sub = [r for r in rows if r["damping_g"] == g and r["condition"] in ("clean", f"zz_scale_{scale}")]
            labels = np.array([r["label"] for r in sub])
            scores = np.array([r["delta_pilot"] for r in sub])
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = float("nan")
            line += f"  {auc:6.4f}"
        print(line)


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
