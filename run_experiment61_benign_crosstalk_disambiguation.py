"""Experiment 61 (README Follow-up 89; Phase-2 peer-review roadmap item
4, responding to the Perspective/NISQ-hardware reviewer's W1 -- the
single most serious finding of that review): the paper's ring-ZZ
task-invariant construction is mathematically the same object as
static, always-on ZZ crosstalk between fixed-frequency transmons. The
reviewer's own order-of-magnitude estimate: residual static
$\zeta/2\pi\sim20$--$500$ kHz, circuit duration $t\sim2$--$4\mu$s, gives
benign accumulated phase per coupled pair $\phi=\zeta t\sim0.25$--$6$
rad -- a range that OVERLAPS the paper's own "full magnitude" attack
condition ($\mathrm{scale}=1.0$ draws $\mathrm{phis}\sim
\mathrm{Uniform}(-1.5,1.5)$ rad). If true, the end-to-end detector
(Experiment 38) may not be distinguishing "tampered" from "clean" at
all, but merely "large deviation from trusted reference" from "small
deviation" -- and realistic benign hardware drift could already BE a
large deviation.

Design: directly implements the reviewer's own proposed minimal test
(their comment D1). Draws "benign" per-pair phase perturbations from
Uniform(-phi_max, phi_max) at a sweep of phi_max levels spanning the
reviewer's estimated realistic range (0.05 to 6.0 rad), applies them via
the IDENTICAL ring-ZZ construction (apply_zz_scaled's underlying
mechanism) used for the paper's own attack, and scores them with the
IDENTICAL pilot-discrepancy statistic and clean-only-calibrated
threshold (95th percentile, Experiment 37's convention) used throughout
this paper. Reports the false-fire rate: what fraction of purely benign,
non-adversarial phase draws at each realistic magnitude would trigger
the detector calibrated only on clean data.

Usage:
    python run_experiment61_benign_crosstalk_disambiguation.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import di, tms
from manifold.restricted_observability import compute_R_manifold
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled
from run_experiment37_gated_adaptive_audit import _r_dc_offdiag, _pilot_discrepancy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46]
N_PILOT = 100_000
N_CALIB = 30          # clean-only, sets tau_gate (matches Experiment 37's convention)
N_BENIGN_REPEATS = 15  # per (seed, phi_max) benign draws
GATE_QUANTILE = 0.95

# Reviewer's own order-of-magnitude estimate: zeta/2pi ~ 20-500 kHz static
# ZZ, circuit duration t ~ 2-4 us, gives phi = zeta*t ~ 0.25-6 rad. Sweep
# spans below, within, and above that estimated realistic range, plus the
# paper's own "full magnitude" attack level (1.5 rad) for direct reference.
PHI_MAX_LEVELS = [0.05, 0.25, 0.5, 1.0, 1.5, 3.0, 6.0]
EXPERIMENT_ID = "paper13_exp61_benign_crosstalk_v1"


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    phi_levels = [0.5, 3.0] if smoke else PHI_MAX_LEVELS
    n_calib = 5 if smoke else N_CALIB
    n_benign = 4 if smoke else N_BENIGN_REPEATS

    calib_rows, rows = [], []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        r_dc_before_offdiag = _r_dc_offdiag(rho_before, jac_before, pool)

        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)
        with torch.no_grad():
            state_base = tms.simulate_final_state(x_ct[:1], theta0, N_QUBITS, 8, ring_pairs)
            flat_base = state_base.reshape(2 ** N_QUBITS)

        # --- Calibration: null distribution of the SAME gate signal on the
        #     CLEAN state only -- identical to Experiment 37's convention,
        #     never tuned on tampered OR benign-crosstalk data. ---
        null_vals = []
        for rep in range(n_calib):
            calib_seed = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"calib_{rep}", "pilot")
            rng = np.random.default_rng(calib_seed)
            delta, _, _ = _pilot_discrepancy(rho_before, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng)
            null_vals.append(delta)
        tau_gate = float(np.quantile(null_vals, GATE_QUANTILE))
        calib_rows.append(dict(seed=seed, n_calib=n_calib, tau_gate=tau_gate,
                                null_mean=float(np.mean(null_vals)), null_max=float(np.max(null_vals))))
        print(f"seed={seed}: calibrated tau_gate={tau_gate:.4f} "
              f"(null mean={np.mean(null_vals):.4f}, max={np.max(null_vals):.4f})", flush=True)
        _write_csv(RESULTS_DIR / "benign_crosstalk_calibration.csv", calib_rows)

        for phi_max in phi_levels:
            deltas, fired = [], []
            for rep in range(n_benign):
                # A fresh, independent "benign crosstalk" phase draw per repeat --
                # NOT the paper's own attack RNG stream, representing an
                # unrelated realistic device-noise realization, not an attacker.
                benign_seed = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"phimax{phi_max}_draw_{rep}", "benign")
                rng_benign = np.random.RandomState(benign_seed)
                benign_phis = list(rng_benign.uniform(-phi_max, phi_max, size=N_QUBITS))
                with torch.no_grad():
                    flat_benign = apply_zz_scaled(flat_base, N_QUBITS, benign_phis, ring_pairs, 1.0)
                    M = flat_benign.reshape(dim_a, dim_b)
                    rho_benign = (M @ M.conj().T).numpy()

                score_seed = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"phimax{phi_max}_score_{rep}", "pilot")
                rng_score = np.random.default_rng(score_seed)
                delta, _, _ = _pilot_discrepancy(rho_benign, jac_before, r_dc_before_offdiag, pool, N_PILOT, rng_score)
                deltas.append(delta)
                fired.append(delta > tau_gate)

            fire_rate = float(np.mean(fired))
            row = dict(seed=seed, phi_max_rad=phi_max, tau_gate=tau_gate,
                       mean_delta_benign=float(np.mean(deltas)), false_fire_rate=fire_rate)
            rows.append(row)
            print(f"  phi_max={phi_max:>5} rad: mean_delta={np.mean(deltas):.4f} "
                  f"(tau={tau_gate:.4f}) false_fire_rate={fire_rate:.2f}", flush=True)
            _write_csv(RESULTS_DIR / "benign_crosstalk_disambiguation.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows, phi_levels)


def _summarize(rows, phi_levels):
    print("\n=== Benign crosstalk false-fire rate (mean across seeds), by phi_max ===")
    for phi_max in phi_levels:
        sub = [r for r in rows if r["phi_max_rad"] == phi_max]
        if not sub:
            continue
        rate = np.mean([r["false_fire_rate"] for r in sub])
        print(f"  phi_max={phi_max:>5} rad: mean false_fire_rate={rate:.2f}  "
              f"(per-seed: {[round(r['false_fire_rate'],2) for r in sub]})")


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
