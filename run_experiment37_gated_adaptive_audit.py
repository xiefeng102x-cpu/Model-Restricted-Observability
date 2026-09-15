"""Experiment 37 (README Follow-up 48; manuscript Section 10.1's "next
design, not yet built"): implements and validates the Gated Adaptive
Audit the guide document's Section 6.4 / Algorithm 2 proposed --
Follow-up 46/47 showed static calibration is BETTER at weak perturbation
magnitudes (adaptive recalibration wastes pilot-noise budget for no
benefit) and WORSE at strong ones, so neither "always static" nor
"always adaptive" is the right fixed policy. This tests whether a cheap,
pilot-computable discrepancy signal can pick the right one per deployment.

Critical design constraint, worth stating explicitly (a real trap this
experiment specifically avoids): the gate signal CANNOT be based on the
already-measured (diagonal, task) Pauli values, because the entire
family of task-invariant perturbations studied throughout this project
is CONSTRUCTED to leave those values exactly unchanged (that is the
whole point of "task-invariant"). A gate built on diagonal discrepancy
would never fire for this exact threat model. Instead, the gate signal
reuses the SAME pilot data the adaptive step already collects (no extra
measurement cost): the OFF-diagonal Pauli coefficients the pilot
estimates, compared against what the trusted, offline-computed clean
("before") state predicts for those same coefficients.

Calibration discipline (guide's own explicit requirement, honored here):
tau_gate is calibrated ONLY on a clean (unperturbed) null distribution --
repeated pilot draws on the actual rho_before itself, with NO
perturbation applied -- never on the tampered/test magnitudes the gate
is then evaluated against.

Usage:
    python run_experiment37_gated_adaptive_audit.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.theta_jacobian import ThetaState, rho_A_and_jacobian, di
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed
from run_experiment35_multiple_perturbation_mechanisms import apply_zz_scaled

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SCALES = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_PILOT = 100_000
N_CALIBRATION_REPEATS = 30   # null-distribution draws, clean state only
N_TEST_REPEATS = 10          # per (seed, scale) evaluation draws
GATE_QUANTILE = 0.95         # tau_gate = 95th percentile of the clean null distribution
EXPERIMENT_ID = "paper13_exp37_gated_adaptive_v1"


def _r_dc_offdiag(rho, jac_before, pool, g_coef_override=None):
    """The manifold-restricted r_DC (NOT raw ambient g) -- the quantity
    that actually drives j*_C throughout this paper (Follow-up 43-47's
    established convention). Always evaluated through jac_before's
    tangent geometry, matching what the adaptive step itself does (the
    defender never has jac_after available)."""
    res = compute_R_manifold(rho, jac_before, pool, g_coef_override=g_coef_override)
    return res.r_DC_coef[pool.offdiag_idx], res.exact_score[pool.offdiag_idx]


def _pilot_discrepancy(rho_true, jac_before, r_dc_before_offdiag, pool, n_pilot, rng):
    """The gate signal: || pilot-estimated r_DC (via jac_before + pilot
    g_hat, the SAME construction the adaptive step itself uses) minus
    the trusted clean-state r_DC ||, using the SAME pilot draw the
    adaptive step already needs -- no extra measurement cost, and
    consistently in r_DC space (not raw ambient g space) throughout."""
    pilot = simulate_pilot(rho_true, pool.all_matrices, n_pilot, rng)
    route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
    g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
    r_dc_hat_offdiag, exact_hat_offdiag = _r_dc_offdiag(route_a_out.rho_hat, jac_before, pool, g_coef_override=g_hat_coef)
    return float(np.linalg.norm(r_dc_hat_offdiag - r_dc_before_offdiag)), r_dc_hat_offdiag, exact_hat_offdiag


def run(smoke: bool):
    pool = build_pool(N_A)
    out_calib = RESULTS_DIR / "gated_adaptive_calibration.csv"
    out_path = RESULTS_DIR / "gated_adaptive_audit.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    scales = [0.03, 1.0] if smoke else SCALES
    n_calib = 5 if smoke else N_CALIBRATION_REPEATS
    n_test = 3 if smoke else N_TEST_REPEATS

    calib_rows, rows = [], []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        r_dc_before_offdiag, exact_before_offdiag = _r_dc_offdiag(rho_before, jac_before, pool)

        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        # --- Calibration: null distribution of the gate signal on the
        #     CLEAN state only (no perturbation applied at all). ---
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
        _write_csv(out_calib, calib_rows)

        # --- Test: magnitude sweep, held out from calibration. j*_C
        #     (static pick) is r_DC's argmax, matching Follow-up 43-47's
        #     established convention throughout this paper -- NOT the
        #     raw ambient gradient's argmax (a different quantity, see
        #     Follow-up 35). ---
        j_before_local = int(np.argmax(exact_before_offdiag))
        j_before_global = pool.offdiag_idx[j_before_local]

        for scale in scales:
            ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=8,
                             ring_pairs=ring_pairs, n_a=N_A, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            intervention_fn = lambda flat, phis=phis, ring_pairs=ring_pairs, scale=scale: apply_zz_scaled(
                flat, N_QUBITS, phis, ring_pairs, scale)
            rho_after, jac_after = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)

            baseline_after = compute_R_manifold(rho_after, jac_after, pool).gamma_D_C
            static_after = compute_R_manifold(rho_after, jac_after, pool,
                                                extra_measured_idx=[j_before_global]).gamma_D_C
            # oracle: the TRUE after-state's own r_DC (full knowledge,
            # jac_after -- unavailable to any real defender, evaluation
            # only, never used for selection).
            exact_after_true = compute_R_manifold(rho_after, jac_after, pool).exact_score[pool.offdiag_idx]
            j_after_local = int(np.argmax(exact_after_true))
            j_after_global = pool.offdiag_idx[j_after_local]
            oracle_after = compute_R_manifold(rho_after, jac_after, pool,
                                               extra_measured_idx=[j_after_global]).gamma_D_C
            oracle_reduction = baseline_after - oracle_after
            static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

            deltas, adaptive_afters, gate_fired = [], [], []
            for rep in range(n_test):
                test_seed = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"scale{scale}_{rep}", "pilot")
                rng = np.random.default_rng(test_seed)
                delta, r_dc_hat_offdiag, exact_hat_offdiag = _pilot_discrepancy(rho_after, jac_before, r_dc_before_offdiag,
                                                               pool, N_PILOT, rng)
                deltas.append(delta)
                fired = delta > tau_gate
                gate_fired.append(fired)
                j_adapt_local = int(np.argmax(exact_hat_offdiag))
                j_adapt_global = pool.offdiag_idx[j_adapt_local]
                adaptive_after = compute_R_manifold(rho_after, jac_after, pool,
                                                     extra_measured_idx=[j_adapt_global]).gamma_D_C
                adaptive_afters.append(adaptive_after)

            adaptive_recovery = float(np.mean([(baseline_after - a) / oracle_reduction
                                                if oracle_reduction > 1e-9 else float("nan")
                                                for a in adaptive_afters]))
            gate_fire_rate = float(np.mean(gate_fired))
            gated_after = [adaptive_afters[i] if gate_fired[i] else static_after for i in range(n_test)]
            gated_recovery = float(np.mean([(baseline_after - a) / oracle_reduction
                                             if oracle_reduction > 1e-9 else float("nan")
                                             for a in gated_after]))

            row = dict(seed=seed, scale=scale, tau_gate=tau_gate,
                       mean_delta_pilot=float(np.mean(deltas)), gate_fire_rate=gate_fire_rate,
                       static_recovery=static_recovery, adaptive_recovery=adaptive_recovery,
                       gated_recovery=gated_recovery,
                       best_of_static_adaptive=max(static_recovery, adaptive_recovery))
            rows.append(row)
            print(f"  scale={scale:>5}: gate_fire_rate={gate_fire_rate:.2f} | "
                  f"static={static_recovery*100:5.1f}% adaptive={adaptive_recovery*100:5.1f}% "
                  f"gated={gated_recovery*100:5.1f}% (best of two={max(static_recovery,adaptive_recovery)*100:5.1f}%)",
                  flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, scales)


def _summarize(rows, scales):
    print("\n=== Gated adaptive audit summary (mean across seeds per scale) ===")
    for scale in scales:
        sub = [r for r in rows if r["scale"] == scale]
        if not sub:
            continue
        s = np.mean([r["static_recovery"] for r in sub])
        a = np.mean([r["adaptive_recovery"] for r in sub])
        g = np.mean([r["gated_recovery"] for r in sub])
        b = np.mean([r["best_of_static_adaptive"] for r in sub])
        fire = np.mean([r["gate_fire_rate"] for r in sub])
        print(f"  scale={scale:>5}: static={s*100:5.1f}% adaptive={a*100:5.1f}% "
              f"GATED={g*100:5.1f}% (best-of-two={b*100:5.1f}%) fire_rate={fire:.2f}")
    all_gated = np.mean([r["gated_recovery"] for r in rows])
    all_best = np.mean([r["best_of_static_adaptive"] for r in rows])
    all_static = np.mean([r["static_recovery"] for r in rows])
    all_adaptive = np.mean([r["adaptive_recovery"] for r in rows])
    print(f"\n  Overall mean: static={all_static*100:.1f}% adaptive={all_adaptive*100:.1f}% "
          f"gated={all_gated*100:.1f}% best-of-two-oracle-selection={all_best*100:.1f}%")


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
