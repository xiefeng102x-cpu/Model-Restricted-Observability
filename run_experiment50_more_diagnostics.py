"""Experiment 50 (README Follow-up 62). Follow-up 57 (Experiment 46)
tested one alternative to von Neumann entropy (linear entropy) and left
Discussion item 7 partially open: "only one alternative diagnostic has
been tested." This experiment adds two more, chosen to be maximally
DIFFERENT from each other and from the two already tested, not just
minor variations:

  1. Renyi-2 entropy D(rho)=-log(Tr[rho^2]) -- a THIRD spectral
     diagnostic, distinct from von Neumann's logarithmic sensitivity
     and linear entropy's plain polynomial: d/dt D(rho+tH)|_0 =
     -2Tr[rho H]/Tr[rho^2] exactly (chain rule through the outer -log),
     so the HS-gradient is G=-2*rho/Tr[rho^2] -- still no
     eigendecomposition needed, but genuinely different curvature from
     linear entropy (a rational, not polynomial, function of rho).
  2. A NON-SPECTRAL diagnostic D(rho)=Tr[W rho] for a FIXED (not
     state-dependent) Hermitian operator W, standing in for "some other
     Hermitian property the auditor cares about" the way this project's
     von Neumann entropy already stands in for "entanglement
     structure." This directly closes the "non-spectral diagnostics"
     half of Discussion item 7's own phrasing, untouched by Follow-up
     57's spectral-only test. Its gradient is exactly W itself --
     CONSTANT, not state-dependent at all -- which makes "adaptive
     pilot recalibration" a degenerate, uninformative test here (the
     true gradient is always exactly known, nothing to estimate), so
     this diagnostic is tested for R_manifold and static recovery only,
     isolating the GEOMETRY-rotation mechanism specifically, decoupled
     from any gradient-estimation question.

Usage:
    python run_experiment50_more_diagnostics.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46]
N_PILOT = 100_000
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp50_more_diagnostics_v1"


def _g_renyi2_coef(rho: np.ndarray, all_matrices: np.ndarray) -> np.ndarray:
    tr_rho2 = float(np.trace(rho @ rho).real)
    return pauli_coefficients(-2.0 * rho / tr_rho2, all_matrices)


def _fixed_W(pool):
    """A single, fixed (not state-dependent) Hermitian observable, drawn
    once with a dedicated seed and reused identically across all 5
    seeds' states -- 'some other Hermitian property' distinct from any
    function of the specific state being audited."""
    rng = np.random.default_rng(derive_seed(EXPERIMENT_ID, "fixed_W", "draw", "diagnostic"))
    coef = rng.normal(size=len(pool.all_tuples))
    coef = coef / np.linalg.norm(coef)
    return coef  # already the "g" for D(rho)=Tr[W rho]: dD/drho = W itself, constant


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_repeats = 3 if smoke else N_REPEATS
    g_W_coef = _fixed_W(pool)

    renyi_rows, linear_diag_rows = [], []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        # --- Renyi-2 entropy: full static+adaptive pipeline, matching Experiment 46 ---
        g_before_r2 = _g_renyi2_coef(rho_before, pool.all_matrices)
        res_before_r2 = compute_R_manifold(rho_before, jac_before, pool, g_coef_override=g_before_r2)
        j_before_r2 = pool.offdiag_idx[int(np.argmax(res_before_r2.exact_score[pool.offdiag_idx]))]

        g_after_r2 = _g_renyi2_coef(rho_after, pool.all_matrices)
        baseline_after_r2 = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_r2).gamma_D_C
        res_true_after_r2 = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_r2)
        j_oracle_r2 = pool.offdiag_idx[int(np.argmax(res_true_after_r2.exact_score[pool.offdiag_idx]))]
        flip_r2 = j_oracle_r2 != j_before_r2
        oracle_after_r2 = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_r2,
                                              extra_measured_idx=[j_oracle_r2]).gamma_D_C
        oracle_reduction_r2 = baseline_after_r2 - oracle_after_r2
        static_after_r2 = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_r2,
                                              extra_measured_idx=[j_before_r2]).gamma_D_C
        static_recovery_r2 = ((baseline_after_r2 - static_after_r2) / oracle_reduction_r2
                               if oracle_reduction_r2 > 1e-9 else float("nan"))

        adaptive_recoveries_r2 = []
        for rep in range(n_repeats):
            s = derive_seed(EXPERIMENT_ID, f"renyi2_{DATASET}_seed{seed}", f"{rep}", "pilot")
            rng = np.random.default_rng(s)
            pilot = simulate_pilot(rho_after, pool.all_matrices, N_PILOT, rng)
            route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
            g_hat_r2 = _g_renyi2_coef(route_a_out.rho_hat, pool.all_matrices)
            sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_r2)
            j_hat = pool.offdiag_idx[int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))]
            eval_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_r2,
                                             extra_measured_idx=[j_hat]).gamma_D_C
            adaptive_recoveries_r2.append((baseline_after_r2 - eval_after) / oracle_reduction_r2
                                           if oracle_reduction_r2 > 1e-9 else float("nan"))

        renyi_rows.append(dict(seed=seed, R_manifold_before=res_before_r2.R_manifold,
                                j_flipped=flip_r2, static_recovery=static_recovery_r2,
                                adaptive_recovery=float(np.mean(adaptive_recoveries_r2))))
        print(f"[Renyi-2] seed={seed}: R_manifold={res_before_r2.R_manifold:.4f} flip={flip_r2} "
              f"static={static_recovery_r2*100:5.1f}% adaptive={np.mean(adaptive_recoveries_r2)*100:5.1f}%",
              flush=True)

        # --- fixed-linear non-spectral diagnostic: R_manifold + static only ---
        res_before_w = compute_R_manifold(rho_before, jac_before, pool, g_coef_override=g_W_coef)
        j_before_w = pool.offdiag_idx[int(np.argmax(res_before_w.exact_score[pool.offdiag_idx]))]

        baseline_after_w = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_W_coef).gamma_D_C
        res_true_after_w = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_W_coef)
        j_oracle_w = pool.offdiag_idx[int(np.argmax(res_true_after_w.exact_score[pool.offdiag_idx]))]
        flip_w = j_oracle_w != j_before_w
        oracle_after_w = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_W_coef,
                                             extra_measured_idx=[j_oracle_w]).gamma_D_C
        oracle_reduction_w = baseline_after_w - oracle_after_w
        static_after_w = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_W_coef,
                                             extra_measured_idx=[j_before_w]).gamma_D_C
        static_recovery_w = ((baseline_after_w - static_after_w) / oracle_reduction_w
                              if oracle_reduction_w > 1e-9 else float("nan"))

        linear_diag_rows.append(dict(seed=seed, R_manifold_before=res_before_w.R_manifold,
                                      j_flipped=flip_w, static_recovery=static_recovery_w))
        print(f"[Fixed-W ] seed={seed}: R_manifold={res_before_w.R_manifold:.4f} flip={flip_w} "
              f"static={static_recovery_w*100:5.1f}% (geometry-only mechanism, no adaptive test)", flush=True)

        _write_csv(RESULTS_DIR / "renyi2_diagnostic_check.csv", renyi_rows)
        _write_csv(RESULTS_DIR / "fixed_linear_diagnostic_check.csv", linear_diag_rows)

    print(f"\nDone. {len(renyi_rows)} Renyi-2 rows, {len(linear_diag_rows)} fixed-W rows written.")
    _summarize("Renyi-2", renyi_rows, has_adaptive=True)
    _summarize("Fixed-W (non-spectral)", linear_diag_rows, has_adaptive=False)


def _summarize(name, rows, has_adaptive):
    n_flip = sum(1 for r in rows if r["j_flipped"])
    print(f"\n=== {name} summary (mean across {len(rows)} seeds) ===")
    print(f"  R_manifold={np.mean([r['R_manifold_before'] for r in rows]):.4f}")
    print(f"  j* flipped: {n_flip}/{len(rows)}")
    print(f"  static recovery={np.mean([r['static_recovery'] for r in rows])*100:.1f}%")
    if has_adaptive:
        print(f"  adaptive recovery={np.mean([r['adaptive_recovery'] for r in rows])*100:.1f}%")


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
