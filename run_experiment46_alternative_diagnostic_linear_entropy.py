"""Experiment 46 (README Follow-up 57; manuscript Discussion item 7,
"Diagnostic scope": the diagnostic studied throughout is von Neumann
entropy, "whether the same mechanism holds for other spectral ...
diagnostics is untested" -- P1(a)). Cross-checks the core mechanism
(blind directions exist; static calibration fails after task-invariant
tampering; adaptive pilot recalibration helps) using LINEAR entropy
D(rho)=1-Tr[rho^2] instead of von Neumann entropy D(rho)=-Tr[rho log
rho].

Cheap by construction, not by shortcut: for any TRACELESS perturbation
H (the only kind this project's Pauli-coefficient machinery ever
projects against, since pool.all_matrices excludes the identity
operator), d/dt D_linear(rho+tH)|_0 = -2 Tr[rho H] exactly -- so the
HS-gradient is G_linear = -2*rho, with no eigendecomposition, no
eig_floor clipping, and (unlike von Neumann entropy's -(log rho + I))
no identity-offset term to drop. This is algebraically exact, not an
approximation, and plugs directly into the SAME `g_coef_override`
mechanism `compute_R_manifold` already exposes -- no new machinery,
same pool, same jac, same pilot simulator, same selector's `rho_hat`
output (only the FINAL step converting rho_hat to a diagnostic gradient
changes, from -hermitian_log(rho_hat) to -2*rho_hat).

Reproduces Follow-up 42-44's exact structure at the primary ZZ
mechanism, 5 BloodMNIST seeds, 100K-shot pilot (Follow-up 44's headline
budget): R_manifold, static recovery, and adaptive recovery, all
recomputed under the swapped diagnostic.

Usage:
    python run_experiment46_alternative_diagnostic_linear_entropy.py [--smoke]
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
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_PILOT = 100_000
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp46_linear_entropy_v1"


def _g_linear_coef(rho: np.ndarray, all_matrices: np.ndarray) -> np.ndarray:
    return pauli_coefficients(-2.0 * rho, all_matrices)


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        g_before_lin = _g_linear_coef(rho_before, pool.all_matrices)
        res_before = compute_R_manifold(rho_before, jac_before, pool, g_coef_override=g_before_lin)
        j_before = pool.offdiag_idx[int(np.argmax(res_before.exact_score[pool.offdiag_idx]))]

        g_after_lin = _g_linear_coef(rho_after, pool.all_matrices)
        baseline_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_lin).gamma_D_C
        res_true_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_lin)
        j_oracle = pool.offdiag_idx[int(np.argmax(res_true_after.exact_score[pool.offdiag_idx]))]
        j_flipped = j_oracle != j_before
        oracle_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_lin,
                                           extra_measured_idx=[j_oracle]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after

        static_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_lin,
                                           extra_measured_idx=[j_before]).gamma_D_C
        static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        adaptive_recoveries = []
        for rep in range(n_repeats):
            s = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{rep}", "pilot")
            rng = np.random.default_rng(s)
            pilot = simulate_pilot(rho_after, pool.all_matrices, N_PILOT, rng)
            route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
            g_hat_lin = _g_linear_coef(route_a_out.rho_hat, pool.all_matrices)
            sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_lin)
            j_hat = pool.offdiag_idx[int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))]
            eval_after = compute_R_manifold(rho_after, jac_after, pool, g_coef_override=g_after_lin,
                                             extra_measured_idx=[j_hat]).gamma_D_C
            adaptive_recoveries.append((baseline_after - eval_after) / oracle_reduction
                                        if oracle_reduction > 1e-9 else float("nan"))

        row = dict(seed=seed, R_manifold_before=res_before.R_manifold,
                   j_flipped=j_flipped, static_recovery=static_recovery,
                   adaptive_recovery=float(np.mean(adaptive_recoveries)))
        rows.append(row)
        print(f"seed={seed}: R_manifold={row['R_manifold_before']:.4f} j_flipped={j_flipped} "
              f"static={static_recovery*100:5.1f}% adaptive={row['adaptive_recovery']*100:5.1f}%", flush=True)
        _write_csv(RESULTS_DIR / "linear_entropy_diagnostic_check.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    n_flip = sum(1 for r in rows if r["j_flipped"])
    print(f"\n=== Linear-entropy diagnostic summary (mean across {len(rows)} seeds) ===")
    print(f"  R_manifold={np.mean([r['R_manifold_before'] for r in rows]):.4f}")
    print(f"  j* flipped (before -> after tampering): {n_flip}/{len(rows)}")
    print(f"  static recovery={np.mean([r['static_recovery'] for r in rows])*100:.1f}%")
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
