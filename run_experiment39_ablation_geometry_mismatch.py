"""Experiment 39 (README Follow-up 50; manuscript Discussion item 8's
own flagged gap: "a plausible symptom... not yet directly confirmed by a
dedicated ablation"). Guide document Ablation A: decompose Follow-up
44's adaptive recovery (~20% mean, well short of the 100%-by-definition
oracle) into how much is lost to PILOT ESTIMATION NOISE alone versus how
much is lost to the JACOBIAN GEOMETRY MISMATCH alone (using the trusted
`jac_before` instead of the unavailable true `jac_after`) -- two
confounded error sources Follow-up 44 could not separate.

Four combinations of (gradient source, tangent-geometry source),
selection always followed by evaluation on the TRUE after-state
(rho_after, jac_after -- held out, exactly as throughout this project):

  (a) g_after (true) + J_after (true)   -- the oracle itself (should
      recover ~100% by construction; a self-consistency check, not a
      new number)
  (b) g_hat (pilot)  + J_after (true)   -- ONLY pilot noise; an
      unrealistic upper bound (defender never has J_after) isolating
      how good the pilot-based gradient estimate alone is
  (c) g_after (true) + J_before (trusted) -- ONLY geometry mismatch; an
      unrealistic upper bound (defender never has the true g_after)
      isolating how much the wrong tangent space alone costs
  (d) g_hat (pilot)  + J_before (trusted) -- the ACTUAL realistic
      adaptive method (Follow-up 44/47's own construction, reproduced
      here as a consistency check, not a new method)

Entirely reuses Experiment 25's persisted artifacts for BOTH before and
after states (rho_A AND jac, saved for each) -- no retraining, no fresh
Jacobian computation at all, only cheap pilot simulation +
compute_R_manifold calls.

Usage:
    python run_experiment39_ablation_geometry_mismatch.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from measurement_simulator.pauli_pilot import simulate_pilot
from selector.reduced_state_plugin import select as route_a_select
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
PILOT_BUDGETS = [10_000, 100_000, 1_000_000, 10_000_000]
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp39_ablation_geometry_v1"


def _j_star_and_recovery(rho_after, jac_after, pool, baseline_after, oracle_reduction,
                          jac_for_selection, g_coef_override=None):
    sel_res = compute_R_manifold(rho_after, jac_for_selection, pool, g_coef_override=g_coef_override)
    j_local = int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))
    j_global = pool.offdiag_idx[j_local]
    eval_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_global]).gamma_D_C
    recovery = (baseline_after - eval_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")
    return recovery


def run(smoke: bool):
    pool = build_pool(N_A)
    out_path = RESULTS_DIR / "ablation_geometry_mismatch.csv"
    seeds = SEEDS[:1] if smoke else SEEDS
    budgets = PILOT_BUDGETS[:2] if smoke else PILOT_BUDGETS
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        art_after = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_after.npz")
        jac_before = art_before["jac"]
        rho_after, jac_after = art_after["rho_A"], art_after["jac"]

        baseline_after = compute_R_manifold(rho_after, jac_after, pool).gamma_D_C
        # true oracle j* and its gamma_D_C, computed once and reused as the
        # normalization denominator for all four combinations below
        oracle_sel = compute_R_manifold(rho_after, jac_after, pool)
        j_oracle_local = int(np.argmax(oracle_sel.exact_score[pool.offdiag_idx]))
        j_oracle_global = pool.offdiag_idx[j_oracle_local]
        oracle_after = compute_R_manifold(rho_after, jac_after, pool,
                                           extra_measured_idx=[j_oracle_global]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after

        # (a) oracle: true g_after + true J_after (self-consistency check, should be ~100%)
        recovery_a = _j_star_and_recovery(rho_after, jac_after, pool, baseline_after, oracle_reduction,
                                            jac_for_selection=jac_after)
        # (c) true g_after + trusted J_before (geometry-mismatch-only)
        recovery_c = _j_star_and_recovery(rho_after, jac_after, pool, baseline_after, oracle_reduction,
                                            jac_for_selection=jac_before)

        for n_pilot in budgets:
            recoveries_b, recoveries_d = [], []
            for rep in range(n_repeats):
                pilot_seed = derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", f"{n_pilot}_{rep}", "pilot")
                rng = np.random.default_rng(pilot_seed)
                pilot = simulate_pilot(rho_after, pool.all_matrices, n_pilot, rng)
                route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
                g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)

                # (b) pilot g_hat + true J_after (pilot-noise-only)
                recoveries_b.append(_j_star_and_recovery(
                    rho_after, jac_after, pool, baseline_after, oracle_reduction,
                    jac_for_selection=jac_after, g_coef_override=g_hat_coef))
                # (d) pilot g_hat + trusted J_before (the actual realistic method)
                recoveries_d.append(_j_star_and_recovery(
                    rho_after, jac_after, pool, baseline_after, oracle_reduction,
                    jac_for_selection=jac_before, g_coef_override=g_hat_coef))

            row = dict(seed=seed, n_pilot=n_pilot,
                       recovery_a_oracle=recovery_a,
                       recovery_b_pilot_noise_only=float(np.mean(recoveries_b)),
                       recovery_c_geometry_mismatch_only=recovery_c,
                       recovery_d_realistic_adaptive=float(np.mean(recoveries_d)))
            rows.append(row)
            print(f"seed={seed} n_pilot={n_pilot:>9}: "
                  f"(a)oracle={recovery_a*100:5.1f}% "
                  f"(b)pilot-noise-only={row['recovery_b_pilot_noise_only']*100:5.1f}% "
                  f"(c)geometry-only={recovery_c*100:5.1f}% "
                  f"(d)realistic={row['recovery_d_realistic_adaptive']*100:5.1f}%", flush=True)
            _write_csv(out_path, rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}.")
    _summarize(rows, budgets)


def _summarize(rows, budgets):
    print("\n=== Ablation A summary (mean across seeds per budget) ===")
    for n_pilot in budgets:
        sub = [r for r in rows if r["n_pilot"] == n_pilot]
        if not sub:
            continue
        a = np.mean([r["recovery_a_oracle"] for r in sub])
        b = np.mean([r["recovery_b_pilot_noise_only"] for r in sub])
        c = np.mean([r["recovery_c_geometry_mismatch_only"] for r in sub])
        d = np.mean([r["recovery_d_realistic_adaptive"] for r in sub])
        print(f"  n_pilot={n_pilot:>9}: (a)oracle={a*100:5.1f}% (b)pilot-noise-only={b*100:5.1f}% "
              f"(c)geometry-only={c*100:5.1f}% (d)realistic={d*100:5.1f}%")


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
