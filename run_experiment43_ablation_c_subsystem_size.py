"""Experiment 43 (README Follow-up 54; guide Section 15's "Ablation C --
subsystem size"). The guide's own framing: BloodMNIST's current audit
subsystem is n_a=4 (128 params vs. 255 ambient dim, R_manifold<1 -- the
whole point of this paper). The guide asks whether a LARGER reduced
subsystem (n_a=5, ambient dim 4^5-1=1023) makes the structural
restriction MORE pronounced (smaller R_manifold, i.e. more diagnostic
blindness the model's own constraints hide) at the cost of a much larger
measurement pool (1023 vs. 255 settings) -- a structural-benefit-vs-
measurement-burden tradeoff, not yet quantified anywhere in this project.

No retraining needed: `n_a` is a pure re-partitioning choice (which
qubits form the traced-out "A" subsystem) applied to the SAME already-
trained theta, exactly as manifold/theta_jacobian.py's own
`rho_A_and_jacobian` already supports via its `ts.n_a` field.

For each of the 5 BloodMNIST seeds: (1) recompute n_a=4 FRESH from theta
and cross-check it reproduces Experiment 25's persisted rho_A/jac
artifacts (a sanity check that the "fresh reconstruction" path used for
n_a=5 is trustworthy before trusting its own, never-before-computed
n_a=5 output); (2) compute the SAME quantities at n_a=5: R_manifold,
tangent_rank, static (clean-calibrated) recovery, and ONE representative
adaptive-recovery data point (100K-shot pilot, matching Follow-up 44's
headline budget) using the identical machinery throughout this project.

Usage:
    python run_experiment43_ablation_c_subsystem_size.py [--smoke]
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

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_QUBITS = 8
LAYER = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_A_VALUES = [4, 5]
N_PILOT = 100_000
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp43_ablation_c_v1"


def _static_and_adaptive_recovery(rho_before, jac_before, rho_after, jac_after, pool, n_repeats, seed_tag):
    baseline_after = compute_R_manifold(rho_after, jac_after, pool).gamma_D_C
    res_before = compute_R_manifold(rho_before, jac_before, pool)
    j_before = pool.offdiag_idx[int(np.argmax(res_before.exact_score[pool.offdiag_idx]))]

    res_true_after = compute_R_manifold(rho_after, jac_after, pool)
    j_oracle = pool.offdiag_idx[int(np.argmax(res_true_after.exact_score[pool.offdiag_idx]))]
    oracle_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_oracle]).gamma_D_C
    oracle_reduction = baseline_after - oracle_after

    static_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_before]).gamma_D_C
    static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

    adaptive_recoveries = []
    for rep in range(n_repeats):
        s = derive_seed(EXPERIMENT_ID, seed_tag, f"{rep}", "pilot")
        rng = np.random.default_rng(s)
        pilot = simulate_pilot(rho_after, pool.all_matrices, N_PILOT, rng)
        route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
        g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
        sel_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
        j_hat = pool.offdiag_idx[int(np.argmax(sel_res.exact_score[pool.offdiag_idx]))]
        eval_after = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_hat]).gamma_D_C
        adaptive_recoveries.append((baseline_after - eval_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan"))

    return dict(
        R_manifold_before=res_before.R_manifold, tangent_rank_before=res_before.tangent_rank,
        gamma_D_before=res_before.gamma_D, gamma_D_C_before=res_before.gamma_D_C,
        n_offdiag_candidates=len(pool.offdiag_idx),
        static_recovery=static_recovery, adaptive_recovery=float(np.mean(adaptive_recoveries)),
    )


def run(smoke: bool):
    seeds = SEEDS[:1] if smoke else SEEDS
    n_repeats = 3 if smoke else N_REPEATS

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=True)
        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        rng_phi = np.random.RandomState(seed)
        phis = list(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))

        for n_a in N_A_VALUES:
            ts = ThetaState(dataset=DATASET, seed=seed, theta=theta0, n_qubits=N_QUBITS, layer=LAYER,
                             ring_pairs=ring_pairs, n_a=n_a, x_ref=x_ct[:1], final_ca=float("nan"), phis=phis)
            rho_before, jac_before = rho_A_and_jacobian(ts, "before")
            rho_after, jac_after = rho_A_and_jacobian(ts, "after")

            if n_a == 4:
                # sanity check: fresh reconstruction must match Experiment 25's
                # persisted artifacts before the same code path is trusted for n_a=5
                err_rho = np.abs(rho_before - art_before["rho_A"]).max()
                err_jac = np.abs(jac_before - art_before["jac"]).max()
                # 1e-8 assumed bit-identical reproduction of another process's
                # autograd/SVD computation; confirmed elsewhere this session
                # (experiment26's own artifact cross-check) that separate
                # process launches of the same computation differ at the
                # ~1e-6 level from multi-threaded BLAS reduction-order
                # non-determinism, not a real bug. 1e-4 comfortably covers
                # that noise floor while still catching a genuinely wrong
                # or stale cached artifact.
                assert err_rho < 1e-4 and err_jac < 1e-4, (
                    f"seed={seed}: fresh n_a=4 reconstruction mismatches persisted artifact "
                    f"(err_rho={err_rho:.2e}, err_jac={err_jac:.2e})")

            pool = build_pool(n_a)
            rec = _static_and_adaptive_recovery(rho_before, jac_before, rho_after, jac_after, pool,
                                                 n_repeats, f"{DATASET}_seed{seed}_na{n_a}")
            rec.update(seed=seed, n_a=n_a)
            rows.append(rec)
            print(f"seed={seed} n_a={n_a}: R_manifold={rec['R_manifold_before']:.4f} "
                  f"tangent_rank={rec['tangent_rank_before']} "
                  f"n_candidates={rec['n_offdiag_candidates']} "
                  f"static={rec['static_recovery']*100:5.1f}% "
                  f"adaptive={rec['adaptive_recovery']*100:5.1f}%", flush=True)
            _write_csv(RESULTS_DIR / "ablation_c_subsystem_size.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    print("\n=== Ablation C summary: mean across seeds, n_a=4 vs n_a=5 ===")
    for n_a in N_A_VALUES:
        sub = [r for r in rows if r["n_a"] == n_a]
        if not sub:
            continue
        print(f"  n_a={n_a} (candidates={sub[0]['n_offdiag_candidates']}): "
              f"R_manifold={np.mean([r['R_manifold_before'] for r in sub]):.4f} "
              f"static={np.mean([r['static_recovery'] for r in sub])*100:5.1f}% "
              f"adaptive={np.mean([r['adaptive_recovery'] for r in sub])*100:5.1f}%")


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
