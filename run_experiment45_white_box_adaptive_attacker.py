"""Experiment 45 (README Follow-up 56; manuscript Discussion item 4 /
Section 10.1, "P0-2": no fully adaptive white-box attacker, previously
unaddressed). Every perturbation tested elsewhere in this project draws
its ring-ZZ phase angles `phis` RANDOMLY (task-invariant, but not
optimized against the defender). This experiment asks the question the
manuscript's own limitation list left open: could an attacker who knows
the defender's trusted before-state (and hence `r_DC_before`, exactly
what a real white-box attacker with read access to the trusted
calibration could compute) choose `phis` ADVERSARIALLY, at the SAME
overall perturbation magnitude as the random baseline, to evade
detection far more effectively than a random direction does?

Objective (what the attacker minimizes): the TRUE post-perturbation
geometry drift ||r_DC_after - r_DC_before|| restricted to the offdiag
subspace, using the trusted J_before as the tangent-geometry reference
throughout the search (matching this project's own realistic-adaptive
convention, and avoiding the need to recompute the ~20-85s autograd
Jacobian per candidate -- each objective evaluation instead costs
<1ms: a forward statevector simulation is cached once per seed, and
only the cheap intervention+partial-trace step varies per candidate).
Constraint: candidate `phis` are rescaled to the SAME L2 norm as the
seed's own original random draw, so this is a fair "same magnitude,
adversarial direction vs. random direction" comparison, not a trivial
"do nothing" evasion.

Optimizer: derivative-free (the intervention functions are plain numpy,
not differentiable -- confirmed by inspection). 500-candidate random
search over directions on the 8-sphere, then local refinement via
scipy.optimize.minimize(method='Powell') from the best random start.

Downstream check: does the attacker's optimized direction also change
the OPERATIONAL outcome (whether j* still flips; the resulting static-
recovery number), not just the abstract geometry-drift objective.

Usage:
    python run_experiment45_white_box_adaptive_attacker.py [--smoke]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold
from manifold.theta_jacobian import di, tms, mni
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
LAYER = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
N_RANDOM_SEARCH = 500
EXPERIMENT_ID = "paper13_exp45_white_box_attacker_v1"


def _make_objective(flat_base, ring_pairs, jac_before, pool, r_dc_before_offdiag, target_norm, dim_a, dim_b,
                     active_mask=None):
    """active_mask: optional boolean array (len n_qubits); components where
    False are held at exactly 0 throughout the search (README Follow-up 56's
    constrained variant, excluding ring pairs entirely outside the audited
    subsystem -- otherwise the search trivially finds those, which evade
    detection for a reason unrelated to the diagnostic's own geometry)."""
    def objective(raw_phis):
        raw = raw_phis if active_mask is None else raw_phis * active_mask
        norm = np.linalg.norm(raw)
        if norm < 1e-8:
            candidate_phis = np.zeros_like(raw)
            free_idx = np.arange(len(raw)) if active_mask is None else np.flatnonzero(active_mask)
            candidate_phis[free_idx[0]] = target_norm
        else:
            candidate_phis = raw / norm * target_norm
        with torch.no_grad():
            flat2 = mni.apply_zz_null_unitary(flat_base.unsqueeze(0), N_QUBITS,
                                               list(candidate_phis), ring_pairs).reshape(-1)
            M = flat2.reshape(dim_a, dim_b)
            rho_candidate = (M @ M.conj().T).numpy()
        res = compute_R_manifold(rho_candidate, jac_before, pool)
        r_dc_candidate = res.r_DC_coef[pool.offdiag_idx]
        return float(np.linalg.norm(r_dc_candidate - r_dc_before_offdiag))
    return objective


def _static_recovery_for_phis(phis, flat_base, ring_pairs, jac_before, jac_after, pool,
                               j_before, dim_a, dim_b):
    with torch.no_grad():
        flat2 = mni.apply_zz_null_unitary(flat_base.unsqueeze(0), N_QUBITS, list(phis), ring_pairs).reshape(-1)
        M = flat2.reshape(dim_a, dim_b)
        rho_after_c = (M @ M.conj().T).numpy()
    baseline_after = compute_R_manifold(rho_after_c, jac_after, pool).gamma_D_C
    res_true_after = compute_R_manifold(rho_after_c, jac_after, pool)
    j_oracle = pool.offdiag_idx[int(np.argmax(res_true_after.exact_score[pool.offdiag_idx]))]
    j_flipped = j_oracle != j_before
    oracle_after = compute_R_manifold(rho_after_c, jac_after, pool, extra_measured_idx=[j_oracle]).gamma_D_C
    oracle_reduction = baseline_after - oracle_after
    static_after = compute_R_manifold(rho_after_c, jac_after, pool, extra_measured_idx=[j_before]).gamma_D_C
    static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")
    return static_recovery, j_flipped


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_random = 30 if smoke else N_RANDOM_SEARCH

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=False)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        ring_pairs = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]
        dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)

        with torch.no_grad():
            state = tms.simulate_final_state(x_ct[:1], theta0, N_QUBITS, LAYER, ring_pairs)
            flat_base = state.reshape(2 ** N_QUBITS)

        res_before = compute_R_manifold(rho_before, jac_before, pool)
        r_dc_before_offdiag = res_before.r_DC_coef[pool.offdiag_idx]
        exact_score_before_offdiag = res_before.exact_score[pool.offdiag_idx]
        j_before = pool.offdiag_idx[int(np.argmax(exact_score_before_offdiag))]

        rng_phi = np.random.RandomState(seed)
        original_phis = np.array(rng_phi.uniform(-1.5, 1.5, size=N_QUBITS))
        target_norm = float(np.linalg.norm(original_phis))

        objective = _make_objective(flat_base, ring_pairs, jac_before, pool,
                                     r_dc_before_offdiag, target_norm, dim_a, dim_b)

        # The true after-state's own Jacobian is not reused here: it would need
        # a fresh, costly autograd pass PER CANDIDATE state, since J(rho) is
        # itself state-dependent. Both selection and evaluation instead reuse
        # the trusted J_before as the tangent-geometry reference throughout,
        # consistent with every other "realistic" pipeline in this project
        # (Follow-up 44's own convention) -- this is why Ablation A's own
        # finding (geometry mismatch, not pilot noise, is the binding
        # constraint) is directly relevant background for interpreting these
        # numbers: any gap here already reflects that same known limitation,
        # not a new one specific to the attacker setting.
        baseline_drift = objective(original_phis)
        baseline_static, baseline_flip = _static_recovery_for_phis(
            original_phis, flat_base, ring_pairs, jac_before, jac_before, pool, j_before, dim_a, dim_b)

        rng_search = np.random.default_rng(derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", "search", "attacker"))
        best_val, best_raw = np.inf, None
        for _ in range(n_random):
            raw = rng_search.normal(size=N_QUBITS)
            val = objective(raw)
            if val < best_val:
                best_val, best_raw = val, raw

        opt = minimize(objective, best_raw, method="Powell", options={"maxiter": 300, "xtol": 1e-4})
        optimized_norm = np.linalg.norm(opt.x)
        optimized_phis = (opt.x / optimized_norm * target_norm) if optimized_norm > 1e-8 else original_phis
        optimized_drift = opt.fun
        optimized_static, optimized_flip = _static_recovery_for_phis(
            optimized_phis, flat_base, ring_pairs, jac_before, jac_before, pool, j_before, dim_a, dim_b)

        row = dict(seed=seed, target_norm=target_norm,
                   baseline_drift=baseline_drift, optimized_drift=optimized_drift,
                   baseline_static_recovery=baseline_static, optimized_static_recovery=optimized_static,
                   baseline_j_flipped=baseline_flip, optimized_j_flipped=optimized_flip)
        rows.append(row)
        print(f"seed={seed}: baseline_drift={baseline_drift:.4f} optimized_drift={optimized_drift:.4f} "
              f"(reduction={100*(1-optimized_drift/baseline_drift):.1f}%) "
              f"baseline_static={baseline_static*100:.1f}% optimized_static={optimized_static*100:.1f}% "
              f"baseline_flip={baseline_flip} optimized_flip={optimized_flip}", flush=True)
        _write_csv(RESULTS_DIR / "white_box_attacker.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    print("\n=== White-box attacker summary (mean across seeds) ===")
    bd = np.mean([r["baseline_drift"] for r in rows])
    od = np.mean([r["optimized_drift"] for r in rows])
    bs = np.mean([r["baseline_static_recovery"] for r in rows])
    os_ = np.mean([r["optimized_static_recovery"] for r in rows])
    n_still_flip = sum(1 for r in rows if r["optimized_j_flipped"])
    print(f"  drift: baseline={bd:.4f} -> optimized={od:.4f} ({100*(1-od/bd):.1f}% reduction)")
    print(f"  static recovery: baseline={bs*100:.1f}% -> optimized={os_*100:.1f}%")
    print(f"  j* still flips under optimized attack: {n_still_flip}/{len(rows)} seeds")


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
