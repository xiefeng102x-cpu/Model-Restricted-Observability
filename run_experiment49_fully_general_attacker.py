"""Experiment 49 (README Follow-up 60). Experiment 45/45b's white-box
attacker was restricted to this project's own ring-topology ZZ family
(8 free parameters, nearest-neighbor pairs only) -- a real limitation:
the ring topology is a specific, low-body-order special case of the
much larger family of task-invariant diagonal generators. This
experiment gives the attacker a genuinely more general search space:
ALL-TO-ALL two-qubit ZZ generators (all C(8,2)=28 pairs, not just the 8
ring-adjacent ones) PLUS all 8 single-qubit Z generators, 36 total
parameters -- still built from the same kind of low-body-order diagonal
Pauli-string generators (matching how a real gate-level backdoor would
plausibly be constructed), just without the ring-topology restriction.

Two searches, exactly mirroring Experiment 45/45b's own structure:

  1. Unconstrained (36 parameters): expected, as a consistency check
     against Experiment 45's own finding, to rediscover the SAME trivial
     exploit -- concentrating the perturbation on generators entirely
     within the traced-out subsystem B (6 of the 28 ZZ pairs and 4 of
     the 8 Z terms are entirely within B for n_a=4), which provably
     cannot affect rho_A at all.
  2. Constrained to the 26 parameters that touch the audited subsystem
     A (excluding those 10 B-only generators): the genuinely harder
     question -- does a MUCH richer search space (26 parameters, versus
     Experiment 45b's 5) let an informed attacker achieve meaningfully
     BETTER evasion than the ring-topology-only constrained case found
     (30.3% mean drift reduction)?

Same objective (minimize true geometry drift at matched perturbation
magnitude), same derivative-free optimizer (random search + Powell
refinement) as Experiment 45/45b.

Usage:
    python run_experiment49_fully_general_attacker.py [--smoke]
"""
import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates.pauli_pool import build_pool
from manifold.restricted_observability import compute_R_manifold, hermitian_log, pauli_coefficients
from manifold.theta_jacobian import di, tms, mni
from seeds import derive_seed

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARTIFACTS_DIR = RESULTS_DIR / "manifold_artifacts"
DATASET = "bloodmnist"
N_A = 4
N_QUBITS = 8
LAYER = 8
SEEDS = [42, 43, 44, 45, 46]
N_RANDOM_SEARCH = 800
EXPERIMENT_ID = "paper13_exp49_fully_general_attacker_v1"

ALL_ZZ_PAIRS = list(combinations(range(N_QUBITS), 2))          # 28 pairs
ALL_Z_TERMS = list(range(N_QUBITS))                             # 8 terms
N_ZZ, N_Z = len(ALL_ZZ_PAIRS), len(ALL_Z_TERMS)
N_TOTAL = N_ZZ + N_Z                                             # 36
A_RELEVANT_MASK = np.array(
    [(p < N_A) or (q < N_A) for (p, q) in ALL_ZZ_PAIRS] + [(q < N_A) for q in ALL_Z_TERMS]
)


def _apply_general_diagonal(psi_flat, n_qubits, zz_phis, z_thetas):
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    total_phase = np.zeros(dim)
    for (i, j), phi in zip(ALL_ZZ_PAIRS, zz_phis):
        bit_i = (idx >> (n_qubits - 1 - i)) & 1
        bit_j = (idx >> (n_qubits - 1 - j)) & 1
        total_phase = total_phase + phi * (1 - 2 * bit_i) * (1 - 2 * bit_j)
    for q, theta_q in zip(ALL_Z_TERMS, z_thetas):
        bit_q = (idx >> (n_qubits - 1 - q)) & 1
        total_phase = total_phase + theta_q * (1 - 2 * bit_q)
    phase_factor = torch.tensor(np.exp(-1j * total_phase), dtype=torch.complex128)
    return psi_flat * phase_factor


def _precompute_tangent_basis(jac, pool, tangent_tol=1e-8, null_tol=1e-6):
    """Duplicates compute_R_manifold's E/N computation (same tolerances,
    same measured_idx=pool.diag_idx convention, no extra_measured_idx) --
    the part that depends ONLY on `jac`/`pool`, never on the candidate
    state. Precomputing this ONCE (rather than once per objective
    evaluation, as compute_R_manifold itself does internally) turns a
    ~80ms-per-call SVD into a <1ms matrix-vector product, essential for a
    derivative-free search needing thousands of evaluations. Verified
    against compute_R_manifold's own output before use (see
    _verify_fast_path)."""
    jac_moved = np.moveaxis(jac, -1, 0)
    jac_coef = pauli_coefficients(jac_moved, pool.all_matrices).T
    U, S, _ = np.linalg.svd(jac_coef, full_matrices=False)
    rank = int((S > tangent_tol * max(S.max(), 1e-300)).sum())
    E = U[:, :rank]
    A = E[pool.diag_idx, :]
    _, Sa, Vha = np.linalg.svd(A, full_matrices=True)
    rank_a = int((Sa > null_tol * max(Sa.max(), 1e-300)).sum())
    N = Vha.T[:, rank_a:]
    return E, N


def _fast_r_dc_offdiag(rho_A, E, N, pool):
    g_coef = pauli_coefficients(-hermitian_log(rho_A), pool.all_matrices)
    c_proj = E.T @ g_coef
    if N.shape[1] == 0:
        return np.zeros(len(pool.offdiag_idx))
    n_coeffs = N.T @ c_proj
    r_DC_in_tangent = N @ n_coeffs
    r_DC_coef = E @ r_DC_in_tangent
    return r_DC_coef[pool.offdiag_idx]


def _verify_fast_path(rho_A, jac_before, pool):
    E, N = _precompute_tangent_basis(jac_before, pool)
    fast = _fast_r_dc_offdiag(rho_A, E, N, pool)
    slow = compute_R_manifold(rho_A, jac_before, pool).r_DC_coef[pool.offdiag_idx]
    err = np.abs(fast - slow).max()
    assert err < 1e-10, f"fast tangent-basis path mismatches compute_R_manifold: max err={err:.2e}"


def _make_objective(flat_base, jac_before, pool, r_dc_before_offdiag, target_norm, dim_a, dim_b, active_mask=None):
    E, N = _precompute_tangent_basis(jac_before, pool)

    def objective(raw_params):
        raw = raw_params if active_mask is None else raw_params * active_mask
        norm = np.linalg.norm(raw)
        if norm < 1e-8:
            candidate = np.zeros_like(raw)
            free_idx = np.arange(len(raw)) if active_mask is None else np.flatnonzero(active_mask)
            candidate[free_idx[0]] = target_norm
        else:
            candidate = raw / norm * target_norm
        zz_phis, z_thetas = candidate[:N_ZZ], candidate[N_ZZ:]
        with torch.no_grad():
            flat2 = _apply_general_diagonal(flat_base, N_QUBITS, zz_phis, z_thetas)
            M = flat2.reshape(dim_a, dim_b)
            rho_candidate = (M @ M.conj().T).numpy()
        r_dc_candidate = _fast_r_dc_offdiag(rho_candidate, E, N, pool)
        return float(np.linalg.norm(r_dc_candidate - r_dc_before_offdiag))
    return objective


def _static_recovery(candidate, flat_base, jac_before, pool, j_before, dim_a, dim_b):
    zz_phis, z_thetas = candidate[:N_ZZ], candidate[N_ZZ:]
    with torch.no_grad():
        flat2 = _apply_general_diagonal(flat_base, N_QUBITS, zz_phis, z_thetas)
        M = flat2.reshape(dim_a, dim_b)
        rho_after_c = (M @ M.conj().T).numpy()
    baseline_after = compute_R_manifold(rho_after_c, jac_before, pool).gamma_D_C
    res_true_after = compute_R_manifold(rho_after_c, jac_before, pool)
    j_oracle = pool.offdiag_idx[int(np.argmax(res_true_after.exact_score[pool.offdiag_idx]))]
    j_flipped = j_oracle != j_before
    oracle_after = compute_R_manifold(rho_after_c, jac_before, pool, extra_measured_idx=[j_oracle]).gamma_D_C
    oracle_reduction = baseline_after - oracle_after
    static_after = compute_R_manifold(rho_after_c, jac_before, pool, extra_measured_idx=[j_before]).gamma_D_C
    static_recovery = (baseline_after - static_after) / oracle_reduction if oracle_reduction > 1e-9 else float("nan")
    return static_recovery, j_flipped


def _search(seed, flat_base, jac_before, pool, r_dc_before_offdiag, target_norm, dim_a, dim_b,
            active_mask, n_random, tag):
    objective = _make_objective(flat_base, jac_before, pool, r_dc_before_offdiag, target_norm,
                                 dim_a, dim_b, active_mask=active_mask)
    rng_search = np.random.default_rng(derive_seed(EXPERIMENT_ID, f"{DATASET}_seed{seed}", tag, "attacker"))
    best_val, best_raw = np.inf, None
    for _ in range(n_random):
        raw = rng_search.normal(size=N_TOTAL) * (1 if active_mask is None else active_mask)
        val = objective(raw)
        if val < best_val:
            best_val, best_raw = val, raw
    opt = minimize(objective, best_raw, method="Powell", options={"maxiter": 150, "xtol": 1e-4})
    raw_opt = opt.x if active_mask is None else opt.x * active_mask
    norm_opt = np.linalg.norm(raw_opt)
    candidate_opt = (raw_opt / norm_opt * target_norm) if norm_opt > 1e-8 else raw_opt
    drift_opt = objective(candidate_opt)
    return candidate_opt, drift_opt


def run(smoke: bool):
    pool = build_pool(N_A)
    seeds = SEEDS[:1] if smoke else SEEDS
    n_random = 40 if smoke else N_RANDOM_SEARCH
    print(f"N_TOTAL={N_TOTAL} (28 ZZ + 8 Z); A-relevant params={A_RELEVANT_MASK.sum()}\n")

    rows = []
    for seed in seeds:
        art_before = np.load(ARTIFACTS_DIR / f"{DATASET}_seed{seed}_before.npz")
        theta0 = torch.tensor(art_before["theta"], dtype=torch.float64, requires_grad=False)
        rho_before, jac_before = art_before["rho_A"], art_before["jac"]
        cfg = di.DATASETS[DATASET]
        x_ct, x_cnt, x_p = cfg["load"](seed, cfg["layer"], cfg["pair"], cfg["pr"])
        dim_a, dim_b = 2 ** N_A, 2 ** (N_QUBITS - N_A)

        with torch.no_grad():
            state = tms.simulate_final_state(x_ct[:1], theta0, N_QUBITS, LAYER,
                                              [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)])
            flat_base = state.reshape(2 ** N_QUBITS)

        res_before = compute_R_manifold(rho_before, jac_before, pool)
        r_dc_before_offdiag = res_before.r_DC_coef[pool.offdiag_idx]
        exact_score_before_offdiag = res_before.exact_score[pool.offdiag_idx]
        j_before = pool.offdiag_idx[int(np.argmax(exact_score_before_offdiag))]
        _verify_fast_path(rho_before, jac_before, pool)

        rng_baseline = np.random.RandomState(seed + 77001)
        baseline_full = rng_baseline.uniform(-0.3, 0.3, size=N_TOTAL)  # modest per-term scale given 36 terms sum
        target_norm_full = float(np.linalg.norm(baseline_full))
        target_norm_constrained = float(np.linalg.norm(baseline_full[A_RELEVANT_MASK]))

        objective_full = _make_objective(flat_base, jac_before, pool, r_dc_before_offdiag,
                                          target_norm_full, dim_a, dim_b)
        baseline_drift = objective_full(baseline_full)
        baseline_static, baseline_flip = _static_recovery(baseline_full, flat_base, jac_before, pool,
                                                            j_before, dim_a, dim_b)

        cand_unconstrained, drift_unconstrained = _search(
            seed, flat_base, jac_before, pool, r_dc_before_offdiag, target_norm_full, dim_a, dim_b,
            active_mask=None, n_random=n_random, tag="unconstrained")
        static_unconstrained, flip_unconstrained = _static_recovery(
            cand_unconstrained, flat_base, jac_before, pool, j_before, dim_a, dim_b)
        b_only_energy = float(np.linalg.norm(cand_unconstrained[~A_RELEVANT_MASK]) / (np.linalg.norm(cand_unconstrained) + 1e-300))

        cand_constrained, drift_constrained = _search(
            seed, flat_base, jac_before, pool, r_dc_before_offdiag, target_norm_constrained, dim_a, dim_b,
            active_mask=A_RELEVANT_MASK, n_random=n_random, tag="constrained")
        static_constrained, flip_constrained = _static_recovery(
            cand_constrained, flat_base, jac_before, pool, j_before, dim_a, dim_b)

        row = dict(seed=seed, baseline_drift=baseline_drift,
                   drift_unconstrained=drift_unconstrained, static_unconstrained=static_unconstrained,
                   flip_unconstrained=flip_unconstrained, b_only_energy_fraction=b_only_energy,
                   drift_constrained=drift_constrained, static_constrained=static_constrained,
                   flip_constrained=flip_constrained, baseline_static=baseline_static, baseline_flip=baseline_flip)
        rows.append(row)
        print(f"seed={seed}: baseline_drift={baseline_drift:.4f} | "
              f"unconstrained: drift={drift_unconstrained:.2e} (reduction={100*(1-drift_unconstrained/baseline_drift):.1f}%) "
              f"static={static_unconstrained*100:.1f}% flip={flip_unconstrained} b_only_frac={b_only_energy:.3f} | "
              f"constrained: drift={drift_constrained:.4f} (reduction={100*(1-drift_constrained/baseline_drift):.1f}%) "
              f"static={static_constrained*100:.1f}% flip={flip_constrained}", flush=True)
        _write_csv(RESULTS_DIR / "fully_general_attacker.csv", rows)

    print(f"\nDone. {len(rows)} rows written.")
    _summarize(rows)


def _summarize(rows):
    print("\n=== Fully general attacker summary (mean across seeds) ===")
    bd = np.mean([r["baseline_drift"] for r in rows])
    du = np.mean([r["drift_unconstrained"] for r in rows])
    dc = np.mean([r["drift_constrained"] for r in rows])
    n_flip_u = sum(1 for r in rows if r["flip_unconstrained"])
    n_flip_c = sum(1 for r in rows if r["flip_constrained"])
    print(f"  unconstrained: drift {bd:.4f} -> {du:.2e} ({100*(1-du/bd):.1f}% reduction), "
          f"flip {n_flip_u}/{len(rows)}, mean B-only energy fraction "
          f"{np.mean([r['b_only_energy_fraction'] for r in rows]):.3f}")
    print(f"  constrained (A-relevant only): drift {bd:.4f} -> {dc:.4f} ({100*(1-dc/bd):.1f}% reduction), "
          f"flip {n_flip_c}/{len(rows)}")
    print(f"  (compare Experiment 45b's ring-only constrained result: 30.3% mean reduction, flip 5/5)")


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
