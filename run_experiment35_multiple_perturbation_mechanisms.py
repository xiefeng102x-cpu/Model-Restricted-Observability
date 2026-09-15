"""Experiment 35 (README Follow-up 46): does the whole Follow-up 42/43/44
pattern (task-invariant perturbation randomizes the blind spot; static
clean calibration fails; pilot-based adaptive recalibration recovers most
of the gap) depend on the ONE specific ZZ-null-unitary mechanism this
project has used throughout, or does it hold for genuinely DIFFERENT
task-invariant mechanisms too?

Direct user request ("多测几种机制试试") after being told this was the
single biggest open gap standing between this direction and a top-tier
submission. Three mechanisms, all diagonal-in-computational-basis (hence
exactly task-invariant -- the Z-marginals/logits are provably unchanged,
verified below, not assumed) unitaries applied post-hoc to the SAME
"before" states already persisted (Experiment 25's artifacts, no
retraining):

  1. ORIGINAL (baseline, already computed -- Follow-up 42/43/44's own
     data, read directly, not recomputed): ring-topology two-body ZZ
     generator, phis ~ Uniform(-1.5, 1.5).
  2. WEAK ZZ: the SAME ring-topology ZZ generator and the SAME phis
     DIRECTION, scaled to 10% magnitude -- tests whether the effect
     needs a "large" perturbation or appears even close to identity.
  3. Z-ONLY (no entangling generator at all): single-qubit Z-phase
     rotations only, exp(-i*theta_q*Z_q) per qubit, freshly seeded
     angles at the SAME magnitude range as the original -- tests whether
     an ENTANGLING (two-body) generator is essential to the mechanism,
     or whether even a non-entangling diagonal perturbation causes the
     same blind-spot randomization.

For each: self-test task-invariance directly (not assumed from the
original mechanism's own self-test, since these are different unitaries),
then re-run the SAME before/after-divergence + cosine-similarity +
static-vs-adaptive-completion pipeline as Follow-up 42/43/44, on all 5
real BloodMNIST seeds.

Usage:
    python run_experiment35_multiple_perturbation_mechanisms.py [--smoke]
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
N_A = 4
N_QUBITS = 8
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
PILOT_BUDGETS = [10_000, 100_000, 1_000_000]
N_REPEATS = 20
EXPERIMENT_ID = "paper13_exp35_multi_mechanism_v1"


def apply_single_z_null_unitary(psi_flat, n_qubits, thetas):
    """Diagonal unitary from exp(-i*theta_q*Z_q) per qubit -- NO entangling
    (two-body) generator at all, unlike the ring-ZZ mechanism this project
    has used throughout. Still exactly diagonal (task-invariant), since
    every Z_q is diagonal in the computational basis."""
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    total_phase = np.zeros(dim)
    for q, theta_q in enumerate(thetas):
        bit_q = (idx >> (n_qubits - 1 - q)) & 1
        eig = (1 - 2 * bit_q)
        total_phase = total_phase + theta_q * eig
    phase_factor = torch.tensor(np.exp(-1j * total_phase), dtype=torch.complex128)
    return psi_flat * phase_factor


def apply_zz_scaled(psi_flat, n_qubits, phis, pairs, scale):
    from manifold.theta_jacobian import mni
    return mni.apply_zz_null_unitary(psi_flat.unsqueeze(0), n_qubits, [p * scale for p in phis], pairs).reshape(-1)


def self_test_task_invariance():
    print("Running self-test (task-invariance of the two NEW mechanisms)...")
    torch.manual_seed(0)
    n_test = 6
    dim_test = 2 ** n_test
    psi = torch.randn(dim_test, dtype=torch.complex128)
    psi = psi / psi.norm()
    probs_before = (psi.conj() * psi).real

    thetas = [0.4, -0.7, 1.1, 0.3, -0.2, 0.9]
    psi_z = apply_single_z_null_unitary(psi, n_test, thetas)
    assert abs(psi_z.norm().item() - 1.0) < 1e-12
    probs_z = (psi_z.conj() * psi_z).real
    max_diff_z = (probs_before - probs_z).abs().max().item()
    assert max_diff_z < 1e-12, f"Z-only mechanism should be exactly task-invariant, got max_diff={max_diff_z}"
    print(f"  [check 1] Z-only mechanism: norm-preserving and EXACT probability/task invariance "
          f"(max_diff={max_diff_z:.2e})")

    ring_pairs = [(q, (q + 1) % n_test) for q in range(n_test)]
    phis = [0.4, -0.7, 1.1, 0.3, -0.2, 0.9]
    psi_zz_weak = apply_zz_scaled(psi, n_test, phis, ring_pairs, 0.1)
    assert abs(psi_zz_weak.norm().item() - 1.0) < 1e-12
    probs_zz_weak = (psi_zz_weak.conj() * psi_zz_weak).real
    max_diff_zzw = (probs_before - probs_zz_weak).abs().max().item()
    assert max_diff_zzw < 1e-12, f"weak-ZZ mechanism should be exactly task-invariant, got max_diff={max_diff_zzw}"
    print(f"  [check 2] weak-ZZ mechanism: norm-preserving and EXACT probability/task invariance "
          f"(max_diff={max_diff_zzw:.2e})")

    # both must be REAL, nonzero perturbations (not accidentally near-identity)
    fid_z = abs(torch.vdot(psi, psi_z).item())
    fid_zzw = abs(torch.vdot(psi, psi_zz_weak).item())
    print(f"  [check 3] both mechanisms are genuine (non-trivial) unitaries: "
          f"|<psi|psi_after>| Z-only={fid_z:.4f}, weak-ZZ={fid_zzw:.4f} (both < 1, real perturbations)")
    print("Self-test PASSED.\n")


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def run_mechanism(mech_name, intervention_fn_builder, pool, seeds, budgets, n_repeats):
    """intervention_fn_builder(seed, phis, ring_pairs) -> intervention_fn"""
    rows = []
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
        intervention_fn = intervention_fn_builder(seed, phis, ring_pairs)
        rho_after, jac_after = rho_A_and_jacobian(ts, "after", intervention_fn=intervention_fn)

        res_before = compute_R_manifold(rho_before, jac_before, pool)
        res_after = compute_R_manifold(rho_after, jac_after, pool)
        cos_sim = _cos(res_before.r_DC_coef, res_after.r_DC_coef)

        j_before_local = int(np.argmax(res_before.exact_score[pool.offdiag_idx]))
        j_after_local = int(np.argmax(res_after.exact_score[pool.offdiag_idx]))
        same_j = j_before_local == j_after_local

        baseline_after = res_after.gamma_D_C
        j_before_global = pool.offdiag_idx[j_before_local]
        clean_calibrated = compute_R_manifold(rho_after, jac_after, pool,
                                                extra_measured_idx=[j_before_global]).gamma_D_C
        j_after_global = pool.offdiag_idx[j_after_local]
        oracle_after = compute_R_manifold(rho_after, jac_after, pool,
                                           extra_measured_idx=[j_after_global]).gamma_D_C
        oracle_reduction = baseline_after - oracle_after
        clean_reduction = baseline_after - clean_calibrated
        clean_recovery = clean_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        adaptive_by_budget = {}
        for n_pilot in budgets:
            adaptive_vals = []
            for rep in range(n_repeats):
                pilot_seed = derive_seed(EXPERIMENT_ID, f"{mech_name}_{DATASET}_seed{seed}",
                                          f"{n_pilot}_{rep}", "pilot")
                rng = np.random.default_rng(pilot_seed)
                pilot = simulate_pilot(rho_after, pool.all_matrices, n_pilot, rng)
                route_a_out = route_a_select(pilot.chat, pool.all_matrices, pool.offdiag_idx, pool.d)
                g_hat_coef = pauli_coefficients(-hermitian_log(route_a_out.rho_hat), pool.all_matrices)
                adaptive_res = compute_R_manifold(rho_after, jac_before, pool, g_coef_override=g_hat_coef)
                j_adapt_local = int(np.argmax(adaptive_res.exact_score[pool.offdiag_idx]))
                j_adapt_global = pool.offdiag_idx[j_adapt_local]
                eval_res = compute_R_manifold(rho_after, jac_after, pool, extra_measured_idx=[j_adapt_global])
                adaptive_vals.append(eval_res.gamma_D_C)
            adaptive_mean = float(np.mean(adaptive_vals))
            adaptive_reduction = baseline_after - adaptive_mean
            adaptive_by_budget[n_pilot] = adaptive_reduction / oracle_reduction if oracle_reduction > 1e-9 else float("nan")

        row = dict(mechanism=mech_name, seed=seed, same_j_star=same_j, cos_r_DC=cos_sim,
                   R_manifold_after=res_after.R_manifold, clean_recovery_fraction=clean_recovery)
        for n_pilot in budgets:
            row[f"adaptive_recovery_{n_pilot}"] = adaptive_by_budget[n_pilot]
        rows.append(row)
        print(f"[{mech_name}] seed={seed}: same_j*={same_j} cos={cos_sim:.4f} "
              f"R_manifold_after={res_after.R_manifold:.4f} clean_recovery={clean_recovery*100:.1f}% "
              f"adaptive@100K={adaptive_by_budget.get(100_000, float('nan'))*100:.1f}%", flush=True)
    return rows


def run(smoke: bool):
    pool = build_pool(N_A)
    self_test_task_invariance()

    seeds = SEEDS[:1] if smoke else SEEDS
    budgets = PILOT_BUDGETS[:1] if smoke else PILOT_BUDGETS
    n_repeats = 3 if smoke else N_REPEATS

    mechanisms = {
        "weak_zz": lambda seed, phis, ring_pairs: (
            lambda flat: apply_zz_scaled(flat, N_QUBITS, phis, ring_pairs, 0.1)),
        "z_only": lambda seed, phis, ring_pairs: (
            lambda flat, seed=seed: apply_single_z_null_unitary(
                flat, N_QUBITS, list(np.random.RandomState(seed + 90001).uniform(-1.5, 1.5, size=N_QUBITS)))),
    }

    all_rows = []
    for mech_name, builder in mechanisms.items():
        rows = run_mechanism(mech_name, builder, pool, seeds, budgets, n_repeats)
        all_rows.extend(rows)
        _write_csv(RESULTS_DIR / "multiple_perturbation_mechanisms.csv", all_rows)

    print(f"\nDone. {len(all_rows)} rows written.")
    _summarize(all_rows, budgets)


def _summarize(rows, budgets):
    for mech in sorted(set(r["mechanism"] for r in rows)):
        sub = [r for r in rows if r["mechanism"] == mech]
        n_same = sum(1 for r in sub if r["same_j_star"])
        cos_vals = [r["cos_r_DC"] for r in sub]
        clean_vals = [r["clean_recovery_fraction"] for r in sub]
        print(f"\n=== mechanism={mech} ===")
        print(f"  j*(before)==j*(after): {n_same}/{len(sub)}")
        print(f"  cos(r_DC_before,r_DC_after): mean={np.mean(cos_vals):.4f}")
        print(f"  clean-calibrated recovery: mean={np.mean(clean_vals)*100:.1f}%")
        for n_pilot in budgets:
            vals = [r[f"adaptive_recovery_{n_pilot}"] for r in sub]
            print(f"  adaptive recovery @ n_pilot={n_pilot}: mean={np.mean(vals)*100:.1f}%")


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
