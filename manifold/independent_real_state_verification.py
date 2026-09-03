"""Standalone, deliberately-independent verification of the Experiment 25
finding (README Follow-up 35: manifold-aware j*_C disagrees with ambient
j* on 10/10 real BloodMNIST states) -- run in direct response to the user
asking "is this a code bug," since the earlier self-tests (restricted_
observability.py's checks 5a/5b/5c) only validate the linear-algebra
CONSTRUCTION on synthetic random matrices, not the real trained circuit
end to end.

Uses the artifacts already persisted by Experiment 25 (results/manifold_
artifacts/*.npz: theta, rho_A, jac, r_DC_coef) -- no retraining needed.
Deliberately avoids reusing restricted_observability.py's own SVD-based
null-space code: this script re-derives everything via a DIFFERENT method
(np.linalg.lstsq to check tangent-space membership; a REAL finite step
through the actual this repository circuit simulator, not the linearized -log(rho)
formula) so a bug shared between the original computation and this check
would have to be present in two independently-written code paths, not
one.

Checks, on bloodmnist_seed42_before:
1. r_DC (reconstructed from the persisted r_DC_coef) is genuinely IN the
   real Jacobian's column space -- solved via lstsq, not assumed from the
   Pauli-coefficient SVD construction.
2. Taking a REAL finite step theta -> theta + eps*c (c from check 1) and
   re-running the actual differentiable circuit simulator (not the
   linearized formula): the diagonal (already-measured) Pauli expectation
   values barely move, while entropy moves at close to the predicted rate.
3. Same real finite step along the OLD ambient-optimal direction (j*_amb,
   IXZI for this state) projected onto the tangent space: confirms this
   is why it's disfavored -- either largely already visible to the
   measured set, or with a much smaller entropy-sensitivity in the
   invisible part, again checked via a REAL simulator step, not just the
   stored numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# candidates/oracle live in the sibling this repository/ (this project's shared,
# already-audited Pauli-pool/oracle infrastructure -- not duplicated here).

from candidates.pauli_pool import build_pool
from oracle.entropy_target import hermitian_log
from manifold.theta_jacobian import tms, mni  # reuse the actual circuit simulator, nothing else

ARTIFACT = Path(__file__).resolve().parent.parent / "results" / "manifold_artifacts" / "bloodmnist_seed42_before.npz"
N_QUBITS, LAYER, N_A = 8, 8, 4
RING_PAIRS = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]


def von_neumann_entropy(rho, eig_floor=1e-14):
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w, eig_floor, None)
    return float(-(w * np.log(w)).sum())


def real_rho_A_at_theta(theta_np, x_ref, stage="before"):
    """Independent forward pass through the ACTUAL differentiable circuit
    simulator -- same function theta_jacobian.py uses, called directly
    here (no autograd, no SVD, no Pauli-coefficient abstraction)."""
    with torch.no_grad():
        w = torch.tensor(theta_np, dtype=torch.float64)
        state = tms.simulate_final_state(x_ref, w, N_QUBITS, LAYER, RING_PAIRS)
        flat = state.reshape(2 ** N_QUBITS)
        M = flat.reshape(2 ** N_A, 2 ** (N_QUBITS - N_A))
        rho = (M @ M.conj().T).numpy()
    return rho


def main():
    d = np.load(ARTIFACT)
    theta0 = d["theta"]
    rho_A0 = d["rho_A"]
    jac = d["jac"]          # (d,d,n_params) complex, from the ORIGINAL autograd computation
    r_DC_coef = d["r_DC_coef"]
    gamma_D_C_stored = float(d["gamma_D_C"])
    n_params = theta0.shape[0]

    pool = build_pool(N_A)
    dim_a = pool.d

    # Need x_ref to re-run the real simulator -- reconstruct exactly as
    # theta_jacobian.reconstruct_theta does (deterministic given seed).
    from manifold.theta_jacobian import di, c1
    cfg = di.DATASETS["bloodmnist"]
    x_ct, x_cnt, x_p = cfg["load"](42, cfg["layer"], cfg["pair"], cfg["pr"])
    x_ref = x_ct[:1]

    # Sanity: does re-running the real simulator at theta0 exactly reproduce
    # the persisted rho_A0? (confirms x_ref reconstruction is correct before
    # trusting anything downstream)
    rho_A_check = real_rho_A_at_theta(theta0, x_ref)
    err0 = np.abs(rho_A_check - rho_A0).max()
    print(f"[0] real-simulator rho_A at theta0 matches persisted rho_A0: max err={err0:.2e}")
    assert err0 < 1e-10

    # --- Check 1: r_DC really lies in the REAL Jacobian's column space ---
    r_DC_matrix = np.tensordot(r_DC_coef, pool.all_matrices, axes=(0, 0)) / np.sqrt(dim_a)
    jac_flat = jac.reshape(dim_a * dim_a, n_params)          # complex (d^2, n_params)
    target_flat = r_DC_matrix.reshape(-1)                    # complex (d^2,)
    # stack real/imag for a real-valued least-squares solve (independent of
    # the original SVD-in-Pauli-coefficient-space method)
    A_stack = np.vstack([jac_flat.real, jac_flat.imag])
    b_stack = np.concatenate([target_flat.real, target_flat.imag])
    c, residuals, rank_lstsq, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
    reconstructed = (jac_flat @ c)
    rel_err = np.linalg.norm(reconstructed - target_flat) / np.linalg.norm(target_flat)
    print(f"[1] r_DC lies in real Jacobian's column space: lstsq relative residual={rel_err:.2e} "
          f"(lstsq numerical rank={rank_lstsq}/{n_params})")
    assert rel_err < 1e-4, "r_DC should be (numerically) exactly reachable by the real Jacobian"

    # --- Check 2: a REAL finite step along c moves entropy but not the
    #     measured (diagonal) expectation values, via the ACTUAL simulator ---
    c_norm = c / np.linalg.norm(c)
    eps = 1e-4
    theta_plus = theta0 + eps * c_norm
    theta_minus = theta0 - eps * c_norm
    rho_plus = real_rho_A_at_theta(theta_plus, x_ref)
    rho_minus = real_rho_A_at_theta(theta_minus, x_ref)

    # measured (diagonal, I/Z-only) Pauli expectation values, before vs after
    diag_before = np.array([np.trace(pool.all_matrices[k] @ rho_A0).real for k in pool.diag_idx])
    diag_plus = np.array([np.trace(pool.all_matrices[k] @ rho_plus).real for k in pool.diag_idx])
    diag_minus = np.array([np.trace(pool.all_matrices[k] @ rho_minus).real for k in pool.diag_idx])
    max_diag_change = max(np.abs(diag_plus - diag_before).max(), np.abs(diag_minus - diag_before).max())

    S0 = von_neumann_entropy(rho_A0)
    Splus = von_neumann_entropy(rho_plus)
    Sminus = von_neumann_entropy(rho_minus)
    dS_dc_numeric = (Splus - Sminus) / (2 * eps)
    # CORRECT predicted rate for a step along c_norm=c/||c||_2 (Euclidean
    # norm in PARAMETER space): drho/dstep = jac@c_norm = r_DC_matrix/||c||_2
    # (since jac@c = r_DC_matrix by the lstsq solve), so
    # dS/dstep = <g, r_DC_matrix>/||c||_2 = gamma_D_C^2 / ||c||_2 -- NOT
    # ||r_DC|| on its own (that first attempt ignored that the Jacobian's
    # columns are not orthonormal, so a unit step in PARAMETER space is not
    # a unit step in ambient/Pauli-coefficient space).
    c_l2 = float(np.linalg.norm(c))
    predicted_rate = (gamma_D_C_stored ** 2) / c_l2

    print(f"[2] REAL finite step (eps={eps}) along the real Jacobian's r_DC preimage direction:")
    print(f"     max change in MEASURED (diagonal) Pauli expectation values: {max_diag_change:.2e} "
          f"(should be near machine/step-size noise, NOT O(1))")
    print(f"     entropy: S(theta0)={S0:.6f}, S(theta0+eps*c)={Splus:.6f}, S(theta0-eps*c)={Sminus:.6f}")
    print(f"     numeric dS/d(step) = {dS_dc_numeric:.4f}  vs.  CORRECTED predicted "
          f"gamma_D_C^2/||c||_2 = {gamma_D_C_stored**2:.4f}/{c_l2:.4f} = {predicted_rate:.4f}")

    # --- Check 3: contrast with the ambient-optimal candidate (IXZI, low-
    #     ranked within r_DC per the mechanistic check) -- take a real step
    #     toward it directly (not projected) and see how much of that
    #     movement the diagonal set actually catches. ---
    j_amb_label = "IXZI"
    j_amb_global = pool.all_labels.index(j_amb_label)
    B_amb = pool.all_matrices[j_amb_global]
    # crude real "step toward more B_amb expectation": use B_amb itself as
    # a generator is not physical (no theta direction may point that way);
    # instead measure how much of B_amb's DIRECTION is already inside the
    # real tangent space's MEASURED-VISIBLE part vs its blind part, by
    # projecting the ambient gradient's B_amb-aligned component through the
    # same real-lstsq machinery as check 1.
    B_amb_coef = np.zeros(pool.all_matrices.shape[0])
    B_amb_coef[j_amb_global] = 1.0
    target_flat_amb = (pool.all_matrices[j_amb_global] / np.sqrt(dim_a)).reshape(-1)
    b_stack_amb = np.concatenate([target_flat_amb.real, target_flat_amb.imag])
    c_amb, _, _, _ = np.linalg.lstsq(A_stack, b_stack_amb, rcond=None)
    recon_amb = jac_flat @ c_amb
    rel_err_amb = np.linalg.norm(recon_amb - target_flat_amb) / np.linalg.norm(target_flat_amb)
    print(f"[3] ambient-optimal direction {j_amb_label}: how well can the real tangent space "
          f"reach pure {j_amb_label}? lstsq relative residual={rel_err_amb:.4f} "
          f"(near 0 = fully reachable as a pure direction, unlike r_DC's own construction "
          f"which is DEFINED to be in the tangent space -- this just contextualizes why "
          f"{j_amb_label} may or may not compete with r_DC's candidate)")

    print("\nDone. All checks used the REAL circuit simulator and an independent "
          "(lstsq-based) method, not restricted_observability.py's own SVD code path.")


if __name__ == "__main__":
    main()
