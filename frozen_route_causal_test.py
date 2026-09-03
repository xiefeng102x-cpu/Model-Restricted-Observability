"""
MVP-133 routing, corrected protocol: Method M1 from the original guide notes at 
quantum_task_information_routing_experiment_plan.md Section 8.4 --
"冻结表示，只训练末端路由" (freeze representation, train ONLY the routing
layer), explicitly called out there as "这是最关键的因果实验" (the most
critical causal experiment).

Motivation / correction: mvp133_routing_control.py and its entangling
variant trained theta_rep and theta_route JOINTLY via strict epoch-by-
epoch alternation, starting from random initialization for BOTH. Per this
new, more detailed guide, that protocol is actually "Baseline B1" (Section
8.2, "普通联合训练...用途：判断收益是否仅来自增加参数量" -- joint training
from scratch, used only as a capacity-control COMPARISON), NOT the primary
causal test. The guide's actual causal claim requires:
  1. Train the representation to convergence FIRST (plain VQE, no routing).
  2. FREEZE it completely (theta_rep receives no further gradient).
  3. Train ONLY theta_route (initialized near identity) to maximize the
     observable capture ratio R_obs (this script's J_route, same fixed
     12-operator pool as before) on that FIXED, unchanging representation.
  4. Verify the global task-information magnitude stays ~unchanged (guide's
     <2% ||Delta rho|| criterion, translated here to Z_t = ||Gamma_t||_F^2,
     the natural analog for a VQE/Hamiltonian task where there is no
     classification Delta-rho to speak of -- see note below).
  5. Check for "routing collapse" (guide Section 16.3): routing must not
     trivially "succeed" by degenerating the state into something with no
     real structure left.

Framework note: the new guide's own formalism (Delta rho = class-mean
difference) is built for CLASSIFICATION tasks and does not directly apply
to a VQE/Hamiltonian ground-state problem (no "classes" here) -- this
script keeps the Gamma_t = i[rho,H_task] framework (appropriate for VQE,
already validated across mvp133_pauli_tracking_smoketest.py /
mvp133_real_trajectory_tracking.py / mvp133_routing_control.py) and
applies the NEW guide's METHODOLOGY (freeze-then-route, not joint
alternation) on top of it, rather than switching formalisms entirely.

Routing architecture: single-qubit-only (RY,RZ per qubit, no entangler),
matching the guide's own recommended FIRST step (Section 4.3, "R1: 末端
路由层...适合作为首个因果实验", with distributed/entangling routing
explicitly deferred to a later step, Section 4.3 "R2").

Success criteria tracked (guide Section 14.2, adapted):
  - Delta R_obs = J_route_after - J_route_before (guide wants >=0.15)
  - Z_t stability: |Z_t_after - Z_t_before| / Z_t_before (guide wants <0.02
    for the Delta-rho analog; reported here for the Gamma_t analog)
  - K_0.8 ratio: k_0.80(after) / k_0.80(before) (guide wants <=0.6)
  - routing collapse check: purity of each qubit's reduced state should
    NOT collapse to a trivial computational basis product state

Model checkpointing (established convention): theta_rep checkpoints saved
to checkpoints/ so this frozen representation can be reused by
future scripts without retraining.

Standalone rebuild: fresh, self-contained code, no edits to / imports of
any other script in this repository.

SCOPE: cluster task, n=6, SEEDS=10 (guide Section 7.3 minimum), 400 epochs
representation training + 200 epochs routing-only training.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 6
L_REP_LAYERS = 6
N_ROUTE_LAYERS = 1
SEEDS = list(range(10))
REP_EPOCHS = 400
ROUTE_EPOCHS = 200
LR_REP = 0.05
LR_ROUTE = 0.05

I2 = np.eye(2, dtype=np.complex128)
X_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_np = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_np = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI_1Q_NP = {"I": I2, "X": X_np, "Y": Y_np, "Z": Z_np}


def pauli_string_op_np(labels):
    op = PAULI_1Q_NP[labels[0]]
    for lab in labels[1:]:
        op = np.kron(op, PAULI_1Q_NP[lab])
    return op


def all_pauli_strings(n):
    return list(itertools.product("IXYZ", repeat=n))


def pauli_weight(labels):
    return sum(1 for c in labels if c != "I")


def cluster_task_hamiltonian(n):
    K_list = []
    for i in range(n):
        labels = ["I"] * n
        if i == 0:
            labels[0] = "X"; labels[1] = "Z"
        elif i == n - 1:
            labels[n - 2] = "Z"; labels[n - 1] = "X"
        else:
            labels[i - 1] = "Z"; labels[i] = "X"; labels[i + 1] = "Z"
        K_list.append(pauli_string_op_np(tuple(labels)))
    dim = 2 ** n
    Iop = np.eye(dim, dtype=np.complex128)
    h_list = [(Iop - K) / 2.0 for K in K_list]
    H_task = sum(h_list)
    return H_task, h_list, K_list


def commutator_gamma_np(rho, H):
    return 1j * (rho @ H - H @ rho)


def full_pauli_decomposition_np(Gamma, n, strings):
    dim = 2 ** n
    ops = np.stack([pauli_string_op_np(s).reshape(-1) for s in strings], axis=0)
    return (ops.conj() @ Gamma.reshape(-1)) / dim


def task_metrics(gammas, strings):
    weights = np.abs(gammas) ** 2
    Z_t = float(weights.sum())
    if Z_t < 1e-15:
        return {"Z_t": 0.0, "d_eff": 0.0, "k_0.10": 0, "k_0.05": 0, "k_0.80": 0, "mean_weight": 0.0}
    p_P = weights / Z_t
    d_eff = float(1.0 / np.sum(p_P ** 2))
    sorted_p = np.sort(p_P)[::-1]
    cum = np.cumsum(sorted_p)

    def k_eps(eps):
        return int(np.searchsorted(cum, 1.0 - eps) + 1)

    weight_arr = np.array([pauli_weight(s) for s in strings])
    mean_weight = float(np.sum(p_P * weight_arr))
    return {"Z_t": Z_t, "d_eff": d_eff, "k_0.10": k_eps(0.10), "k_0.05": k_eps(0.05),
            "k_0.80": k_eps(0.20), "mean_weight": mean_weight}


def ry_matrix(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rz_matrix(theta):
    e_m = torch.exp(-1j * theta / 2)
    e_p = torch.exp(1j * theta / 2)
    z = torch.zeros((), dtype=torch.complex128)
    return torch.stack([torch.stack([e_m, z]), torch.stack([z, e_p])])


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    axis = qubit
    state = torch.movedim(state, axis, 0)
    shape = state.shape
    state = state.reshape(2, -1)
    state = torch.einsum("ij,jk->ik", gate.to(state.dtype), state)
    state = state.reshape(shape)
    state = torch.movedim(state, 0, axis)
    return state


def apply_cnot(state, control, target, n_qubits):
    state = torch.movedim(state, (control, target), (0, 1))
    s0 = state[0].clone()
    s1 = state[1].clone()
    new_s1 = torch.flip(s1, dims=[0])
    state = torch.stack([s0, new_s1], dim=0)
    state = torch.movedim(state, (0, 1), (control, target))
    return state


def simulate_rep_state(theta_rep, n_qubits, n_layers):
    state = torch.zeros((2,) * n_qubits, dtype=torch.complex128)
    state.reshape(-1)[0] = 1.0
    wr = theta_rep.reshape(n_layers, n_qubits, 2)
    for l in range(n_layers):
        for q in range(n_qubits):
            g_ry = ry_matrix(wr[l, q, 0].to(torch.float64)).to(torch.complex128)
            state = apply_single_qubit_gate(state, g_ry, q, n_qubits)
            g_rz = rz_matrix(wr[l, q, 1].to(torch.float64))
            state = apply_single_qubit_gate(state, g_rz, q, n_qubits)
        for q in range(n_qubits):
            state = apply_cnot(state, q, (q + 1) % n_qubits, n_qubits)
    return state


def apply_route_layer(state, theta_route, n_qubits):
    """R1 (guide Section 4.3): single-qubit-only, no entangler -- the
    guide's recommended FIRST causal-test architecture."""
    wr = theta_route.reshape(N_ROUTE_LAYERS, n_qubits, 2)
    for l in range(N_ROUTE_LAYERS):
        for q in range(n_qubits):
            g_ry = ry_matrix(wr[l, q, 0].to(torch.float64)).to(torch.complex128)
            state = apply_single_qubit_gate(state, g_ry, q, n_qubits)
            g_rz = rz_matrix(wr[l, q, 1].to(torch.float64))
            state = apply_single_qubit_gate(state, g_rz, q, n_qubits)
    return state


def build_fixed_pool_torch(n_qubits, K_list_np):
    pool_ops = []
    for q in range(n_qubits):
        labels = ["I"] * n_qubits
        labels[q] = "Z"
        pool_ops.append(pauli_string_op_np(tuple(labels)))
    pool_ops.extend(K_list_np)
    return torch.tensor(np.stack(pool_ops, axis=0), dtype=torch.complex128)


def differentiable_gamma_and_jroute(psi, H_task_torch, pool_torch):
    rho = torch.outer(psi, psi.conj())
    Gamma = 1j * (rho @ H_task_torch - H_task_torch @ rho)
    Z_t = torch.real(torch.einsum("ij,ij->", Gamma, Gamma.conj()))
    tr_P_Gamma = torch.einsum("pij,ji->p", pool_torch, Gamma)
    dim = pool_torch.shape[-1]
    pool_weight = torch.sum(torch.abs(tr_P_Gamma) ** 2) / dim
    J_route = pool_weight / (Z_t + 1e-15)
    return Gamma, J_route, Z_t


def vqe_energy_torch(psi, H_task_torch):
    Hpsi = H_task_torch @ psi
    return torch.real(torch.vdot(psi, Hpsi))


def qubit_purity(psi_np, n_qubits, qubit):
    """Purity Tr(rho_q^2) of a single qubit's reduced state -- for a
    ROUTING-COLLAPSE check: if routing degenerates the state into
    something trivial, individual-qubit purities would all jump to 1
    (pure product-state computational-basis collapse) in a way disconnected
    from the task; this is a sanity signal, not by itself proof either way."""
    dim = 2 ** n_qubits
    psi_tensor = psi_np.reshape((2,) * n_qubits)
    axis_order = [qubit] + [q for q in range(n_qubits) if q != qubit]
    s = np.transpose(psi_tensor, axis_order).reshape(2, -1)
    rho_q = s @ s.conj().T
    return float(np.real(np.trace(rho_q @ rho_q)))


def self_test():
    print("Running self-test (Method M1 frozen-representation causal test)...")
    n = N_QUBITS
    H_task_np, h_list_np, K_list_np = cluster_task_hamiltonian(n)
    H_task_torch = torch.tensor(H_task_np, dtype=torch.complex128)
    pool_torch = build_fixed_pool_torch(n, K_list_np)
    strings = all_pauli_strings(n)

    torch.manual_seed(1)
    theta_rep_test = torch.randn(L_REP_LAYERS * n * 2, dtype=torch.float64, requires_grad=True)
    theta_route_test = torch.zeros(N_ROUTE_LAYERS * n * 2, dtype=torch.float64, requires_grad=True)
    state_rep = simulate_rep_state(theta_rep_test, n, L_REP_LAYERS)
    state_final = apply_route_layer(state_rep, theta_route_test, n)
    psi = state_final.reshape(-1)

    Gamma_torch, J_route, Z_t_torch = differentiable_gamma_and_jroute(psi, H_task_torch, pool_torch)
    psi_np = psi.detach().numpy()
    rho_np = np.outer(psi_np, psi_np.conj())
    Gamma_np = commutator_gamma_np(rho_np, H_task_np)
    gamma_diff = np.linalg.norm(Gamma_torch.detach().numpy() - Gamma_np)
    assert gamma_diff < 1e-8, f"differentiable Gamma disagrees with numpy: {gamma_diff:.2e}"

    gammas_full_np = full_pauli_decomposition_np(Gamma_np, n, strings)
    weights_full = np.abs(gammas_full_np) ** 2
    Z_t_np = weights_full.sum()
    pool_labels = []
    for q in range(n):
        lab = ["I"] * n; lab[q] = "Z"; pool_labels.append(tuple(lab))
    for i in range(n):
        if i == 0:
            lab = ["I"] * n; lab[0] = "X"; lab[1] = "Z"
        elif i == n - 1:
            lab = ["I"] * n; lab[n - 2] = "Z"; lab[n - 1] = "X"
        else:
            lab = ["I"] * n; lab[i - 1] = "Z"; lab[i] = "X"; lab[i + 1] = "Z"
        pool_labels.append(tuple(lab))
    idx_pool = [strings.index(lab) for lab in pool_labels]
    pool_weight_np = weights_full[idx_pool].sum()
    J_route_np = pool_weight_np / (Z_t_np + 1e-15)
    j_diff = abs(J_route.item() - J_route_np)
    assert j_diff < 1e-6, f"differentiable J_route disagrees with numpy: {j_diff:.2e}"
    print(f"  cross-checks OK: Gamma err={gamma_diff:.2e}, J_route err={j_diff:.2e}")

    grad = torch.autograd.grad(J_route, theta_route_test, retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum().item() > 0
    print(f"  gradient check OK: ||d(J_route)/d(theta_route)||={grad.norm().item():.4e}")

    print("Self-test PASSED.\n")
    return H_task_np, H_task_torch, pool_torch, strings, K_list_np


def checkpoint_path(seed):
    return CKPT_DIR / f"mvp133_frozen_rep_cluster_n{N_QUBITS}_seed{seed}.pt"


def train_representation(seed, H_task_torch, epochs=REP_EPOCHS):
    ckpt = checkpoint_path(seed)
    n = N_QUBITS
    if ckpt.exists():
        print(f"  [checkpoint] loading frozen representation from {ckpt}")
        return torch.load(ckpt, weights_only=True)["theta_rep"]

    torch.manual_seed(seed)
    theta_rep = (torch.randn(L_REP_LAYERS * n * 2, dtype=torch.float64) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([theta_rep], lr=LR_REP)
    for epoch in range(epochs):
        state = simulate_rep_state(theta_rep, n, L_REP_LAYERS)
        psi = state.reshape(-1)
        energy = vqe_energy_torch(psi, H_task_torch)
        opt.zero_grad()
        energy.backward()
        opt.step()
    with torch.no_grad():
        final_energy = vqe_energy_torch(simulate_rep_state(theta_rep, n, L_REP_LAYERS).reshape(-1), H_task_torch)
    print(f"  seed={seed}: representation trained, final energy={final_energy.item():.6f}")
    theta_rep_d = theta_rep.detach().clone()
    torch.save({"theta_rep": theta_rep_d, "seed": seed, "final_energy": final_energy.item()}, ckpt)
    return theta_rep_d


def route_only_training(seed, theta_rep_frozen, H_task_np, H_task_torch, pool_torch, strings,
                         epochs=ROUTE_EPOCHS):
    n = N_QUBITS
    torch.manual_seed(seed + 10000)  # separate stream for route-layer init/training randomness
    theta_route = torch.zeros(N_ROUTE_LAYERS * n * 2, dtype=torch.float64, requires_grad=True)
    opt_route = torch.optim.Adam([theta_route], lr=LR_ROUTE)

    with torch.no_grad():
        state_rep = simulate_rep_state(theta_rep_frozen, n, L_REP_LAYERS)
        psi_rep = state_rep.reshape(-1)
        energy_frozen = vqe_energy_torch(psi_rep, H_task_torch).item()
        m_before = task_metrics(full_pauli_decomposition_np(
            commutator_gamma_np(np.outer(psi_rep.numpy(), psi_rep.numpy().conj()), H_task_np), n, strings), strings)
        _, J_before_t, _ = differentiable_gamma_and_jroute(psi_rep, H_task_torch, pool_torch)
        J_before = J_before_t.item()
        purity_before = [qubit_purity(psi_rep.numpy(), n, q) for q in range(n)]

    for epoch in range(epochs):
        state_final = apply_route_layer(state_rep, theta_route, n)
        psi_final = state_final.reshape(-1)
        _, J_route_val, _ = differentiable_gamma_and_jroute(psi_final, H_task_torch, pool_torch)
        loss = -J_route_val
        opt_route.zero_grad()
        loss.backward()
        opt_route.step()

    with torch.no_grad():
        state_final = apply_route_layer(state_rep, theta_route, n)
        psi_final = state_final.reshape(-1)
        energy_after = vqe_energy_torch(psi_final, H_task_torch).item()  # sanity only; NOT the training signal
        _, J_after_t, Z_t_after_t = differentiable_gamma_and_jroute(psi_final, H_task_torch, pool_torch)
        J_after = J_after_t.item()
        m_after = task_metrics(full_pauli_decomposition_np(
            commutator_gamma_np(np.outer(psi_final.numpy(), psi_final.numpy().conj()), H_task_np), n, strings), strings)
        purity_after = [qubit_purity(psi_final.numpy(), n, q) for q in range(n)]

    row = {
        "seed": seed, "energy_frozen_rep": energy_frozen, "energy_after_routing_view": energy_after,
        "J_route_before": J_before, "J_route_after": J_after, "delta_R_obs": J_after - J_before,
        "Z_t_before": m_before["Z_t"], "Z_t_after": m_after["Z_t"],
        "Z_t_relative_change": abs(m_after["Z_t"] - m_before["Z_t"]) / (m_before["Z_t"] + 1e-15),
        "d_eff_before": m_before["d_eff"], "d_eff_after": m_after["d_eff"],
        "k80_before": m_before["k_0.80"], "k80_after": m_after["k_0.80"],
        "k80_ratio": m_after["k_0.80"] / (m_before["k_0.80"] + 1e-9),
        "mean_purity_before": float(np.mean(purity_before)), "mean_purity_after": float(np.mean(purity_after)),
    }
    return row


def main():
    t_start = time.time()
    H_task_np, H_task_torch, pool_torch, strings, K_list_np = self_test()

    rows = []
    for seed in SEEDS:
        print(f"\n=== seed={seed}: Phase A (train representation to convergence) ===")
        t0 = time.time()
        theta_rep_frozen = train_representation(seed, H_task_torch)
        print(f"  done in {time.time()-t0:.1f}s")

        print(f"=== seed={seed}: Phase B (freeze representation, train ONLY routing) ===")
        t0 = time.time()
        row = route_only_training(seed, theta_rep_frozen, H_task_np, H_task_torch, pool_torch, strings)
        print(f"  done in {time.time()-t0:.1f}s -> "
              f"J_route: {row['J_route_before']:.4f} -> {row['J_route_after']:.4f} "
              f"(Delta={row['delta_R_obs']:.4f}), Z_t_rel_change={row['Z_t_relative_change']:.4f}, "
              f"k80: {row['k80_before']} -> {row['k80_after']} (ratio={row['k80_ratio']:.3f})")
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_DIR / "mvp133_frozen_route_causal_test_results.csv", index=False)

    df = pd.DataFrame(rows)
    print(f"\n\n=== METHOD M1 CAUSAL TEST SUMMARY (total wall time {time.time()-t_start:.1f}s, {len(SEEDS)} seeds) ===")
    print(df[["J_route_before", "J_route_after", "delta_R_obs", "Z_t_relative_change",
              "k80_before", "k80_after", "k80_ratio", "mean_purity_before", "mean_purity_after"]]
          .agg(["mean", "std", "median"]).to_string())

    n_pos = (df["delta_R_obs"] > 0).sum()
    median_delta = df["delta_R_obs"].median()
    median_k80_ratio = df["k80_ratio"].median()
    median_zt_change = df["Z_t_relative_change"].median()
    print(f"\n=== Guide Section 14.2 success-criteria check ===")
    print(f"  seeds with Delta R_obs > 0: {n_pos}/{len(SEEDS)} (guide wants >= 8/10)")
    print(f"  median Delta R_obs: {median_delta:.4f} (guide wants >= 0.15)")
    print(f"  median k80 ratio (after/before): {median_k80_ratio:.4f} (guide wants <= 0.6 for K_0.8 criterion)")
    print(f"  median Z_t relative change: {median_zt_change:.4f} (guide wants < 0.02 for the Delta-rho analog)")
    passed = (n_pos >= 8) and (median_delta >= 0.15) and (median_k80_ratio <= 0.6) and (median_zt_change < 0.02)
    print(f"\n  {'PASSES' if passed else 'DOES NOT PASS'} the guide's pre-registered success criteria (adapted to Gamma_t).")


if __name__ == "__main__":
    main()
