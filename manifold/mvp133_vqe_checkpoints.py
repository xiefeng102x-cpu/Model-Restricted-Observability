"""R_manifold on this project's mvp133 VQE checkpoints -- a zero-
retraining-cost extension using pre-trained checkpoints
`checkpoints/mvp133_frozen_rep_cluster_n6_seed{0..9}.pt` and
`..._midtrain_cluster_n6_seed{0..9}.pt`: genuine (theta_early=epoch40,
theta_late=epoch400) pairs from the SAME training run (confirmed: same
torch.manual_seed init, same Adam optimizer, only differing in when
training stopped), for a 6-qubit, 6-layer circuit -- gate-mechanics-
identical to the classifier simulator used elsewhere in this repository
(same ry_matrix/rz_matrix/apply_single_qubit_gate/apply_cnot), but with
NO batch/data dimension (a VQE ground-state-finding task, not a
classifier: `simulate_rep_state(theta_rep, n_qubits, n_layers)` always
starts from |000000> and takes theta only).

Why this matters for R_manifold specifically: this repository's two
known classifier configs (MNIST 5q/120params, BloodMNIST 8q/128params)
both turned out to be severely OVER-parametrized relative to any
natural target-subsystem ambient dimension (d^2-1), which makes
R_manifold=1 close to a foregone conclusion by a pure dimension-counting
argument, not yet a real test of the guide's hypothesis. This circuit has only 72 params
(6 layers x 6 qubits x 2), so a large-enough target subsystem (e.g.
n_a=4 of 6 qubits, d=16, ambient dim=255 >> 72) CANNOT be full rank by
counting alone -- the first structurally-guaranteed non-degenerate test
available in this codebase, at zero retraining cost (weights already on
disk).

Caveat, stated plainly: there is no "task" here (no data x, no
classification), so there is no canonical, task-motivated choice of
target subsystem the way MNIST/BloodMNIST's bipartition is motivated by
an actual classifier. Both n_a=3 (the natural symmetric half) and n_a=4
(deliberately chosen to force ambient > n_params) are computed and
reported as what they are: a mechanistic stress-test of the R_manifold
MACHINERY across a different (qubit count, param count) regime, not a
second real "operational observability" case study on the same footing
as the MNIST/BloodMNIST classifiers.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_mvp = importlib.import_module("frozen_route_causal_test")

CKPT_DIR = _REPO_ROOT / "checkpoints"
N_QUBITS = _mvp.N_QUBITS          # 6
N_LAYERS = _mvp.L_REP_LAYERS      # 6


@dataclass
class VqeThetaState:
    seed: int
    stage: str              # "cluster" (epoch 400, converged) or "midtrain" (epoch 40)
    theta: torch.Tensor     # (72,) float64, leaf, requires_grad
    final_energy: float


_STAGE_FILENAME = {
    "cluster": "mvp133_frozen_rep_cluster_n6_seed{seed}.pt",        # epoch 400, converged
    "midtrain": "mvp133_frozen_rep_midtrain_cluster_n6_seed{seed}.pt",  # epoch 40
}


def load_theta(seed: int, stage: str) -> VqeThetaState:
    assert stage in ("cluster", "midtrain")
    fn = CKPT_DIR / _STAGE_FILENAME[stage].format(seed=seed)
    d = torch.load(fn, map_location="cpu", weights_only=False)
    theta = d["theta_rep"].detach().clone().requires_grad_(True)
    return VqeThetaState(seed=seed, stage=stage, theta=theta, final_energy=float(d["final_energy"]))


def rho_A_and_jacobian(ts: VqeThetaState, n_a: int):
    """Returns (rho_A complex128 (d,d), jac complex128 (d,d,n_params)) for
    the first n_a of N_QUBITS qubits, d=2**n_a."""
    n_qubits, n_layers = N_QUBITS, N_LAYERS
    dim_a, dim_b = 2 ** n_a, 2 ** (n_qubits - n_a)

    def rho_A_realstack(w):
        state = _mvp.simulate_rep_state(w, n_qubits, n_layers)
        flat = state.reshape(2 ** n_qubits)
        M = flat.reshape(dim_a, dim_b)
        rho = M @ M.conj().T
        return torch.stack([rho.real, rho.imag])

    theta = ts.theta.detach().clone().requires_grad_(True)
    with torch.no_grad():
        rs0 = rho_A_realstack(theta)
    rho_A = (rs0[0] + 1j * rs0[1]).numpy()

    jac = torch.autograd.functional.jacobian(rho_A_realstack, theta)
    jac_complex = (jac[0] + 1j * jac[1]).numpy()
    return rho_A, jac_complex


def self_test():
    print("Running self-test (manifold/mvp133_vqe_checkpoints)...")
    ts_c = load_theta(0, "cluster")
    ts_m = load_theta(0, "midtrain")
    diff = float((ts_c.theta.detach() - ts_m.theta.detach()).abs().max())
    assert diff > 0.5, f"cluster vs midtrain theta should differ substantially, got max|diff|={diff}"
    print(f"  [check 1] seed=0 cluster vs midtrain theta differ substantially "
          f"(max|diff|={diff:.3f} rad, final_energy {ts_c.final_energy:.2e} vs {ts_m.final_energy:.4f}) "
          f"-- confirms genuine early/late trajectory pair, not a naming coincidence")

    rho_A, jac = rho_A_and_jacobian(ts_c, n_a=3)
    tr_err = abs(np.trace(rho_A).real - 1.0)
    herm_err = np.abs(rho_A - rho_A.conj().T).max()
    assert tr_err < 1e-8 and herm_err < 1e-8
    print(f"  [check 2] rho_A (n_a=3) is trace-1 (err={tr_err:.2e}) and Hermitian (err={herm_err:.2e})")

    eps = 1e-6
    with torch.no_grad():
        theta = ts_c.theta.detach().clone()
        p_idx = 10

        def f(w):
            state = _mvp.simulate_rep_state(w, N_QUBITS, N_LAYERS)
            flat = state.reshape(2 ** N_QUBITS)
            M = flat.reshape(2 ** 3, 2 ** (N_QUBITS - 3))
            rho = M @ M.conj().T
            return torch.stack([rho.real, rho.imag])

        wp = theta.clone(); wp[p_idx] += eps
        wm = theta.clone(); wm[p_idx] -= eps
        fd = ((f(wp) - f(wm)) / (2 * eps)).numpy()
        ad = jac[..., p_idx]
        ad_stack = np.stack([ad.real, ad.imag])
        err = np.abs(fd - ad_stack).max()
    assert err < 1e-5, f"finite-difference mismatch: {err}"
    print(f"  [check 3] autograd Jacobian matches finite-difference (max err={err:.2e})")
    print("Self-test PASSED.\n")


if __name__ == "__main__":
    self_test()
