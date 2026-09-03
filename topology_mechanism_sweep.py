"""
topology_mechanism_sweep.py
====================================
Tests whether the "measurement-alignment" mechanism found in
qml_adaptive_nc_project (the NC-rejected / npjQI-pending / Neural-Networks-
under-review paper "physical vs algorithmic security boundary") is genuinely
TOPOLOGY-INDEPENDENT, as currently only argued analytically, not empirically.

Project memory records this explicitly: "第二电路族 | P3 | 理论论证替代 |
Bloch球约束拓扑无关，可理论说明" -- i.e. the second-circuit-topology
falsification test listed in the theory blueprint's own "competing
explanations to exclude" table (THEORY_BLUEPRINT_ZH.md section 6: "结果是
特定电路拓扑偶然" / "result is circuit-topology-specific") was deprioritized
and replaced with a theoretical argument instead of being run. This script
runs it.

Mechanism under test (from paper_nc_adaptive/code/e2_nullspace_attack.py):
  r_hidden = ||Proj_{ker J_M}(g)|| / ||g||
where J_M = d(XYZ features)/d(input) and g = d(target logit)/d(input). A low
r_hidden means the attack-relevant gradient direction is nearly orthogonal to
the detector's blind null-space -- i.e. the attacker cannot move toward
successful attack without also moving in a detector-visible direction. This
is the paper's core positive mechanism (Proposition 3 in THEORY_BLUEPRINT_ZH.md).

Sweep: entangling topology across increasing connectivity, n_qubits=5,
n_layers=12 (matching the paper's core "L=12" experiment):
  linear  (4 edges, open chain, least connected)
  ring    (5 edges, the project's existing default)
  all2all (10 edges, fully connected, most connected)

Data: REAL MNIST t7_vs_t0 poisoned dataset (seed=42, pr=0.1), loaded
read-only from data/Mnist/detail_single_samle/... -- the exact same data
every other script in qml_adaptive_nc_project uses. Never modified.

Two-phase protocol, each phase matching a specific real script exactly
(verified by direct reading, not assumption -- an earlier version of this
script incorrectly ran a 3-term data-poisoning loss in Phase 1, which does
NOT match the real pipeline and was corrected after auditing
data_generation/scb_mnist_baseline.py):
  Phase 1 = _train_clean_baseline() in data_generation/scb_mnist_baseline.py:
            CLEAN-ONLY 2-term CE (target_clean vs non_target_clean), Adam
            lr=0.003, 200 epochs, batch=64. No poisoned/triggered samples
            appear anywhere in this phase.
  Phase 2 = adaptive_qnn_fast_with_asr.py: starting from the Phase-1 clean
            weights, gradient descent (SGD momentum=0, lr=0.05, 200 steps)
            on CIRCUIT WEIGHTS ONLY (classifier frozen from Phase 1),
            minimizing the Mahalanobis distance of poisoned_non_target's
            XYZ features toward a clean reference fit once at the start.
            This weight-space attack is what actually CREATES the backdoor
            behavior (ASR) -- there is no data-poisoning training step at
            all in the real pipeline this script reproduces.

ALL THREE topologies (including ring) are retrained from scratch through
THIS SAME self-contained pipeline, rather than reusing the project's
existing ring checkpoint -- reusing it would confound "topology" with
"which pipeline produced these weights". The real ring checkpoint's
published numbers are used only as an external sanity-check reference
(is my retrained ring in the same ballpark?), never as the actual ring
datapoint in the topology trend.

Two deliberate, documented simplifications vs the original scripts
(justified because this is a simulated ablation, never touching hardware):
  1. Gradients (Phase-2 attack AND the E2 Jacobian) use exact PyTorch
     autograd instead of parameter-shift / finite-difference. On an exact
     simulator these are mathematically identical up to numerical precision
     -- parameter-shift only matters because real hardware cannot backprop.
     Verified by self-test (autograd Jacobian vs finite-difference Jacobian).
  2. Single seed (42) for this first pass, matching the explicit "look at
     the trend first" framing -- more seeds is a natural follow-up once the
     single-seed trend shape is known.

Standalone rebuild: paper_nc_adaptive/code/*.py is imported read-only
(fast_circuit.py, for the Ring-topology cross-validation self-test only)
and never modified, matching this project's standing convention of never
editing original project files.

Self-tests (run before any training):
  1. Ring-topology torch circuit forward pass matches the REAL
     fast_circuit.py (numpy) output on real data samples.
  2. Exact-autograd Jacobian of XYZ features w.r.t. input matches
     finite-difference Jacobian (eps=1e-4, matching e2_nullspace_attack.py).
  3. Null-space projector: J_M @ P_null ~= 0.
  4. Mahalanobis/LedoitWolf sanity: distance at the fitted mean is ~0.

Outputs (results/):
  - topology_sweep_phase1_history.csv
  - topology_sweep_phase2_history.csv
  - topology_sweep_e2_summary.csv   <- the key trend table
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/Mnist/detail_single_samle"
NC_PROJECT_CODE = ROOT / "qml_adaptive_nc_project" / "paper_nc_adaptive" / "code"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
LAYER = 12
PAIR = "t7_vs_t0"
PR = 0.1
N_QUBITS = 5
FEATURE_DIM = 32
TOPOLOGIES = ["linear", "ring", "all2all"]

PHASE1_EPOCHS = 200  # matches EnhancedSCBConfig.qnn_epochs (real training script, _train_clean_baseline)
PHASE1_LR = 0.003    # matches EnhancedSCBConfig.qnn_lr
PHASE1_BATCH = 64    # matches EnhancedSCBConfig.batch_size
PHASE2_STEPS = 200
PHASE2_LR = 0.05
PHASE2_EVAL_EVERY = 10
E2_N_STEPS = 100
E2_STEP_FRAC = 0.02
E2_EPS = 1e-4
E2_SVD_THRESH = 1e-6


# =========================================================================
# Topology-parametrized circuit (torch, complex128, autograd-differentiable)
# =========================================================================

def get_entangling_pairs(topology: str, n_qubits: int) -> list[tuple[int, int]]:
    if topology == "linear":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    elif topology == "ring":
        return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    elif topology == "all2all":
        return list(itertools.combinations(range(n_qubits), 2))
    raise ValueError(f"unknown topology: {topology}")


def ry_matrix(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def rz_matrix(theta):
    e_m = torch.exp(-1j * theta / 2)
    e_p = torch.exp(1j * theta / 2)
    z = torch.zeros((), dtype=torch.complex128)
    return torch.stack([torch.stack([e_m, z]), torch.stack([z, e_p])])


_H_GATE = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
_HSDAG = torch.tensor([[1, -1j], [1, 1j]], dtype=torch.complex128) / np.sqrt(2)


def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    axis = qubit + 1
    state = torch.movedim(state, axis, 1)
    shape = state.shape
    state = state.reshape(shape[0], 2, -1)
    state = torch.einsum("ij,bjk->bik", gate.to(state.dtype), state)
    state = state.reshape(shape)
    state = torch.movedim(state, 1, axis)
    return state


def apply_cnot(state, control, target, n_qubits):
    c_axis, t_axis = control + 1, target + 1
    state = torch.movedim(state, (c_axis, t_axis), (1, 2))
    s0 = state[:, 0, :].clone()
    s1 = state[:, 1, :].clone()
    new_s1 = torch.flip(s1, dims=[1])
    state = torch.stack([s0, new_s1], dim=1)
    state = torch.movedim(state, (1, 2), (c_axis, t_axis))
    return state


def simulate_final_state(x_batch, weights, n_qubits, n_layers, entangling_pairs):
    B = x_batch.shape[0]
    norm = x_batch.norm(dim=-1, keepdim=True) + 1e-12
    amp = (x_batch / norm).to(torch.complex128)
    state = amp.reshape((B,) + (2,) * n_qubits)
    wr = weights.reshape(n_layers, n_qubits, 2)
    for l in range(n_layers):
        for q in range(n_qubits):
            g_ry = ry_matrix(wr[l, q, 0].to(torch.float64)).to(torch.complex128)
            state = apply_single_qubit_gate(state, g_ry, q, n_qubits)
            g_rz = rz_matrix(wr[l, q, 1].to(torch.float64))
            state = apply_single_qubit_gate(state, g_rz, q, n_qubits)
        for (c, t) in entangling_pairs:
            state = apply_cnot(state, c, t, n_qubits)
    return state


def xyz_features_from_state(state, n_qubits):
    """[N, 3n] in order [X0..Xn, Y0..Yn, Z0..Zn], matching e2_nullspace_attack.py."""
    B = state.shape[0]
    flat = state.reshape(B, 2 ** n_qubits)
    probs = (flat.conj() * flat).real
    idx = torch.arange(2 ** n_qubits)
    bits = ((idx.unsqueeze(1) >> torch.arange(n_qubits - 1, -1, -1)) & 1)
    z_ev = (1.0 - 2.0 * bits.to(probs.dtype)).T  # (n_qubits, dim)
    z_vals = probs @ z_ev.T  # (B, n_qubits)

    x_vals_list, y_vals_list = [], []
    for q in range(n_qubits):
        sx = apply_single_qubit_gate(state, _H_GATE, q, n_qubits).reshape(B, 2 ** n_qubits)
        px = (sx.conj() * sx).real
        x_vals_list.append(px @ z_ev[q])
        sy = apply_single_qubit_gate(state, _HSDAG, q, n_qubits).reshape(B, 2 ** n_qubits)
        py = (sy.conj() * sy).real
        y_vals_list.append(py @ z_ev[q])
    x_vals = torch.stack(x_vals_list, dim=1)
    y_vals = torch.stack(y_vals_list, dim=1)
    return torch.cat([x_vals, y_vals, z_vals], dim=1).to(torch.float64), z_vals.to(torch.float64)


def target_logit_from_state(state, n_qubits):
    """Mean <Z_q> across qubits -- matches e2_nullspace_attack.py's _target_logit_raw."""
    _, z_vals = xyz_features_from_state(state, n_qubits)
    return z_vals.mean(dim=1)


# =========================================================================
# Classifier head (matches FixedClassifier architecture: 5->32->ReLU->2)
# =========================================================================

class ClassifierHead(nn.Module):
    """Linear(n->32) -> ReLU -> Dropout(0.2) -> Linear(32->2), matching the REAL checkpoint's
    architecture exactly (data_generation/scb_mnist_baseline.py's QuantumNeuralNetwork.classifier,
    confirmed by the classifier.0/classifier.3 weight-key indices seen in clean_model_seed_*_epoch_200.pt:
    index 0=Linear, 1=ReLU, 2=Dropout, 3=Linear)."""
    def __init__(self, n_qubits, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_qubits, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n_classes))

    def forward(self, z):
        return self.net(z.to(torch.float32))


# =========================================================================
# Mahalanobis / detection helpers (match e2_nullspace_attack.py / adaptive_qnn_fast_with_asr.py)
# =========================================================================

def fit_mahal(feat_clean: np.ndarray):
    lw = LedoitWolf().fit(feat_clean)
    prec = lw.get_precision()
    mu = lw.location_

    def score(feat: np.ndarray) -> np.ndarray:
        d = feat - mu
        return np.einsum("ni,ij,nj->n", d, prec, d)

    return score, mu, prec, float(lw.shrinkage_)


def detection_auc(feats_clean: np.ndarray, feats_poison: np.ndarray, n: int = 300, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    nc = min(len(feats_clean), n); np_ = min(len(feats_poison), n)
    ic = rng.choice(len(feats_clean), nc, replace=False)
    ip = rng.choice(len(feats_poison), np_, replace=False)
    fc, fp = feats_clean[ic], feats_poison[ip]
    nn_ = min(nc, np_)
    fc, fp = fc[:nn_], fp[:nn_]
    lw = LedoitWolf().fit(fc)
    diff = np.vstack([fc, fp]) - lw.location_
    scores = np.einsum("ni,ij,nj->n", diff, lw.get_precision(), diff)
    labels = np.array([0] * nn_ + [1] * nn_)
    return float(roc_auc_score(labels, scores))


def null_projector(J_M: np.ndarray, svd_thresh: float = E2_SVD_THRESH):
    _, S, Vt = np.linalg.svd(J_M, full_matrices=True)
    rank = int(np.sum(S > svd_thresh))
    V_null = Vt[rank:].T
    P_null = V_null @ V_null.T
    return P_null, rank, S


# =========================================================================
# Data loading (real, read-only)
# =========================================================================

def load_real_data(seed: int, layer: int, pair: str, pr: float):
    pr_s = f"{float(pr):.12g}"
    base = DATA_ROOT / f"seed_{seed}" / f"layer_{layer}_grid" / pair / f"pr_{pr_s}"
    meta = torch.load(base / f"poisoned_samples_meta_seed_{seed}_pr_{pr_s}.pt",
                       map_location="cpu", weights_only=False)
    x_ct = meta["target_clean_data"].double()
    x_cnt = meta["non_target_clean_data"].double()
    x_p = meta["poisoned_non_target_data"].double()
    return x_ct, x_cnt, x_p


# =========================================================================
# Self-tests
# =========================================================================

def self_test():
    print("Running self-test (topology mechanism sweep)...")

    # 1. Ring-topology torch circuit matches the REAL fast_circuit.py (numpy) output.
    sys.path.insert(0, str(NC_PROJECT_CODE))
    import importlib
    fast_circuit = importlib.import_module("fast_circuit")

    rng = np.random.RandomState(0)
    x_np = rng.randn(4, FEATURE_DIM).astype(np.float64)
    w_np = rng.randn(3 * N_QUBITS * 2).astype(np.float64) * 0.5

    real_out = fast_circuit.run_circuit_batch(x_np, w_np, n_layers=3, n_qubits=N_QUBITS)
    real_z = real_out[:, :N_QUBITS]

    x_t = torch.tensor(x_np, dtype=torch.float64)
    w_t = torch.tensor(w_np, dtype=torch.float64)
    ring_pairs = get_entangling_pairs("ring", N_QUBITS)
    state = simulate_final_state(x_t, w_t, N_QUBITS, 3, ring_pairs)
    _, my_z = xyz_features_from_state(state, N_QUBITS)
    my_z_np = my_z.detach().numpy()

    err = np.max(np.abs(real_z - my_z_np))
    assert err < 1e-6, f"Ring-topology circuit mismatch vs real fast_circuit.py: max_err={err}"
    print(f"  [check] torch Ring circuit vs real fast_circuit.py: max|diff|={err:.2e} (expect <1e-6)")

    # 2. Exact-autograd Jacobian matches finite-difference Jacobian.
    sample = torch.tensor(x_np[0:1], dtype=torch.float64, requires_grad=True)
    w_fixed = torch.tensor(w_np, dtype=torch.float64)
    state_s = simulate_final_state(sample, w_fixed, N_QUBITS, 3, ring_pairs)
    xyz_s, _ = xyz_features_from_state(state_s, N_QUBITS)
    xyz_s = xyz_s[0]
    J_auto = torch.zeros(3 * N_QUBITS, FEATURE_DIM, dtype=torch.float64)
    for k in range(3 * N_QUBITS):
        grad_k, = torch.autograd.grad(xyz_s[k], sample, retain_graph=True)
        J_auto[k] = grad_k[0]
    J_auto_np = J_auto.numpy()

    D = FEATURE_DIM
    eps = E2_EPS
    x0 = x_np[0]
    X_batch = np.tile(x0, (D + 1, 1))
    for i in range(D):
        X_batch[i + 1, i] += eps
    Xb_t = torch.tensor(X_batch, dtype=torch.float64)
    state_b = simulate_final_state(Xb_t, w_fixed, N_QUBITS, 3, ring_pairs)
    xyz_b, _ = xyz_features_from_state(state_b, N_QUBITS)
    xyz_b_np = xyz_b.detach().numpy()
    J_fd = (xyz_b_np[1:] - xyz_b_np[0]).T / eps

    rel_err = np.max(np.abs(J_auto_np - J_fd)) / (np.max(np.abs(J_fd)) + 1e-12)
    assert rel_err < 1e-3, f"autograd vs finite-diff Jacobian mismatch: rel_err={rel_err}"
    print(f"  [check] exact-autograd Jacobian vs finite-difference: max rel err={rel_err:.2e} (expect <1e-3)")

    # 3. Null-space projector: J_M @ P_null ~= 0.
    P_null, rank, S = null_projector(J_fd)
    residual = np.max(np.abs(J_fd @ P_null))
    j_scale = np.max(np.abs(J_fd)) + 1e-12
    assert residual / j_scale < 1e-6, f"null projector residual too large: {residual}"
    print(f"  [check] null-space projector: max|J_M @ P_null|/scale = {residual/j_scale:.2e} (expect ~0), rank={rank}/{D}")

    # 4. Mahalanobis/LedoitWolf sanity: distance at fitted mean ~0.
    feat_clean = rng.randn(50, 6)
    score_fn, mu, prec, shrink = fit_mahal(feat_clean)
    d_at_mean = score_fn(mu[None])[0]
    assert d_at_mean < 1e-8, f"Mahalanobis distance at own mean should be ~0, got {d_at_mean}"
    print(f"  [check] Mahalanobis distance at fitted mean = {d_at_mean:.2e} (expect ~0)")

    print("Self-test PASSED.\n")


# =========================================================================
# Phase 1: CLEAN-ONLY baseline pretraining
# =========================================================================
# Matches _train_clean_baseline() in data_generation/scb_mnist_baseline.py
# EXACTLY: 2-term CE loss on target_clean vs non_target_clean ONLY. No
# poisoned/triggered samples appear anywhere in this phase -- confirmed by
# reading the real function (it builds train_X/train_y purely from
# processed_splits['train'][target_class] and [non_target_class]).
#
# This was a real bug in an earlier version of this script: Phase 1
# originally included poisoned_non_target_data as a third, target-labeled
# training term (a conventional dirty-label backdoor). That is NOT what the
# actual paper does. In the real pipeline (adaptive_qnn_fast_with_asr.py,
# which this Phase 2 matches), the "backdoor" is created ENTIRELY by the
# Phase-2 weight-space attack (gradient descent on circuit weights only,
# minimizing the Mahalanobis distance of poisoned_non_target's XYZ features
# toward the clean reference) applied on top of a purely clean-trained
# model+classifier -- not by any data-poisoning training step.

def phase1_pretrain(x_ct, x_cnt, entangling_pairs, n_layers, n_qubits, topology_name, seed):
    torch.manual_seed(seed)
    weights = (torch.randn(n_layers * n_qubits * 2, dtype=torch.float64) * 0.1).requires_grad_(True)
    head = ClassifierHead(n_qubits)
    opt = torch.optim.Adam(list(head.parameters()) + [weights], lr=PHASE1_LR)

    x_tr = torch.cat([x_ct, x_cnt], dim=0)
    y_tr = torch.cat([torch.ones(len(x_ct)), torch.zeros(len(x_cnt))]).long()
    rng = np.random.default_rng(seed)

    history = []
    head.train()
    for epoch in range(PHASE1_EPOCHS):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), PHASE1_BATCH):
            bi = idx[i:i + PHASE1_BATCH]
            xb, yb = x_tr[bi], y_tr[bi]
            state = simulate_final_state(xb, weights, n_qubits, n_layers, entangling_pairs)
            _, z = xyz_features_from_state(state, n_qubits)
            loss = F.cross_entropy(head(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 50 == 0 or epoch == PHASE1_EPOCHS - 1:
            head.eval()
            with torch.no_grad():
                state_ct = simulate_final_state(x_ct, weights, n_qubits, n_layers, entangling_pairs)
                _, z_ct = xyz_features_from_state(state_ct, n_qubits)
                state_cnt = simulate_final_state(x_cnt, weights, n_qubits, n_layers, entangling_pairs)
                _, z_cnt = xyz_features_from_state(state_cnt, n_qubits)
                ca = 0.5 * ((head(z_ct).argmax(1) == 1).float().mean().item() +
                            (head(z_cnt).argmax(1) == 0).float().mean().item())
            history.append({"topology": topology_name, "seed": seed, "epoch": epoch + 1, "ca": ca})
            print(f"    [{topology_name}/seed={seed}] phase1 epoch {epoch+1:4d}: CA={ca:.4f} (clean-only training, no poison)")
            head.train()

    head.eval()
    return weights.detach().clone().requires_grad_(True), head, history


# =========================================================================
# Phase 2: adaptive Mahalanobis-minimizing attack
# =========================================================================

def phase2_adaptive_attack(weights, head, x_ct, x_p, x_cnt, entangling_pairs, n_layers, n_qubits, topology_name, seed):
    weights = weights.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([weights], lr=PHASE2_LR, momentum=0.0)

    with torch.no_grad():
        state_ref = simulate_final_state(x_ct, weights, n_qubits, n_layers, entangling_pairs)
        xyz_ref, _ = xyz_features_from_state(state_ref, n_qubits)
    xyz_ref_np = xyz_ref.numpy()
    lw_ref = LedoitWolf().fit(xyz_ref_np)
    mu_ref = torch.tensor(lw_ref.location_, dtype=torch.float64)
    prec_ref = torch.tensor(lw_ref.get_precision(), dtype=torch.float64)

    def evaluate(w):
        with torch.no_grad():
            state_p = simulate_final_state(x_p, w, n_qubits, n_layers, entangling_pairs)
            xyz_p, z_p = xyz_features_from_state(state_p, n_qubits)
            state_ct = simulate_final_state(x_ct, w, n_qubits, n_layers, entangling_pairs)
            xyz_ct, z_ct = xyz_features_from_state(state_ct, n_qubits)
            state_cnt = simulate_final_state(x_cnt, w, n_qubits, n_layers, entangling_pairs)
            _, z_cnt = xyz_features_from_state(state_cnt, n_qubits)

            asr = (head(z_p).argmax(1) == 1).float().mean().item()
            ca = 0.5 * ((head(z_ct).argmax(1) == 1).float().mean().item() +
                        (head(z_cnt).argmax(1) == 0).float().mean().item())
            auc_z = detection_auc(z_ct.numpy(), z_p.numpy())
            auc_xyz = detection_auc(xyz_ct.numpy(), xyz_p.numpy())
            diff = xyz_p - mu_ref
            train_loss = (diff @ prec_ref * diff).sum(dim=1).mean().item()
        return {"asr": asr, "ca": ca, "auc_z": auc_z, "auc_xyz": auc_xyz, "train_loss": train_loss}

    history = []
    det0 = evaluate(weights)
    det0.update({"topology": topology_name, "seed": seed, "step": 0})
    history.append(det0)
    print(f"    [{topology_name}/seed={seed}] phase2 step    0: Z={det0['auc_z']:.4f} XYZ={det0['auc_xyz']:.4f} "
          f"ASR={det0['asr']:.4f} CA={det0['ca']:.4f} loss={det0['train_loss']:.4f}")

    for step in range(1, PHASE2_STEPS + 1):
        state_p = simulate_final_state(x_p, weights, n_qubits, n_layers, entangling_pairs)
        xyz_p, _ = xyz_features_from_state(state_p, n_qubits)
        diff = xyz_p - mu_ref
        loss = (diff @ prec_ref * diff).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % PHASE2_EVAL_EVERY == 0:
            det = evaluate(weights)
            det.update({"topology": topology_name, "seed": seed, "step": step})
            history.append(det)
            print(f"    [{topology_name}/seed={seed}] phase2 step {step:4d}: Z={det['auc_z']:.4f} XYZ={det['auc_xyz']:.4f} "
                  f"ASR={det['asr']:.4f} CA={det['ca']:.4f} loss={det['train_loss']:.4f}")

    return weights.detach().clone(), history


# =========================================================================
# E2: null-space projection mechanism analysis (exact-autograd Jacobian)
# =========================================================================

def compute_jacobian_and_grad(sample_1d, weights, entangling_pairs, n_layers, n_qubits):
    """sample_1d: (D,) tensor, requires_grad. Returns J_M (3n, D) numpy, g_logit (D,) numpy."""
    x = sample_1d.clone().detach().requires_grad_(True)
    state = simulate_final_state(x.unsqueeze(0), weights, n_qubits, n_layers, entangling_pairs)
    xyz, z = xyz_features_from_state(state, n_qubits)
    xyz = xyz[0]
    logit = z[0].mean()

    D = len(x)
    J_M = torch.zeros(3 * n_qubits, D, dtype=torch.float64)
    for k in range(3 * n_qubits):
        grad_k, = torch.autograd.grad(xyz[k], x, retain_graph=True)
        J_M[k] = grad_k
    g_logit, = torch.autograd.grad(logit, x, retain_graph=False)
    return J_M.numpy(), g_logit.numpy()


def run_projected_attack(sample_np, weights, entangling_pairs, n_layers, n_qubits,
                          P_proj, n_steps, alpha):
    x = sample_np.copy()
    with torch.no_grad():
        xyz_0, _ = xyz_features_from_state(
            simulate_final_state(torch.tensor(sample_np[None], dtype=torch.float64), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
        xyz_0 = xyz_0[0].numpy()

    for _ in range(n_steps):
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        state = simulate_final_state(x_t.unsqueeze(0), weights, n_qubits, n_layers, entangling_pairs)
        _, z = xyz_features_from_state(state, n_qubits)
        logit = z[0].mean()
        g, = torch.autograd.grad(logit, x_t)
        g_np = g.numpy()

        v = P_proj @ g_np
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            break
        x = x + alpha * (v / v_norm)

    with torch.no_grad():
        xyz_f, z_f = xyz_features_from_state(
            simulate_final_state(torch.tensor(x[None], dtype=torch.float64), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
        logit_final = z_f[0].mean().item()
        xyz_final = xyz_f[0].numpy()
        _, z_0 = xyz_features_from_state(
            simulate_final_state(torch.tensor(sample_np[None], dtype=torch.float64), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
        logit_init = z_0[0].mean().item()

    return x, logit_final - logit_init, float(np.linalg.norm(xyz_final - xyz_0))


def e2_analysis(weights, x_ct, x_p, entangling_pairs, n_layers, n_qubits, topology_name, seed,
                 n_steps=E2_N_STEPS, step_frac=E2_STEP_FRAC):
    with torch.no_grad():
        state_ct = simulate_final_state(x_ct, weights, n_qubits, n_layers, entangling_pairs)
        xyz_ct, _ = xyz_features_from_state(state_ct, n_qubits)
    xyz_ct_np = xyz_ct.numpy()
    mahal_fn, mu, prec, shrink = fit_mahal(xyz_ct_np)
    thresh_95 = float(np.percentile(mahal_fn(xyz_ct_np), 95))

    x_p_np = x_p.numpy()
    D = x_p_np.shape[1]
    rng = np.random.default_rng(seed + 1000)

    rows = []
    for i, sample in enumerate(x_p_np):
        alpha = step_frac * float(np.linalg.norm(sample))
        sample_t = torch.tensor(sample, dtype=torch.float64)

        J_M, g_logit = compute_jacobian_and_grad(sample_t, weights, entangling_pairs, n_layers, n_qubits)
        P_null, rank, S = null_projector(J_M)
        null_dim = D - rank
        r_dim = null_dim / D
        g_norm = float(np.linalg.norm(g_logit))
        v_proj = P_null @ g_logit
        v_proj_norm = float(np.linalg.norm(v_proj))
        r_hidden = v_proj_norm / (g_norm + 1e-30)

        with torch.no_grad():
            xyz_base, _ = xyz_features_from_state(
                simulate_final_state(sample_t.unsqueeze(0), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
            xyz_base_np = xyz_base[0].numpy()
        mahal_0 = float(mahal_fn(xyz_base_np[None])[0])

        x_proj, dlogit_proj, drift_proj = run_projected_attack(
            sample, weights, entangling_pairs, n_layers, n_qubits, P_null, n_steps, alpha)
        with torch.no_grad():
            xyz_proj, _ = xyz_features_from_state(
                simulate_final_state(torch.tensor(x_proj[None], dtype=torch.float64), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
        mahal_proj = float(mahal_fn(xyz_proj[0].numpy()[None])[0])

        Q, _ = np.linalg.qr(rng.standard_normal((D, max(null_dim, 1))))
        P_rand = Q @ Q.T
        x_rand, dlogit_rand, drift_rand = run_projected_attack(
            sample, weights, entangling_pairs, n_layers, n_qubits, P_rand, n_steps, alpha)
        with torch.no_grad():
            xyz_rand, _ = xyz_features_from_state(
                simulate_final_state(torch.tensor(x_rand[None], dtype=torch.float64), weights, n_qubits, n_layers, entangling_pairs), n_qubits)
        mahal_rand = float(mahal_fn(xyz_rand[0].numpy()[None])[0])

        rows.append(dict(
            topology=topology_name, seed=seed, sample_idx=i, D=D, rank_JM=rank, null_dim=null_dim,
            r_dim=r_dim, r_hidden=r_hidden, g_norm=g_norm,
            mahal_before=mahal_0, detected_before=int(mahal_0 > thresh_95),
            logit_delta_proj=dlogit_proj, xyz_drift_proj=drift_proj,
            mahal_after_proj=mahal_proj, evades_proj=int(mahal_proj <= thresh_95),
            logit_delta_rand=dlogit_rand, xyz_drift_rand=drift_rand,
            mahal_after_rand=mahal_rand, evades_rand=int(mahal_rand <= thresh_95),
        ))

    return pd.DataFrame(rows)


# =========================================================================
# Main
# =========================================================================

def main(seed_filter=None):
    """seed_filter: if given (a list of ints), only run those seeds and write to
    seed-suffixed output files -- lets multiple parallel processes (one per seed,
    or one per small seed group) run safely without clobbering each other's CSVs.
    If None, runs all SEEDS and writes to the original unsuffixed filenames
    (single-process, backward-compatible mode)."""
    t_start = time.time()
    self_test()

    seeds_to_run = seed_filter if seed_filter is not None else SEEDS
    suffix = f"_seeds{'-'.join(str(s) for s in seeds_to_run)}" if seed_filter is not None else ""

    all_phase1_hist, all_phase2_hist = [], []
    e2_frames = []

    for seed in seeds_to_run:
        x_ct, x_cnt, x_p = load_real_data(seed, LAYER, PAIR, PR)
        print(f"\n\n{'#'*70}\n# seed = {seed}\n{'#'*70}")
        print(f"Loaded real data: target_clean={x_ct.shape}, non_target_clean={x_cnt.shape}, poisoned_non_target={x_p.shape}\n")

        for topology in TOPOLOGIES:
            pairs = get_entangling_pairs(topology, N_QUBITS)
            print(f"\n{'='*70}\nTopology = {topology}  ({len(pairs)} entangling edges: {pairs})  seed={seed}\n{'='*70}")

            t0 = time.time()
            weights, head, hist1 = phase1_pretrain(x_ct, x_cnt, pairs, LAYER, N_QUBITS, topology, seed)
            all_phase1_hist.extend(hist1)
            print(f"  Phase 1 done ({time.time()-t0:.1f}s)")

            t0 = time.time()
            weights_final, hist2 = phase2_adaptive_attack(weights, head, x_ct, x_p, x_cnt, pairs, LAYER, N_QUBITS, topology, seed)
            all_phase2_hist.extend(hist2)
            print(f"  Phase 2 done ({time.time()-t0:.1f}s)")

            t0 = time.time()
            e2_df = e2_analysis(weights_final, x_ct, x_p, pairs, LAYER, N_QUBITS, topology, seed)
            e2_frames.append(e2_df)
            print(f"  E2 analysis done ({time.time()-t0:.1f}s)")
            print(f"  [{topology}/seed={seed}] r_dim={e2_df['r_dim'].mean():.4f}  r_hidden={e2_df['r_hidden'].mean():.3e}  "
                  f"mahal_before={e2_df['mahal_before'].mean():.2f}  mahal_after_proj={e2_df['mahal_after_proj'].mean():.2f}  "
                  f"JSER_proj={e2_df['evades_proj'].mean():.3f}  JSER_rand={e2_df['evades_rand'].mean():.3f}")

            pd.DataFrame(all_phase1_hist).to_csv(OUT_DIR / f"topology_sweep_phase1_history{suffix}.csv", index=False)
            pd.DataFrame(all_phase2_hist).to_csv(OUT_DIR / f"topology_sweep_phase2_history{suffix}.csv", index=False)
            pd.concat(e2_frames, ignore_index=True).to_csv(OUT_DIR / f"topology_sweep_e2_raw{suffix}.csv", index=False)

    e2_all = pd.concat(e2_frames, ignore_index=True)
    per_seed = e2_all.groupby(["topology", "seed"]).agg(
        r_dim=("r_dim", "mean"), r_hidden=("r_hidden", "mean"),
        mahal_before=("mahal_before", "mean"), mahal_after_proj=("mahal_after_proj", "mean"),
        mahal_after_rand=("mahal_after_rand", "mean"),
        JSER_proj=("evades_proj", "mean"), JSER_rand=("evades_rand", "mean"),
    ).reset_index()
    per_seed.to_csv(OUT_DIR / f"topology_sweep_e2_per_seed{suffix}.csv", index=False)

    print(f"\n\n=== DONE seeds={seeds_to_run} (wall time {time.time()-t_start:.1f}s) ===")
    print(per_seed.to_string(index=False))
    print(f"\nWrote: topology_sweep_e2_per_seed{suffix}.csv (+ raw/phase1/phase2 history files with the same suffix)")
    if seed_filter is not None:
        print("This was a single-seed-group parallel worker. Run aggregate_topology_sweep_results() "
              "(or the --aggregate CLI mode) after ALL parallel workers finish to combine into the "
              "final cross-seed mean+/-std summary.")


def aggregate_results():
    """Combine all topology_sweep_e2_per_seed*.csv files (from parallel single-seed
    runs, or the single combined run) into the final cross-seed mean+/-std summary."""
    files = sorted(OUT_DIR.glob("topology_sweep_e2_per_seed*.csv"))
    if not files:
        print("No topology_sweep_e2_per_seed*.csv files found in", OUT_DIR)
        return
    print(f"Aggregating {len(files)} file(s): {[f.name for f in files]}")
    per_seed = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per_seed = per_seed.drop_duplicates(subset=["topology", "seed"])
    per_seed = per_seed.sort_values(["topology", "seed"])
    per_seed.to_csv(OUT_DIR / "topology_sweep_e2_per_seed_ALL.csv", index=False)

    summary = per_seed.groupby("topology").agg(
        n_edges=("topology", lambda s: len(get_entangling_pairs(s.iloc[0], N_QUBITS))),
        n_seeds=("r_hidden", "count"),
        r_dim_mean=("r_dim", "mean"), r_dim_std=("r_dim", "std"),
        r_hidden_mean=("r_hidden", "mean"), r_hidden_std=("r_hidden", "std"),
        mahal_before_mean=("mahal_before", "mean"), mahal_before_std=("mahal_before", "std"),
        JSER_proj_mean=("JSER_proj", "mean"), JSER_proj_std=("JSER_proj", "std"),
        JSER_rand_mean=("JSER_rand", "mean"), JSER_rand_std=("JSER_rand", "std"),
    ).reindex(TOPOLOGIES)
    summary.to_csv(OUT_DIR / "topology_sweep_e2_summary.csv")

    print(f"\n=== TOPOLOGY TREND SUMMARY, mean +/- std over {per_seed['seed'].nunique()} seeds ===")
    print(summary.to_string())
    print("\n=== per-seed breakdown (all seeds combined) ===")
    print(per_seed.to_string(index=False))
    print("\nPrediction under 'topology-independent' theory: r_hidden should stay ~machine-precision")
    print("(similar order of magnitude, e.g. 1e-10 to 1e-12) across linear/ring/all2all, WITH LOW")
    print("SEED-TO-SEED VARIANCE. If r_hidden grows systematically with connectivity (or drops), or")
    print("varies wildly across seeds, the topology-independence claim needs revision.")


if __name__ == "__main__":
    if "--self-test-only" in sys.argv:
        self_test()
    elif "--aggregate" in sys.argv:
        aggregate_results()
    elif "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed_args = [int(s) for s in sys.argv[idx + 1:] if s.lstrip("-").isdigit()]
        main(seed_filter=seed_args)
    else:
        main()
