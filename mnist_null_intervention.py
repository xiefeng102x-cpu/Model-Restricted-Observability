"""
mnist_null_intervention.py
================================================================
MVP-3 from the original experiment protocol notes
(Sections 23, 47): on a REAL trained Z-readout QML classifier (candidate1's
exact architecture: n_qubits=5, layer=12, real MNIST t7_vs_t0 data), insert
a "ZZ-phase null unitary" U_phi = product_(i,j) exp(-i*phi_ij*Z_i Z_j) after
the trained circuit's output, before the Z-based classifier head reads it.
Since Z_iZ_j is diagonal in the computational basis for every pair, U_phi
is diagonal too -- it only multiplies each computational-basis amplitude
by a pure phase, leaving the PROBABILITY distribution (hence every Z_i
marginal, hence the classifier's logits/predictions) exactly unchanged,
while it generically changes the bipartite entanglement entropy (same
mechanism as MVP-1, now applied post-hoc to a REAL trained model's output
states instead of hand-picked reference states).

This does not require retraining a new model with the intervention baked
in -- it's a POST-HOC transformation applied to the already-trained
circuit's output states, confirming the effect is real for actual trained
representations, not just the idealized MVP-1 two-qubit toy state.

Reuses (imports, does not duplicate): candidate1_architecture_ladder_
mnist.py's load_real_data / phase1_pretrain_generic UNCHANGED (Phase 2's
attack is not needed for this test, only a trained Phase-1 classifier);
topology_mechanism_sweep.simulate_final_state for the raw state (Phase1
training already validated this circuit); entanglement_capacity_with_
shot_noise.von_neumann_entropy_bipartition (already validated) for entropy.

Self-tests: apply_zz_null_unitary is exactly unitary (elementwise pure
phase, |factor|=1); applied to a random state, the FULL computational-
basis probability distribution is unchanged (a stronger, more direct
check than just "Z expectations unchanged" -- if probabilities are
untouched, EVERY function of them, including all Z_i marginals, is
automatically untouched too); entropy DOES change for a generic
(non-fine-tuned) choice of phi on a random state (confirms the
intervention has real teeth, isn't accidentally a no-op).

Outputs (data/):
  - mnist_null_intervention_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import importlib
tms = importlib.import_module("topology_mechanism_sweep")
c1 = importlib.import_module("candidate1_architecture_ladder_mnist")
ecn = importlib.import_module("entanglement_capacity_with_shot_noise")

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LAYER = 12
PAIR = "t7_vs_t0"
PR = 0.1
N_QUBITS = 5
BIP = N_QUBITS // 2
RING_PAIRS = [(q, (q + 1) % N_QUBITS) for q in range(N_QUBITS)]


def apply_zz_null_unitary(psi_flat: torch.Tensor, n_qubits: int, phis, pairs) -> torch.Tensor:
    """psi_flat: (..., dim) batched or single flat state. Diagonal unitary
    (elementwise phase multiplication) built from exp(-i*phi_k*Z_i Z_j) for
    each (pair, phi) -- exact, no approximation, no matrix ever built."""
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    total_phase = np.zeros(dim)
    for (i, j), phi in zip(pairs, phis):
        bit_i = (idx >> (n_qubits - 1 - i)) & 1
        bit_j = (idx >> (n_qubits - 1 - j)) & 1
        eig = (1 - 2 * bit_i) * (1 - 2 * bit_j)
        total_phase = total_phase + phi * eig
    phase_factor = torch.tensor(np.exp(-1j * total_phase), dtype=torch.complex128)
    return psi_flat * phase_factor


def self_test():
    print("Running self-test (mnist_null_intervention)...")

    n_test = 4
    dim_test = 2 ** n_test
    pairs_test = [(0, 1), (1, 2), (2, 3), (3, 0)]
    phis_test = [0.4, -0.7, 1.1, 0.3]

    # 1. exactly unitary (elementwise pure phase).
    torch.manual_seed(0)
    psi = torch.randn(dim_test, dtype=torch.complex128)
    psi = psi / psi.norm()
    psi_after = apply_zz_null_unitary(psi, n_test, phis_test, pairs_test)
    assert abs(psi_after.norm().item() - 1.0) < 1e-12
    print(f"  [check 1] apply_zz_null_unitary preserves norm exactly: {psi_after.norm().item():.12f}")

    # 2. full computational-basis probability distribution unchanged (stronger
    #    than just checking Z_i marginals individually).
    probs_before = (psi.conj() * psi).real
    probs_after = (psi_after.conj() * psi_after).real
    max_diff = (probs_before - probs_after).abs().max().item()
    assert max_diff < 1e-12, f"probabilities changed: max_diff={max_diff}"
    print(f"  [check 2] full computational-basis probability distribution unchanged: max|diff|={max_diff:.2e}")

    # 3. entropy DOES change for a generic (non-fine-tuned) phi choice -- confirms
    #    the intervention has real teeth, isn't accidentally a no-op.
    s_before = ecn.von_neumann_entropy_bipartition(psi, n_test, n_test // 2).item()
    s_after = ecn.von_neumann_entropy_bipartition(psi_after, n_test, n_test // 2).item()
    assert abs(s_before - s_after) > 1e-4, f"entropy barely changed: {s_before} vs {s_after}"
    print(f"  [check 3] entropy changes under a generic null intervention: {s_before:.4f} -> {s_after:.4f}")

    print("Self-test PASSED.\n")


def main():
    self_test()

    x_ct, x_cnt, x_p = c1.load_real_data(SEED, LAYER, PAIR, PR)
    print(f"\nLoaded real MNIST data: target_clean={x_ct.shape}, non_target_clean={x_cnt.shape}, "
          f"poisoned={x_p.shape}")

    torch.manual_seed(SEED)
    weights = (torch.randn(LAYER * N_QUBITS * 2, dtype=torch.float64) * 0.1).requires_grad_(True)

    class QMLWrap(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weights = nn.Parameter(w)
            self.n_qubits = N_QUBITS
        def forward(self, x):
            state = tms.simulate_final_state(x, self.weights, N_QUBITS, LAYER, RING_PAIRS)
            xyz, _ = tms.xyz_features_from_state(state, N_QUBITS)
            return xyz

    model = QMLWrap(weights)
    print("\nTraining Phase 1 (clean-only, real MNIST) -- MVP-3 only needs a trained classifier, "
          "Phase 2's attack is not part of this test...")
    head, hist1 = c1.phase1_pretrain_generic(model, x_ct, x_cnt, SEED, "QML")
    print(f"Phase 1 done, final CA={hist1[-1]['ca']:.4f}")

    n_units = c1.get_zfeat_dim(model)
    x_test = torch.cat([x_ct[:100], x_cnt[:100]], dim=0)

    with torch.no_grad():
        state_before = tms.simulate_final_state(x_test, model.weights, N_QUBITS, LAYER, RING_PAIRS)
        flat_before = state_before.reshape(x_test.shape[0], 2 ** N_QUBITS)
        xyz_before, _ = tms.xyz_features_from_state(state_before, N_QUBITS)
        z_before = xyz_before[:, -n_units:]
        logits_before = head(z_before)

        rng = np.random.RandomState(0)
        phis = list(rng.uniform(-1.5, 1.5, size=len(RING_PAIRS)))
        flat_after = apply_zz_null_unitary(flat_before, N_QUBITS, phis, RING_PAIRS)

        probs_before = (flat_before.conj() * flat_before).real
        probs_after = (flat_after.conj() * flat_after).real
        prob_max_diff = (probs_before - probs_after).abs().max().item()

        z_after = torch.stack([
            (probs_after * torch.tensor(
                (1.0 - 2.0 * ((np.arange(2 ** N_QUBITS) >> (N_QUBITS - 1 - q)) & 1)), dtype=torch.float64
            )).sum(dim=1) for q in range(N_QUBITS)
        ], dim=1).to(torch.float32)
        logits_after = head(z_after)

        logit_max_diff = (logits_before - logits_after).abs().max().item()
        pred_before = logits_before.argmax(1)
        pred_after = logits_after.argmax(1)
        pred_match = (pred_before == pred_after).float().mean().item()

        ent_before = np.array([ecn.von_neumann_entropy_bipartition(flat_before[i], N_QUBITS, BIP).item()
                                for i in range(flat_before.shape[0])])
        ent_after = np.array([ecn.von_neumann_entropy_bipartition(flat_after[i], N_QUBITS, BIP).item()
                               for i in range(flat_after.shape[0])])
        ent_diff = np.abs(ent_after - ent_before)

    print("\n=== MVP-3 result ===")
    print(f"  phis used: {[round(p, 4) for p in phis]}")
    print(f"  max |P(before)-P(after)| (full prob distribution) = {prob_max_diff:.3e}   (pass: < 1e-10)")
    print(f"  max |logit(before)-logit(after)|                  = {logit_max_diff:.3e}   (pass: < 1e-6)")
    print(f"  prediction match rate                             = {pred_match:.4f}   (pass: == 1.0)")
    print(f"  entropy |after-before|: mean={ent_diff.mean():.4f}  max={ent_diff.max():.4f}   "
          f"(pass: mean > 0.05, i.e. entropy actually moves)")

    pass_task_invariant = prob_max_diff < 1e-10 and logit_max_diff < 1e-6 and pred_match == 1.0
    pass_entropy_moves = ent_diff.mean() > 0.05
    print(f"\n  Gate check: task output (probs/logits/predictions) exactly unchanged -> {pass_task_invariant}")
    print(f"  Gate check: entropy meaningfully changes -> {pass_entropy_moves}")
    print(f"\n  MVP-3 {'PASSED' if (pass_task_invariant and pass_entropy_moves) else 'FAILED'}")

    pd.DataFrame([{"seed": SEED, "prob_max_diff": prob_max_diff, "logit_max_diff": logit_max_diff,
                   "pred_match": pred_match, "ent_diff_mean": ent_diff.mean(), "ent_diff_max": ent_diff.max(),
                   "passed": pass_task_invariant and pass_entropy_moves}]
                 ).to_csv(OUT_DIR / "mnist_null_intervention_summary.csv", index=False)
    return pass_task_invariant and pass_entropy_moves


if __name__ == "__main__":
    if "--self-test-only" in sys.argv:
        self_test()
    else:
        main()
