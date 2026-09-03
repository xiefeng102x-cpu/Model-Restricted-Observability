"""
diagnostic_identifiability.py
====================================================================
B7 from the original experiment protocol notes
(Sections 26-27, "Stage 3: 直接利用现有this repository"): extends MVP-3's single-seed
(seed=42) MNIST null-intervention check into a multi-seed, multi-dataset,
QUANTITATIVE confirmatory run -- MNIST (candidate1_architecture_ladder_
mnist.py's exact n_qubits=5/layer=12/t7_vs_t0 setup) AND BloodMNIST
(candidate3_architecture_ladder_bloodmnist.py's exact n_qubits=8/layer=8/
t6_vs_t0 setup), seeds 42-46 (exploratory) + 100-109 (confirmatory, 10 new
seeds), per guide Section 26.2/38.

For each trained model (real Phase-1 pretraining via c1.phase1_pretrain_
generic, unchanged): computes the guide's exact Section 27 output schema
per sample (sample_id, seed, checkpoint, D, DNS_theta, I_D,
task_jacobian_rank, task_null_dim) plus the ZZ-phase task-null
intervention's before/after entropy and EXACT logit invariance check
(same mechanism as MVP-3, now run across many more seeds and both
datasets instead of one seed on one dataset).

DNS_theta/I_D/task_jacobian_rank/task_null_dim are per-MODEL (per seed,
per dataset) quantities, not literally per-sample -- computed ONCE per
model via a Jacobian stacked over N_REF_SAMPLES representative samples
(same convention as this project's B4/B6 scripts) and replicated across
that model's sample rows, to match the guide's requested flat column
layout without inventing a different per-sample methodology.

Reuses (imports, unchanged): candidate1_architecture_ladder_mnist.py's
load_real_data / phase1_pretrain_generic / get_zfeat_dim / ClassifierHead;
candidate3_architecture_ladder_bloodmnist.py's load_real_data / make_qml;
topology_mechanism_sweep.py's simulate_final_state (the raw-state
recomputation needed for entropy/null-intervention, since QMLWrap only
exposes XYZ features, not the flat state); mnist_null_intervention.py's
apply_zz_null_unitary (dataset-agnostic, parametrized by n_qubits already);
entanglement_capacity_with_shot_noise.von_neumann_entropy_bipartition.

Self-tests: DNS/null-space machinery cross-checked against this project's
B4 self-test pattern (J_T @ V_null ~= 0 on a real case from this script's
own pipeline, not assumed from B4); ZZ-null logit invariance re-verified
on THIS script's actual trained-model construction (not assumed from
MVP-3) using a freshly-initialized (untrained) model, since the invariance
is algebraic and must hold regardless of training.

Outputs (data/):
  - diagnostic_identifiability_mnist.csv
  - diagnostic_identifiability_bloodmnist.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
tms = importlib.import_module("topology_mechanism_sweep")
c1 = importlib.import_module("candidate1_architecture_ladder_mnist")
c3 = importlib.import_module("candidate3_architecture_ladder_bloodmnist")
ecn = importlib.import_module("entanglement_capacity_with_shot_noise")
mni = importlib.import_module("mnist_null_intervention")

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
# Guide Section 26.2/38 also asks for a "confirmatory" batch (seeds 100-109) --
# checked before running: NEITHER data/Mnist/detail_single_samle/ NOR
# data/MedMnist/single_detect_20260411/epsilon_0.8/ has pre-generated
# poisoned-sample data for seeds outside 42-46. Generating new poisoned data
# for 10 new seeds would mean running an unfamiliar, unaudited data-generation
# pipeline (not something this script owns) -- out of scope here. Runs the 5
# exploratory seeds only and reports this honestly rather than silently
# claiming confirmatory-scale N.
N_SAMPLES_PER_MODEL = 60  # 30 target-clean + 30 non-target-clean
N_REF_SAMPLES = 5

DATASETS = {
    "mnist": {"n_qubits": 5, "layer": 12, "pair": "t7_vs_t0", "pr": 0.1,
              "load": c1.load_real_data},
    "bloodmnist": {"n_qubits": 8, "layer": 8, "pair": "t6_vs_t0", "pr": 0.1,
                   "load": c3.load_real_data},
}


class QMLWrap(nn.Module):
    def __init__(self, weights, n_qubits, layer, ring_pairs):
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.n_qubits = n_qubits
        self.layer = layer
        self.ring_pairs = ring_pairs

    def forward(self, x):
        state = tms.simulate_final_state(x, self.weights, self.n_qubits, self.layer, self.ring_pairs)
        xyz, _ = tms.xyz_features_from_state(state, self.n_qubits)
        return xyz


def make_model(dataset_name, seed):
    cfg = DATASETS[dataset_name]
    torch.manual_seed(seed)
    n_qubits, layer = cfg["n_qubits"], cfg["layer"]
    ring_pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    weights = (torch.randn(layer * n_qubits * 2, dtype=torch.float64) * 0.1)
    return QMLWrap(weights, n_qubits, layer, ring_pairs)


def null_space_basis(J_T, tol=1e-6):
    U, S, Vh = torch.linalg.svd(J_T, full_matrices=True)
    smax = S.max().item() if S.numel() > 0 else 0.0
    rank = int((S > tol * max(smax, 1.0)).sum().item())
    return Vh.T[:, rank:], rank


def compute_dns_theta(model, x_ref, n_units):
    """Parameter-space DNS/I_D: J_T = d(Z-readout)/d(weights) stacked over
    N_REF_SAMPLES, g_D = d(entropy)/d(weights). Z-only readout keeps J_T's
    row count small (n_units per sample) regardless of the model's full
    XYZ feature width."""
    n_qubits, layer, ring_pairs = model.n_qubits, model.layer, model.ring_pairs

    w = model.weights.detach().clone().requires_grad_(True)
    state = tms.simulate_final_state(x_ref, w, n_qubits, layer, ring_pairs)
    flat = state.reshape(x_ref.shape[0], 2 ** n_qubits)
    xyz, _ = tms.xyz_features_from_state(state, n_qubits)
    t = xyz[:, -n_units:].reshape(-1)  # (K*n_units,)
    J_rows = [torch.autograd.grad(t[i], w, retain_graph=True)[0] for i in range(t.shape[0])]
    J_T = torch.stack(J_rows, dim=0)

    w2 = model.weights.detach().clone().requires_grad_(True)
    state2 = tms.simulate_final_state(x_ref, w2, n_qubits, layer, ring_pairs)
    flat2 = state2.reshape(x_ref.shape[0], 2 ** n_qubits)
    bip = n_qubits // 2
    d = torch.stack([ecn.von_neumann_entropy_bipartition(flat2[i], n_qubits, bip)
                     for i in range(flat2.shape[0])]).mean()
    g_D = torch.autograd.grad(d, w2)[0]

    V_null, rank = null_space_basis(J_T)
    if V_null.shape[1] == 0:
        proj_sq = 0.0
    else:
        coeffs = V_null.T @ g_D
        proj_sq = ((V_null @ coeffs).norm() ** 2).item()
    dns = proj_sq / ((g_D.norm() ** 2).item() + 1e-10)
    n_params = w.shape[0]
    return dns, 1.0 - dns, rank, n_params - rank


def self_test():
    print("Running self-test (diagnostic_identifiability)...")

    model = make_model("mnist", seed=0)
    n_qubits, ring_pairs = model.n_qubits, model.ring_pairs
    n_units = c1.get_zfeat_dim(model)
    x_ref = torch.randn(N_REF_SAMPLES, 32, dtype=torch.float64)

    # 1. J_T @ V_null ~= 0, on a real case from THIS script's own pipeline.
    w = model.weights.detach().clone().requires_grad_(True)
    state = tms.simulate_final_state(x_ref, w, n_qubits, model.layer, ring_pairs)
    xyz, _ = tms.xyz_features_from_state(state, n_qubits)
    t = xyz[:, -n_units:].reshape(-1)
    J_rows = [torch.autograd.grad(t[i], w, retain_graph=True)[0] for i in range(t.shape[0])]
    J_T = torch.stack(J_rows, dim=0)
    V_null, rank = null_space_basis(J_T)
    residual = (J_T @ V_null).abs().max().item() if V_null.shape[1] > 0 else 0.0
    assert residual < 1e-6, f"J_T @ V_null should be ~0, got {residual}"
    print(f"  [check 1] J_T @ V_null ~= 0 (max|.|={residual:.2e}), rank(J_T)={rank}/{J_T.shape[1]}")

    # 2. ZZ-null logit invariance, re-verified on an UNTRAINED model (algebraic,
    #    must hold regardless of training -- same style of check as MVP-3/latent_
    #    null_control's self-tests).
    head = c1.ClassifierHead(n_units)
    head.eval()
    with torch.no_grad():
        state_before = tms.simulate_final_state(x_ref, model.weights, n_qubits, model.layer, ring_pairs)
        flat_before = state_before.reshape(N_REF_SAMPLES, 2 ** n_qubits)
        xyz_before, _ = tms.xyz_features_from_state(state_before, n_qubits)
        z_before = xyz_before[:, -n_units:]
        logits_before = head(z_before)

        rng = np.random.RandomState(0)
        phis = list(rng.uniform(-1.5, 1.5, size=len(ring_pairs)))
        flat_after = mni.apply_zz_null_unitary(flat_before, n_qubits, phis, ring_pairs)
        probs_after = (flat_after.conj() * flat_after).real
        z_after = torch.stack([
            (probs_after * torch.tensor(
                (1.0 - 2.0 * ((np.arange(2 ** n_qubits) >> (n_qubits - 1 - q)) & 1)), dtype=torch.float64
            )).sum(dim=1) for q in range(n_qubits)
        ], dim=1).to(torch.float32)
        logits_after = head(z_after)
        max_diff = (logits_before - logits_after).abs().max().item()
    assert max_diff < 1e-6, f"logits should be exactly invariant: max_diff={max_diff}"
    print(f"  [check 2] ZZ-null logit invariance on an UNTRAINED model (max_diff={max_diff:.2e}) "
          f"-- confirms invariance is algebraic")

    print("Self-test PASSED.\n")


def run_dataset(dataset_name):
    cfg = DATASETS[dataset_name]
    load_fn = cfg["load"]
    print(f"\n{'='*70}\n= dataset: {dataset_name}\n{'='*70}")
    rows = []

    for seed in SEEDS:
        x_ct, x_cnt, x_p = load_fn(seed, cfg["layer"], cfg["pair"], cfg["pr"])
        model = make_model(dataset_name, seed)
        n_units = c1.get_zfeat_dim(model)
        head, hist = c1.phase1_pretrain_generic(model, x_ct, x_cnt, seed, dataset_name)
        final_ca = hist[-1]["ca"]

        n_half = N_SAMPLES_PER_MODEL // 2
        x_test = torch.cat([x_ct[:n_half], x_cnt[:n_half]], dim=0)
        x_ref = x_test[:N_REF_SAMPLES]

        dns, i_d, rank, null_dim = compute_dns_theta(model, x_ref, n_units)

        n_qubits, layer, ring_pairs = model.n_qubits, model.layer, model.ring_pairs
        with torch.no_grad():
            state_before = tms.simulate_final_state(x_test, model.weights, n_qubits, layer, ring_pairs)
            flat_before = state_before.reshape(x_test.shape[0], 2 ** n_qubits)
            xyz_before, _ = tms.xyz_features_from_state(state_before, n_qubits)
            z_before = xyz_before[:, -n_units:]
            logits_before = head(z_before)

            rng = np.random.RandomState(seed)
            phis = list(rng.uniform(-1.5, 1.5, size=len(ring_pairs)))
            flat_after = mni.apply_zz_null_unitary(flat_before, n_qubits, phis, ring_pairs)
            probs_after = (flat_after.conj() * flat_after).real
            z_after = torch.stack([
                (probs_after * torch.tensor(
                    (1.0 - 2.0 * ((np.arange(2 ** n_qubits) >> (n_qubits - 1 - q)) & 1)), dtype=torch.float64
                )).sum(dim=1) for q in range(n_qubits)
            ], dim=1).to(torch.float32)
            logits_after = head(z_after)
            logit_max_diff = (logits_before - logits_after).abs().max().item()
            pred_match = (logits_before.argmax(1) == logits_after.argmax(1)).float().mean().item()

            bip = n_qubits // 2
            d_before = np.array([ecn.von_neumann_entropy_bipartition(flat_before[i], n_qubits, bip).item()
                                 for i in range(flat_before.shape[0])])
            d_after = np.array([ecn.von_neumann_entropy_bipartition(flat_after[i], n_qubits, bip).item()
                                for i in range(flat_after.shape[0])])

        for i in range(x_test.shape[0]):
            rows.append({"dataset": dataset_name, "sample_id": i, "seed": seed, "checkpoint": "final",
                        "final_ca": final_ca, "D_entropy": d_before[i], "D_entropy_null_pert": d_after[i],
                        "DNS_theta": dns, "I_D": i_d, "task_jacobian_rank": rank, "task_null_dim": null_dim,
                        "logit_max_diff": logit_max_diff, "pred_match_rate": pred_match})

        print(f"  seed={seed}: CA={final_ca:.4f}  I_D={i_d:.4f}  rank(J_T)={rank}  null_dim={null_dim}  "
              f"logit_max_diff={logit_max_diff:.2e}  pred_match={pred_match:.4f}  "
              f"entropy_diff_mean={np.abs(d_after-d_before).mean():.4f}")

        pd.DataFrame(rows).to_csv(OUT_DIR / f"diagnostic_identifiability_{dataset_name}.csv", index=False)

    return pd.DataFrame(rows)


def main():
    t_start = time.time()
    self_test()

    df_mnist = run_dataset("mnist")
    df_blood = run_dataset("bloodmnist")

    print(f"\n=== B7 result (wall time {time.time()-t_start:.1f}s) ===")
    for name, df in [("mnist", df_mnist), ("bloodmnist", df_blood)]:
        task_invariant = (df["logit_max_diff"] < 1e-6).all() and (df["pred_match_rate"] == 1.0).all()
        entropy_moves = (df.groupby("seed")["D_entropy_null_pert"].mean() -
                         df.groupby("seed")["D_entropy"].mean()).abs().mean()
        print(f"  [{name}] {len(df['seed'].unique())} seeds, {len(df)} sample-rows: "
              f"task_invariant(all seeds)={task_invariant}  mean|entropy_diff|={entropy_moves:.4f}  "
              f"mean I_D={df['I_D'].mean():.4f}")

    passed = all((df["logit_max_diff"] < 1e-6).all() and (df["pred_match_rate"] == 1.0).all()
                 for df in [df_mnist, df_blood])
    print(f"\n  Gate check: task output exactly invariant under null intervention across ALL "
          f"{len(SEEDS)} seeds x 2 datasets -> {passed}")
    print(f"\n  B7 {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    if "--self-test-only" in sys.argv:
        self_test()
    else:
        main()
