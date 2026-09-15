# Code for "Model-Restricted Observability and Measurement Completion in Quantum Learning Systems"

**Authors:** xiefeng102

---

## Overview

This repository contains the code needed to reproduce the experiments and
figures in the paper. The paper studies **model-restricted operational
observability**: a trained parameterized quantum circuit exposes only a
restricted measurement interface, and post-training tampering that exactly
preserves the task-facing output distribution can still be invisible to a
diagnostic unless the diagnostic's blind gradient falls inside the model's
own locally reachable tangent space. The repository implements:

- the **R_manifold** structural-restriction diagnostic (Theorem 1, Section 3),
- an **exact normalized observable-completion rule** (Proposition 2, Section 4)
  for constructing a measurement set that recovers as much of the blind
  gradient as the model's own geometry allows,
- a **task-invariant deployment perturbation** and a **pilot-based adaptive
  recalibration** scheme that restores completion performance once enough
  perturbation has accumulated (Sections 5-6),
- a **local stability bound** for how the restricted blind projector drifts
  under deployment-time tangent-geometry perturbation (Lemma 1 / Theorem 3,
  Section 6.3), with an estimator-error corollary connecting it to the
  pilot-recalibration error decomposition,
- the full experimental suite across BloodMNIST, MNIST, and an
  architecturally distinct VQE system, including adaptive white-box
  attackers, alternative diagnostics, noise-channel robustness checks, and a
  non-diagonal stress test (Section 8).

### What this code release does and does not include

The paper's headline classifier results use a **10-seed cohort per dataset**
(seeds 42-51): 5 "legacy" seeds (42-46), the original verification cohort,
and 5 "fresh" seeds (47-51), added later. **This code release bundles the
trained `theta` checkpoint for all 10 seeds per dataset** at
`cohort_fresh/checkpoints/`, so `manifold/theta_jacobian.py`'s fast
checkpoint-loading path is exercised for every seed instead of retraining;
each loaded checkpoint is cross-checked against the persisted `.npz` state
to machine precision (see "Data provenance" below). Retraining from scratch
remains available as a fallback code path (used automatically for any
seed whose checkpoint is missing), but is not what happens by default given
the bundled checkpoints. Every script's own seed list already reflects the
paper's true 10-seed design, and running them as-is now succeeds for all
ten seeds 42-51: every number reported in the paper is reproducible
end-to-end from this repository alone.

`theta`/held-out-accuracy persistence, the unified end-to-end detection
pipeline, and the Theorem 3 condition audit are all implemented in the code
exactly as the paper describes them (`manifold/theta_jacobian.py`'s fast
checkpoint-loading path, `run_experiment59_trace_distance_baseline.py`'s
paired-score detector, etc.), and this release also bundles the result
artifacts those produce on the full 10-seed cohort
(`theorem3_condition_audit.csv`, `unified_detection_auc_summary.csv`, the
fresh-cohort checkpoints), so they can be inspected directly without
re-running the audit scripts.

---

## Repository structure

```
manifold/
  restricted_observability.py         R_manifold / gamma_D,C computation (Theorem 1, Section 3)
  theta_jacobian.py                   Deterministic theta re-derivation + d(rho_A)/d(theta)
  mvp133_vqe_checkpoints.py           VQE checkpoint loader + R_manifold on the VQE circuit
  independent_real_state_verification.py   Independent finite-difference cross-check (SI A/C)

candidates/pauli_pool.py              Candidate off-diagonal Pauli-observable pool construction
oracle/entropy_target.py              Ambient blind-gradient amplitude gamma_D (entropy diagnostic)
selector/reduced_state_plugin.py      Route-A observable-selection plug-in interface
states/qml_reduced_states.py          Loader for the persisted classifier reduced states
measurement_simulator/pauli_pilot.py  Per-operator Pauli pilot-measurement simulator
measurement_simulator_grouped.py      Grouped-Pauli pilot-measurement baseline
measurement_simulator_shadow.py       Random-Pauli classical-shadow pilot-measurement estimator
seeds.py                              Deterministic per-role seed derivation

frozen_route_causal_test.py           VQE circuit simulator (statevector, gate mechanics)
topology_mechanism_sweep.py           Classifier circuit simulator + training loop
candidate1_architecture_ladder_mnist.py       MNIST classifier architecture + training recipe
candidate3_architecture_ladder_bloodmnist.py  BloodMNIST classifier architecture + training recipe
classical_architectures.py            Classical-control architecture baselines
entanglement_capacity_with_shot_noise.py      Entanglement-capacity diagnostic (VQE-line utility)
mnist_null_intervention.py            ZZ-null-unitary "after"-state construction
diagnostic_identifiability.py         Shared training/model-loading utilities

run_experiment23_manifold_restricted_observability.py   R_manifold across all classifier states
run_experiment24_mvp133_vqe_R_manifold.py                R_manifold on the VQE circuit (Section 8.1)
run_experiment25_manifold_aware_completion.py            Manifold-aware vs. naive completion (Section 4)
run_experiment26_j_star_C_stability.py                   Stability of the manifold-aware pick
run_experiment27_multi_sample_robust_completion.py       Multi-sample robust completion
run_experiment28_brute_force_oracle_completion.py        Brute-force oracle completion upper bound
run_experiment29_greedy_multi_observable_budget.py       Greedy multi-observable budget sweep
run_experiment30_smaller_subsystem_n_a3.py               Smaller-subsystem ablation
run_experiment31_within_vs_across_class_structure.py     Within- vs. across-class blind-direction structure
run_experiment32_before_after_completion_evasion.py      Task-invariant perturbation vs. static completion (Section 5)
run_experiment33_adaptive_pilot_recalibration.py         Pilot-based adaptive recalibration (Section 6)
run_experiment34_mnist_consistency_check.py              MNIST cross-check of the evasion/recalibration pattern
run_experiment35_multiple_perturbation_mechanisms.py     Multiple diagonal perturbation mechanisms
run_experiment36_magnitude_sweep.py                      Perturbation-magnitude sweep (Section 8.5)
run_experiment37_gated_adaptive_audit.py                 Gated static/adaptive audit policy (Section 10.1)
run_experiment38_end_to_end_detection.py                 Original end-to-end clean-vs-tampered detector (superseded by experiment59's unified pipeline for headline numbers)
run_experiment39_ablation_geometry_mismatch.py           Geometry-mismatch-only ablation
run_experiment40_ablation_candidate_pool_restriction.py  Candidate-pool-restriction ablation
run_experiment41_ceiling_predictability_check.py         Predictability of the geometry-mismatch ceiling
run_experiment42_realistic_best_k_adaptive.py            Realistic best-k adaptive completion
run_experiment43_ablation_c_subsystem_size.py            Subsystem-size trade-off (Section 8.6)
run_experiment44_sublinear_shadow_pilot.py               Classical-shadow vs. direct-tomography pilot (Section 8.7)
run_experiment45_white_box_adaptive_attacker.py          Ring-topology white-box adaptive attacker (Section 8.8)
run_experiment45b_white_box_attacker_constrained.py      Subsystem-touching-constrained attacker variant
run_experiment46_alternative_diagnostic_linear_entropy.py  Linear-entropy alternative diagnostic (Section 8.9)
run_experiment47_noise_robustness_depolarizing.py        Depolarizing-noise robustness (Section 8.9)
run_experiment48_near_threshold_noise_robustness.py      Near-threshold depolarizing sweep
run_experiment49_fully_general_attacker.py               All-to-all 36-parameter attacker family (Section 8.8)
run_experiment50_more_diagnostics.py                     Renyi-2 and further alternative diagnostics
run_experiment51_nondiagonal_mechanism.py                Non-diagonal stress test (Section 8.10)
run_experiment52_normalized_completion_score.py          Exact normalized vs. numerator-only selection rule (Section 4)
run_experiment53_grouped_pauli_baseline.py               Grouped-Pauli measurement baseline
run_experiment54_naive_diagnostic_baseline.py            Naive per-operator tomography baseline
run_experiment55_white_box_attacker_vs_pilot_detector.py Ring attacker vs. deployed pilot detector
run_experiment56_constrained_attacker_vs_pilot_detector.py  Constrained attacker vs. deployed pilot detector
run_experiment57_noise_robustness_amplitude_damping.py   Amplitude-damping-noise robustness (Section 8.9)
run_experiment58_near_threshold_amplitude_damping.py     Near-threshold amplitude-damping sweep
run_experiment59_trace_distance_baseline.py              Unified detector pipeline: manifold-restricted, raw off-diagonal, and trace-norm scores from one frozen paired pilot-draw stream (Section 8.5's headline AUC numbers)
run_experiment60_vqe_adaptive_recalibration.py           Adaptive recalibration on the VQE circuit
run_experiment61_benign_crosstalk_disambiguation.py      Benign static-ZZ crosstalk disambiguation (Section 8.9)

data/
  saved_states/            Raw persisted rho_A before/after per (dataset, seed) -- seeds 42-51
  manifold_artifacts/      Persisted R_manifold/Jacobian artifacts per (dataset, seed) -- seeds 42-51
  Mnist/detail_single_samle/seed_{42..51}/layer_12_grid/t7_vs_t0/pr_0.1/
                           Per-seed MNIST feature tensors theta re-derivation trains on
  MedMnist/single_detect_20260411/epsilon_0.8/seed_{42..51}/layer_8_grid/t6_vs_t0/pr_0.1/
                           Per-seed BloodMNIST feature tensors theta re-derivation trains on
  qmlreal_oracle_truth.csv Ambient blind-gradient ground truth per state (seeds 42-51)
  *.csv                    Pre-computed per-experiment result tables (seeds 42-51)

cohort_fresh/checkpoints/   Trained theta checkpoints for all 10 classifier seeds (42-51,
                             both datasets); `manifold/theta_jacobian.py` loads these directly
                             instead of retraining. Deterministic retraining from the recorded
                             seed remains available as a fallback for any seed without one.

checkpoints/                20 VQE checkpoints (theta_early / theta_late, 10 seeds -- VQE was
                             always a 10-seed design)
```

---

## Reproduction

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### Option A -- Verify from pre-computed data (fastest)

The `data/`, `cohort_fresh/checkpoints/`, and `checkpoints/` directories
already contain everything needed to re-derive and cross-check the paper's
full seed-42-51 numbers without retraining any classifier from scratch:

```bash
python run_experiment23_manifold_restricted_observability.py --smoke
```

This reconstructs theta for one seed, verifies the reconstructed final
classification accuracy and reduced state match the persisted `.npz` to
machine precision, and reports `R_manifold`. Verified end-to-end: this
completes with `ca_match=True`, `rho_match_err` at machine precision
(~1e-17), and the `gamma_D` oracle cross-check matching to <1e-6. Drop
`--smoke` to run every classifier state across all ten seeds -- each seed
loads its bundled checkpoint from `cohort_fresh/checkpoints/` directly
(fast, no retraining; ~0.02s per seed), verified to reproduce the persisted
`.npz` states to machine precision the same way for all ten seeds, both
datasets.

Theta loading (or, for a seed with no bundled checkpoint, retraining) reads
the per-seed feature tensors already bundled at
`data/Mnist/detail_single_samle/` (MNIST) and
`data/MedMnist/single_detect_20260411/epsilon_0.8/` (BloodMNIST) -- no
manual data placement is needed. (These are not the raw MNIST/BloodMNIST
images themselves, but the PCA-reduced/amplitude-encoded per-seed feature
tensors the training recipe consumes directly.)

The VQE-checkpoint path is fast (no retraining, checkpoints are loaded
directly) and covers all 10 VQE seeds:

```bash
python run_experiment24_mvp133_vqe_R_manifold.py
```

### Option B -- Full experimental suite

Each `run_experimentNN_*.py` script is self-contained and can be run
directly from the repository root:

```bash
python run_experiment25_manifold_aware_completion.py
python run_experiment59_trace_distance_baseline.py
python run_experiment45_white_box_adaptive_attacker.py
# ... etc.
```

All of these iterate the full seed list in the paper's current design
(42-51 for the two classifier datasets), and all ten seeds now run
end-to-end against the bundled data (each loading its checkpoint from
`cohort_fresh/checkpoints/`). Scripts write their output to a `results/`
directory created next to the script; `data/*.csv` holds the
already-computed seed-42-51 versions of the same tables used in the paper,
for direct comparison.

---

## Data provenance

- `data/saved_states/{dataset}_seed{seed}.npz` (seeds 42-51) -- the real,
  trained `rho_A` before and after the ZZ-null-unitary intervention, for
  each of these 10 MNIST + 10 BloodMNIST trained classifiers, plus each
  state's final classification accuracy and von Neumann entropy. `theta`
  itself is not persisted in this .npz (see `manifold/theta_jacobian.py`);
  instead, the trained `theta` checkpoint for every one of the ten seeds is
  bundled at `cohort_fresh/checkpoints/` and loaded directly (no
  retraining). Every loaded checkpoint is cross-checked against this
  persisted `.npz` state's final accuracy and reduced state before being
  trusted, and reproduces it to machine precision for all ten seeds, both
  datasets (see "Reproduction" above). Deterministic retraining from the
  recorded seed via `torch.manual_seed` and the exact training recipe in
  `candidate1_architecture_ladder_mnist.py` /
  `candidate3_architecture_ladder_bloodmnist.py` remains available as a
  fallback code path for any seed without a bundled checkpoint. The paper's
  own submission additionally persists a genuine held-out test accuracy for
  all ten seeds per dataset, computed from a real, never-before-used test
  split; that held-out-eval harness (as opposed to the checkpoints/artifacts
  themselves, which are bundled) is not part of this minimal code release.
- `data/Mnist/detail_single_samle/seed_{42..51}/...` and
  `data/MedMnist/single_detect_20260411/epsilon_0.8/seed_{42..51}/...` --
  for seeds 42-46 these are the original per-seed feature tensors; for
  seeds 47-51 they are generated by the same `clean_pool_for_new_seed`
  routine `load_real_data()`'s own fallback path already uses, so the
  bundled files are bit-identical to what that fallback would produce live
  -- this just persists the result once instead of regenerating it on every
  load.
- `checkpoints/mvp133_frozen_rep_{cluster,midtrain}_n6_seed{0..9}.pt` --
  genuine (epoch-40, epoch-400) checkpoint pairs from the same training run
  for a 6-qubit, 6-layer VQE circuit, used as an architecturally distinct
  corroboration system (over-parametrized classifiers make
  `R_manifold=1` a near-foregone conclusion by dimension-counting alone;
  this circuit's 72 parameters do not).
- `data/manifold_artifacts/` and the top-level `data/*.csv` files are this
  repository's own persisted analysis output (R_manifold values, Jacobians,
  completion scores, detector AUCs, etc.) for seeds 42-51, included so every
  reported number in the paper can be checked without re-running the full
  experimental suite.

---

## Contact

xiefeng102 -- xiefeng102x@gmail.com
