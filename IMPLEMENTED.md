# IMPLEMENTED

## Branch History (origin/main..feat/symplectic-gating)
- [2025-12-20 18:48:48] `f0de542` `feat: Add Symplectic Gating mechanism for adaptive memory decay`
- [2025-12-22 14:34:08] `7736ff7` `Refactor Symplectic Gate: Use Tanh, Max Pooling, and Learnable Scale`
- [2025-12-22 14:34:27] `ed0d149` `Add Symplectic verification benchmarks`
- [2025-12-23 07:45:18] `465cdd8` `Fix vmap dimensions, implement Manifold Paging for Objective Reduction`
- [2025-12-30 17:49:47] `eae41a9` `feat: integrate symplectic reduction with moment map and per-sample paging`
- [2025-12-30 19:36:18] `2575ca4` `feat: add Dynamic Mode Decomposition (DMD) module and DMD-based Neural Memory Gating`
- [2026-02-08 15:15:45] `bf7839d` `Add gated symplectic options and synthetic benchmarks`
- [2026-02-08 15:35:44] `9b277c5` `Guard torch.compile and ignore uv lockfile`
- [2026-02-08 16:23:00] `bf58584` `Add soft/top-k gating and tracked progress docs`
- [2026-02-08 16:28:43] `792d65d` `Refresh IMPLEMENTED history after rebase`
- [2026-02-08 16:55:40] `dd8ef58` `Add phase-aware gating and benchmark regression tracking`
- [2026-02-08 17:17:26] `5cd9b22` `Advance manifold gating, ablations, and regression tracking`
- [2026-02-08 17:26:30] `8e6f4ac` `Add long-horizon recovery benchmark and next research topic`

## Uncommitted Work (Current Session)
- [2026-02-08 17:40:33] Added manifold-state keyed paging prototype in `NeuralMemory`.
- New constructor toggle: `manifold_state_keyed_paging=False` (opt-in, no default behavior change).
- When enabled and manifold state is available, page routing uses circular mean phase angle to select a page bucket.
- Added regression test: `tests/test_symplectic_reduction.py::test_manifold_state_keyed_paging_routes_by_angle_bucket`.

- [2026-02-08 17:40:33] Added gate-variant ablation benchmark script: `benchmarks/benchmark_gate_variants.py`.
- Variants: `symplectic_default`, `hard_diag`, `soft_diag`, `hard_topk8`, `adaptive_topk`, `phase_mix`.
- Outputs: `benchmarks/results/gate_variants_latest.json`, `benchmarks/results/gate_variants_history.csv`.
- Run tag `gate_variants_v2` (CUDA, steps=12):
- `symplectic_default`: spiral `0.027641`, helix `0.026637`.
- `hard_diag`: spiral `0.033863`, helix `0.026360`.
- `soft_diag`: spiral `0.031234`, helix `0.026194`.
- `hard_topk8`: spiral `0.025152`, helix `0.027509`, sparse_k `8.0`.
- `adaptive_topk`: spiral `0.031652`, helix `0.025994`, sparse_k `7.46875`.
- `phase_mix`: spiral `0.029299`, helix `0.026211`.

- [2026-02-08 17:40:33] Added threshold calibration benchmark script: `benchmarks/benchmark_threshold_sweep.py`.
- Sweeps `symplectic_page_threshold` and tracks loss plus switch-rate bands.
- Outputs: `benchmarks/results/threshold_sweep_latest.json`, `benchmarks/results/threshold_sweep_history.csv`.
- Run tag `threshold_sweep_v1` (CUDA, steps=12):
- `0.05`: switch_rate `4.000`, mean loss roughly `0.027`.
- `0.10`: switch_rate `4.000`, mean loss roughly `0.029`.
- `0.20`: switch_rate `0.667`, mean loss roughly `0.032`.
- `0.35`: switch_rate `0.000`, mean loss roughly `0.026`.
- `0.50`: switch_rate `0.000`, mean loss roughly `0.029`.
- `0.70`: switch_rate `0.000`, mean loss roughly `0.026`.

- [2026-02-08 17:55:14] Added regression guard utility: `benchmarks/check_regression.py`.
- Compares baseline vs latest benchmark JSON on selected timing/loss/recovery metrics with configurable tolerance gates.
- Exits non-zero on regression so it can be used in CI.
- Validation run:
- `python benchmarks/check_regression.py --baseline benchmarks/results/symplectic_latest.json --latest benchmarks/results/symplectic_latest.json` -> `passed`.

- [2026-02-08 18:57:55] Added cross-field narrative hypothesis for next experiment set in `TODO.md`.
- Source pool spans microbiology, genetics, chemistry, taste/odor detection, taxonomy science, information theory, and economic theory.
- Narrative target: quorum-budgeted combinatorial gating for robust page switching under noise.

- [2026-02-08 18:57:55] Added quorum-budgeted gating experiment controls in `SymplecticGating` (all opt-in).
- New kwargs: `quorum_mix`, `quorum_window`, `quorum_threshold`, `quorum_temperature`, `budget_topk_ratio`.
- Added `apply_quorum_policy(...)` helper and optional `return_quorum_map` forward output.
- Integrated quorum policy into final complexity score post-processing.

- [2026-02-08 18:57:55] Added tests for quorum policy behavior in `tests/test_symplectic.py`:
- `test_symplectic_gate_quorum_policy_prefers_sustained_signal`
- `test_symplectic_gate_quorum_budget_limits_positions`
- `test_symplectic_gate_quorum_budget_requires_quorum_mix`

- [2026-02-08 18:57:55] Extended gate ablation benchmark in `benchmarks/benchmark_gate_variants.py`:
- Added `quorum_budget` variant.
- Added quorum diagnostics (`quorum_mean`, `quorum_active_frac`) to gate summaries.
- Run tag `gate_variants_v3` (CUDA, steps=12):
- `quorum_budget`: spiral `0.033196`, helix `0.023264`, quorum `0.1178`.

- [2026-02-08 19:29:49] Implemented taxonomy-inspired hierarchical page routing in `NeuralMemory` (opt-in).
- New constructor kwargs: `hierarchical_paging`, `coarse_pages`, `fine_pages`, `hierarchy_mix`.
- Routing policy:
- coarse page from complexity bucket.
- fine page from manifold phase-angle bucket when available, else sequential fallback.
- blended target via `hierarchy_mix` in `[0, 1]`.
- Added validation checks: `coarse_pages * fine_pages == num_pages` and range guards.

- [2026-02-08 19:29:49] Added hierarchical routing test:
- `tests/test_symplectic_reduction.py::test_hierarchical_paging_routes_with_coarse_and_fine_keys`.

- [2026-02-08 19:29:49] Extended gate ablation benchmark with `hierarchical_route` variant.
- Run tag `gate_variants_v4` (CUDA, steps=12):
- `hierarchical_route`: spiral `0.029726`, helix `0.027642`, complexity `0.1531`.

- [2026-02-08 19:54:39] Implemented genetics-inspired mutation/selection search benchmark.
- Added `benchmarks/benchmark_mutation_selection.py`.
- Search space covers `phase_mix`, `quorum_mix`, `budget_topk_ratio`, and hierarchical routing toggles.
- Uses population selection with mutation and elite carry-over; no runtime behavior changes to core modules.
- Outputs:
- `benchmarks/results/mutation_selection_latest.json`
- `benchmarks/results/mutation_selection_history.csv`
- Run tag `mutation_selection_v1` (CUDA, steps=8, population=6, generations=3):
- Best fitness `0.061281`, mean loss `0.041680`, mean step `19.60 ms`.
- Best candidate:
- `phase_mix=0.5`, `quorum_mix=0.75`, `budget_topk_ratio=0.3`,
- `hierarchical=False`, `quorum_window=3`, `quorum_threshold=0.2`, `quorum_temperature=0.15`.

- [2026-02-08 20:38:49] Implemented chemistry-inspired kinetics coupling in `NeuralMemory` (opt-in).
- New constructor kwargs: `kinetics_coupling`, `kinetics_mix`, `kinetics_eps`.
- Coupling policy:
- derives chunk-level reaction progress from adaptive-lr/decay balance.
- reduces decay under high reaction progress.
- increases adaptive lr toward a bounded target under high reaction progress.
- Baseline behavior remains unchanged by default.

- [2026-02-08 20:38:49] Added kinetics tests:
- `tests/test_symplectic_reduction.py::test_kinetics_coupling_modulates_adaptive_lr`
- `tests/test_symplectic_reduction.py::test_kinetics_coupling_zero_mix_is_noop_for_adaptive_lr`

- [2026-02-08 20:38:49] Extended gate ablation benchmark with `kinetics_coupled` variant.
- Run tag `gate_variants_v5` (CUDA, steps=12):
- `kinetics_coupled`: spiral `0.028904`, helix `0.029320`, complexity `0.0463`, quorum `0.1632`.

- [2026-02-08 22:57:50] Implemented information/economic budget controller benchmark loop.
- Added `benchmarks/benchmark_switch_budget.py`.
- Adds benchmark-time penalties:
- switch-budget penalty for exceeding target switch rate.
- routing-entropy penalty toward target entropy (rational inattention proxy).
- No runtime model behavior changes; benchmark-only objective shaping.
- Outputs:
- `benchmarks/results/switch_budget_latest.json`
- `benchmarks/results/switch_budget_history.csv`
- Run tag `switch_budget_v1` (CUDA, steps=20, page-threshold=0.05):
- Spiral: unconstrained recon `0.015061`, constrained recon `0.011531`, entropy `0.086`.
- Helix: unconstrained recon `0.010996`, constrained recon `0.009805`, entropy `0.045`.

- [2026-02-09 00:54:38] Implemented taste/odor-inspired combinatorial codebook probe in `SymplecticGating` (opt-in).
- New gate kwargs: `codebook_mix`, `codebook_size`, `codebook_temperature`, `codebook_topk`.
- Policy:
- projects tokens to a codebook assignment simplex.
- computes transition intensity from total-variation distance between consecutive assignments.
- blends codebook transition score into complexity (`codebook_mix`).
- Added optional `return_codebook_map` output path for diagnostics.

- [2026-02-09 00:54:38] Added codebook tests in `tests/test_symplectic.py`:
- `test_symplectic_gate_codebook_map_shape`
- `test_symplectic_gate_codebook_prefers_mixture_shifts`

- [2026-02-09 00:54:38] Extended gate ablation benchmark with `combinatorial_codebook` variant.
- Run tag `gate_variants_v6` (CUDA, steps=12):
- `combinatorial_codebook`: spiral `0.029904`, helix `0.030032`, codebook `0.3595`, quorum `0.5643`.

- [2026-02-09 08:03:57] Added dedicated codebook calibration benchmark for multi-motif recall.
- Added `benchmarks/benchmark_codebook_sweep.py`.
- Sweeps `codebook_mix`, `codebook_size`, and `codebook_topk` using a multi-motif sequence with boundary-focused error.
- Outputs:
- `benchmarks/results/codebook_sweep_latest.json`
- `benchmarks/results/codebook_sweep_history.csv`
- Run tag `codebook_sweep_v1` (CUDA, steps=8):
- Best config: `codebook_mix=0.5`, `codebook_size=16`, `codebook_topk=None`.
- Best metrics: boundary MSE `0.009467`, total MSE `0.009267`, step `21.97 ms`, score `0.031376`.

- [2026-02-10 06:45:13] Added transfer benchmark for champion-vs-non-codebook comparisons on long-horizon and interference tasks.
- Added `benchmarks/benchmark_codebook_transfer.py`.
- Variants compared under identical settings:
- `baseline`
- `symplectic_paging`
- `phase_quorum_paging`
- `codebook_champion_paging` (`codebook_mix=0.5`, `codebook_size=16`, `codebook_topk=None`)
- Outputs:
- `benchmarks/results/codebook_transfer_latest.json`
- `benchmarks/results/codebook_transfer_history.csv`
- Run tag `codebook_transfer_v1` (CUDA):
- Winner: `codebook_champion_paging` with transfer score `0.100389`.
- Relative to `phase_quorum_paging`, champion improved:
- long-horizon clean post-MSE from `0.004147` to `0.003696` (`+10.87%`).
- long-horizon phase error from `0.058091` to `0.051676` (`+11.04%`).
- interference post-B task-A loss from `0.008036` to `0.007104` (`+11.60%`).
- aggregate transfer score from `0.105256` to `0.100389` (`+4.62%`).

- [2026-02-10 07:50:11] Extended regression guard to cover codebook transfer champion metrics.
- Added optional `--codebook-baseline` and `--codebook-latest` inputs to `benchmarks/check_regression.py`.
- Tracks `clean_mse_post`, `phase_err_post`, `interference.post_a_loss`, and `transfer_score` for `codebook_champion_paging`.

- [2026-02-10 07:52:39] Added manifold-keyed paging transfer benchmark against threshold paging.
- Added `benchmarks/benchmark_manifold_paging_transfer.py`.
- Outputs:
- `benchmarks/results/manifold_paging_latest.json`
- `benchmarks/results/manifold_paging_history.csv`
- Run tag `manifold_paging_transfer_v1` (CUDA):
- `threshold_paging`: long clean post `0.004963`, phase post `0.049742`, interference post-A `0.007729`, score `0.119738`.
- `manifold_keyed_paging`: long clean post `0.004750`, phase post `0.056093`, interference post-A `0.013046`, score `0.110337`.
- Summary: manifold-keyed paging lowered the aggregate transfer score but increased post-A interference loss, so it is a tradeoff rather than a clear win.

- [2026-02-10 07:53:25] Documented paging threshold switch-rate bands in `README.md` based on `benchmark_threshold_sweep.py`.
- High switch-rate: `0.05` to `0.10`.
- Moderate: `0.20`.
- Low or off: `0.35` and above.

- [2026-02-10 07:57:32] Wired regression guard into CI with rolling baseline artifacts.
- Added baseline files:
- `benchmarks/results/symplectic_baseline.json`
- `benchmarks/results/codebook_transfer_baseline.json`
- Updated `.github/workflows/test.yaml` to run lightweight benchmark variants and compare against baselines via `benchmarks/check_regression.py`.

- [2026-02-10 08:01:00] Added mutation champion transfer benchmark against hand-designed variants.
- Added `benchmarks/benchmark_mutation_transfer.py`.
- Outputs:
- `benchmarks/results/mutation_transfer_latest.json`
- `benchmarks/results/mutation_transfer_history.csv`
- Run tag `mutation_transfer_v1` (CUDA):
- `symplectic_paging` score `0.117808`.
- `phase_quorum_paging` score `0.108885`.
- `quorum_budget_paging` score `0.103567` (winner).
- `hierarchical_paging` score `0.137802`.
- `mutation_champion` score `0.108531`.
- Summary: mutation champion did not win; hand-designed quorum budget performed better on the transfer score.

- [2026-02-10 08:05:25] Added kinetics coupling sweep benchmark on long-horizon and interference tasks.
- Added `benchmarks/benchmark_kinetics_sweep.py`.
- Outputs:
- `benchmarks/results/kinetics_sweep_latest.json`
- `benchmarks/results/kinetics_sweep_history.csv`
- Run tag `kinetics_sweep_v1` (CUDA):
- Best `kinetics_mix=0.25` with transfer score `0.101662`.
- Baseline (`kinetics_mix=0.00`) score `0.116968`.

- [2026-02-10 08:07:44] Added switch-budget target sweep for nontrivial switch-rate reduction.
- Added `benchmarks/benchmark_switch_budget_sweep.py`.
- Outputs:
- `benchmarks/results/switch_budget_sweep_latest.json`
- `benchmarks/results/switch_budget_sweep_history.csv`
- Run tag `switch_budget_sweep_v1` (CUDA):
- Best `switch_target=0.10`, `entropy_target=0.60`, score `0.504275`.
- Observation: switch rates stayed ~`1.0` across targets in this sweep, so nontrivial reduction was not achieved.

- [2026-02-10 08:12:50] Expanded switch-budget sweep to vary page thresholds and quorum budget ratios.
- Added `page_thresholds` and `budget_ratios` sweeps in `benchmarks/benchmark_switch_budget_sweep.py` (supports `none`).
- Rerun tag `switch_budget_sweep_v2` showed reduced switch rates for higher thresholds and tighter budgets (best score at `thr=0.20`, `budget=0.15`, `switch=0.10`, `entropy=0.60`).

- [2026-02-10 08:12:50] Updated paging test to use stateful `active_page_indices` instead of deprecated `active_page_index`.

- [2026-02-10 08:34:39] Extended mutation-selection benchmark with opt-in transfer-aware fitness.
- Updated `benchmarks/benchmark_mutation_selection.py` with:
- `--use-transfer-fitness`, `--transfer-weight`
- long-horizon knobs (`--long-seq-len`, `--long-warmup-steps`, `--long-perturb-steps`, `--long-recovery-steps`)
- interference knobs (`--interference-steps`, `--interference-eval-iters`)
- Behavior: default remains classic spiral/helix fitness unless transfer fitness is explicitly enabled.
- Run tag `mutation_selection_transfer_v1` (CUDA, steps=8, population=6, generations=3, transfer-weight=0.7):
- generation-1 best fitness `0.076419` with candidate:
- `phase_mix=0.5`, `quorum_mix=0.75`, `budget_topk_ratio=0.3`, `hierarchical=False`, `quorum_window=5`, `quorum_threshold=0.25`, `quorum_temperature=0.15`.

- [2026-02-10 08:34:39] Reran mutation transfer benchmark cleanly after GPU contention.
- Run tag `mutation_transfer_v2` (CUDA):
- winner remained `quorum_budget_paging` with score `0.102306`.
- `mutation_champion` score `0.103167`.

- [2026-02-10 08:40:59] Added isolated `nanochat` transfer harness for 16GB GPU experiments.
- Added files:
- `experiments/nanochat_transfer/README.md`
- `experiments/nanochat_transfer/setup_nanochat.ps1`
- `experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1`
- `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1`
- Added ignore rule for local clone workspace: `external/nanochat/`.
- Added `README.md` section with setup/smoke/24h commands.
- Outcome: external pilot is now reproducible and isolated from core Titans behavior.

- [2026-02-10 08:59:56] Implemented first optional Titans-inspired `nanochat` candidate patch and wired it into protocol scripts.
- Added patch artifact:
- `experiments/nanochat_transfer/patches/nanochat_symplectic_candidate.patch`
- Added patch management scripts:
- `experiments/nanochat_transfer/apply_candidate_patch.ps1`
- `experiments/nanochat_transfer/revert_candidate_patch.ps1`
- Candidate patch scope (`nanochat` external repo):
- `nanochat/gpt.py`: optional token-complexity residual gate (`symplectic_gate_enabled`, `symplectic_gate_mix`, `symplectic_gate_eps`).
- `scripts/base_train.py`: new CLI flags `--symplectic-gate-enabled`, `--symplectic-gate-mix`, `--symplectic-gate-eps`.
- Protocol integration:
- `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1` can apply patch and runs `candidate_slot` with gate flags.
- `experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1` supports optional patch and candidate gate flags.
- Runtime fix:
- switched harness launches from `torchrun` to `python -m torch.distributed.run` for environments where `torchrun` is not on PATH.

- [2026-02-10 12:34:48] Stabilized nanochat runner reliability after contention and CLI/runtime failures.
- Updated `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1`:
- removed the stray `--` separator and unsupported `--seed` forwarding.
- normalized `-Seeds` input (supports comma-delimited values such as `"1337,2026"`).
- moved launch to direct `python -m scripts.base_train` and added fail-fast exit-code checks.
- defaulted to compile-disabled mode (`TORCHDYNAMO_DISABLE=1`, `TORCHINDUCTOR_DISABLE=1`) to avoid Triton dependency failures in Windows GPU runs.
- added automatic tokenizer/data prep when artifacts are missing (`AutoPrepareIfMissing=true`).
- switched default short-window attention pattern to `window_pattern=L` for non-FA3 GPUs.
- added structured run summaries:
- `experiments/nanochat_transfer/results/nanochat_protocol_latest.json`
- `experiments/nanochat_transfer/results/nanochat_protocol_history.csv`
- summary fields include per-run `val_bpb`, `min_val_bpb`, duration, and candidate-control deltas.

- [2026-02-10 12:34:48] Updated smoke/pilot support scripts.
- `experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1` now mirrors protocol safeguards (compile toggle, prep checks, candidate-flag validation, explicit `window_pattern`).
- `experiments/nanochat_transfer/apply_candidate_patch.ps1` and `experiments/nanochat_transfer/revert_candidate_patch.ps1` now resolve absolute `NanochatDir` paths before applying git operations.
- `experiments/nanochat_transfer/README.md` and root `README.md` updated with current harness defaults and summary artifact paths.

- [2026-02-10 12:34:48] Completed a short-window control-vs-candidate directional pass (single GPU, 2 seeds).
- Protocol command used `NumIterations=1`, `Seeds=1337,2026`, and candidate patch enabled.
- Mean candidate minus control bpb: `-0.003770` (lower is better), with candidate slot slightly slower in wall time.
- Outcome: directional signal is positive in micro-window screening; promotion to larger short-window remains pending.

- [2026-02-10 12:39:31] Added throughput-aware protocol deltas for promotion gating.
- Extended `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1` summaries with:
- per-run `trained_tokens` and `avg_tok_per_sec`.
- per-seed `candidate_minus_control_duration_sec` and `candidate_speed_ratio`.
- aggregate `mean_candidate_speed_ratio`.
- Latest 2-seed micro-window result:
- `mean_candidate_minus_control_bpb = -0.003770`.
- `mean_candidate_speed_ratio = 0.946965` (candidate about `5.3%` slower on average).

- [2026-02-10 13:13:36] Ran expanded short-window protocol at `NumIterations=128` (2 seeds, control vs candidate).
- Runtime profile:
- full 4-run protocol took about `32` minutes total on single RTX 5080 laptop GPU.
- control runs were about `442` to `447` sec; candidate runs were about `518` sec.
- Quality/speed result:
- `mean_candidate_minus_control_bpb = +0.076094` (candidate worse).
- `mean_candidate_speed_ratio = 0.857753` (candidate about `14.2%` slower).
- Interpretation:
- micro-window gains did not transfer to this larger short-window budget.
- candidate is currently not ready for 24h promotion without retuning.

- [2026-02-10 13:31:20] Added tunable candidate protocol knobs for faster retuning loops.
- Updated `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1` with:
- `CandidateGateMix`, `CandidateWeightDecay`, `CandidateMatrixLr`, and `RunLabel` parameters.
- Summary settings now record candidate hyperparameters and run label.
- `RunLabel` prefixes run tags so sweep variants do not collide in checkpoint directories.

- [2026-02-10 13:31:20] Ran first candidate-retune sweep (`mix005_n64`).
- Config: `NumIterations=64`, `CandidateGateMix=0.05`, `CandidateWeightDecay=0.2`, `CandidateMatrixLr=0.02`, 2 seeds.
- Result:
- `mean_candidate_minus_control_bpb = +0.003640` (candidate still worse, but much smaller gap than `+0.076094` at n128 baseline candidate settings).
- `mean_candidate_speed_ratio = 0.858812` (candidate still about `14.1%` slower).
- Interpretation:
- lower mix reduced quality regression magnitude, but candidate remains below control on both quality and speed.

- [2026-02-10 15:07:39] Completed true 90+ minute short-window evaluation (`mix005_n384`) and confirmed regression persists.
- Config: `NumIterations=384`, `CandidateGateMix=0.05`, `CandidateWeightDecay=0.2`, `CandidateMatrixLr=0.02`, 2 seeds.
- Runtime:
- full protocol took about `94.5` minutes.
- Quality/speed result:
- `mean_candidate_minus_control_bpb = +0.028736` (candidate worse).
- `mean_candidate_speed_ratio = 0.855154` (candidate about `14.5%` slower).
- Conclusion:
- candidate is not ready for 24h promotion.
- longer short-window evaluation validates that regression is not a micro-window artifact.

- [2026-02-10 16:01:16] Added structural odd-layer candidate path and patch-revision refresh for `nanochat` transfer harness.
- Updated `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1`:
- new toggle `-CandidateOddLayersOnly` (candidate-only).
- records `candidate_odd_layers_only` in protocol summary settings.
- Updated `experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1`:
- new toggle `-CandidateOddLayersOnly` for smoke parity with protocol runs.
- Updated `experiments/nanochat_transfer/apply_candidate_patch.ps1`:
- patch revision detection now checks odd-layer markers.
- supports `-ForceReapply` to reset target files and refresh the patch safely.

- [2026-02-10 16:01:16] Extended candidate patch artifact with odd-layer CLI wiring.
- Updated `experiments/nanochat_transfer/patches/nanochat_symplectic_candidate.patch`:
- adds `symplectic_gate_odd_layers_only` to `nanochat/gpt.py` config and block logic.
- adds `--symplectic-gate-odd-layers-only` to `scripts/base_train.py`.
- includes validation for `--symplectic-gate-mix` range in `scripts/base_train.py`.

- [2026-02-10 16:01:16] Ran structural odd-layer short-window sweeps.
- Micro check (`odd_mix005_n1`, 1 seed):
- `mean_candidate_minus_control_bpb = -0.003875`.
- `mean_candidate_speed_ratio = 0.985745`.
- Main retune (`odd_mix005_n64`, 2 seeds):
- `mean_candidate_minus_control_bpb = +0.002307`.
- `mean_candidate_speed_ratio = 0.918117`.
- Comparison vs full-depth `mix005_n64`:
- bpb delta improved from `+0.003640` to `+0.002307`.
- speed ratio improved from `0.858812` to `0.918117`.
- Follow-up (`odd_mix005_n128`, 2 seeds):
- `mean_candidate_minus_control_bpb = +0.038428`.
- `mean_candidate_speed_ratio = 0.911419`.
- Interpretation:
- odd-layer gating is a meaningful throughput improvement and narrows quality loss at `n64`.
- candidate still regresses on quality at `n128`, so not yet promotable to 24h.

- [2026-02-10 16:01:16] Reran mutation-transfer benchmark under isolated GPU load.
- Run tag `mutation_transfer_v4` (CUDA):
- winner: `mutation_champion` with transfer score `0.105628`.
- next: `quorum_budget_paging` at `0.108259`, then `phase_quorum_paging` at `0.109793`.
- Interpretation:
- winner ordering can flip relative to earlier contention-affected passes; repeatability checks are required before champion promotion.

- [2026-02-10 18:28:36] Added schedule-aware candidate gate controls for `nanochat` transfer harness and patch.
- Updated `experiments/nanochat_transfer/patches/nanochat_symplectic_candidate.patch`:
- `nanochat/gpt.py` now supports gate schedule fields:
- `symplectic_gate_start_iter`, `symplectic_gate_ramp_iters`.
- training forward path now computes an effective gate mix with delayed start and linear ramp.
- `scripts/base_train.py` now exposes:
- `--symplectic-gate-start-iter`, `--symplectic-gate-ramp-iters`.
- plus non-negative argument validation.
- Updated harness scripts:
- `run_nanochat_24h_protocol.ps1` and `run_nanochat_16gb_smoke.ps1` now expose `CandidateGateStartIter` and `CandidateGateRampIters`.
- protocol summaries now include `candidate_gate_start_iter` and `candidate_gate_ramp_iters`.
- Updated patch apply safety:
- `apply_candidate_patch.ps1` now checks schedule markers and still supports `-ForceReapply`.

- [2026-02-10 18:28:36] Validated scheduled odd-layer candidate (`mix=0.05`, `start=16`, `ramp=32`) across short and long windows.
- `odd_sched16r32_mix005_n64`:
- `mean_candidate_minus_control_bpb = -0.000145`.
- `mean_candidate_speed_ratio = 0.912058`.
- `odd_sched16r32_mix005_n128`:
- `mean_candidate_minus_control_bpb = -0.001091`.
- `mean_candidate_speed_ratio = 0.917453`.
- `odd_sched16r32_mix005_n384`:
- `mean_candidate_minus_control_bpb = -0.000097`.
- `mean_candidate_speed_ratio = 0.911722`.
- Interpretation:
- first candidate configuration to stay non-regressing at `n64`, `n128`, and `n384`.
- ready for full 24h promotion testing if ~`0.91` speed ratio is acceptable.

- [2026-02-10 18:28:36] Continued mutation-transfer stability checks (isolated runs).
- `mutation_transfer_v5`: winner `mutation_champion` (`0.108445`) over `quorum_budget_paging` (`0.109181`).
- `mutation_transfer_v6`: winner `quorum_budget_paging` (`0.104785`) over `mutation_champion` (`0.106363`).
- Interpretation:
- ordering remains unstable; continue repeated runs before final champion claims.

- [2026-02-10 19:12:48] Added one-command long-run orchestration for nanochat promotion runs.
- Added `experiments/nanochat_transfer/run_nanochat_full_cycle.ps1`.
- Flow:
- runs `run_nanochat_24h_protocol.ps1` with patch application and promoted candidate defaults.
- runs quick nanochat unit checks (`tests/test_attention_fallback.py` by default).
- runs `scripts.base_eval` (`bpb,sample`) for control and candidate run tags from the protocol summary.
- writes postcheck artifacts to `experiments/nanochat_transfer/results/nanochat_postcheck_latest.json` plus per-step logs.
- Rationale:
- keeps long-run training and post-training sanity checks serialized in one reproducible command.
- reduces operator error when switching between protocol, test, and eval commands.

- [2026-02-10 19:12:48] Reran mutation transfer benchmark after user-reported GPU contention.
- Run tag `mutation_transfer_v7` (CUDA):
- winner `quorum_budget_paging` with score `0.103228`.
- `mutation_champion` score `0.104346`.
- Interpretation:
- `quorum_budget_paging` regained the lead, but ranking remains close enough to keep stability checks active.

- [2026-02-10 19:21:04] Validated full-cycle nanochat orchestration with smoke runs and launched promotion run.
- Smoke validation:
- `run_nanochat_full_cycle.ps1` succeeded at `NumIterations=1` with `RunLabel=odd_sched16r32_mix005_smoke_fullcycle_fix3`.
- produced protocol summary and postcheck summary artifacts, including quick-test and eval logs.
- Promotion launch:
- started long run in background with `NumIterations=6000`, `Seeds=1337,2026`, `RunLabel=odd_sched16r32_mix005_n6000`.
- runtime logs:
- `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000.out.log`
- `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000.err.log`

- [2026-02-11 08:08:09] Added checkpoint-aware continuation for nanochat protocol/full-cycle runs.
- Updated `experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1`:
- new options: `-ContinueFromLatest`, `-SaveEvery`.
- behavior with `-ContinueFromLatest`:
- skips runs already at or beyond target `NumIterations`.
- resumes interrupted runs from latest available checkpoint step.
- records `run_status` and `resumed_from_step` in run summaries.
- Updated `experiments/nanochat_transfer/run_nanochat_full_cycle.ps1`:
- default `ContinueFromLatest=true`.
- default `SaveEvery=500` for finer-grained recovery points on long runs.
- passthrough of resume/save knobs to protocol script.

- [2026-02-11 08:08:09] Reboot recovery status and continuation launch for promoted `n6000` run.
- Recovered artifact:
- `control_s1337` completed to `step=6000` with `val_bpb=0.971755`.
- Missing artifacts after interruption:
- `candidate_slot_s1337`, `control_s2026`, `candidate_slot_s2026`.
- Relaunched continuation with same run label:
- skips completed `control_s1337` and continues remaining runs.
- active logs:
- `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000_resume.out.log`
- `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000_resume.err.log`

## Validation
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_symplectic.py` -> `14 passed`.
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_symplectic_reduction.py` -> `5 passed`.
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_symplectic_reduction.py` -> `5 passed`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_symplectic_reduction.py` -> `6 passed`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 22:57:50] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 22:57:50] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-09 08:03:57] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-09 08:03:57] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-10 07:50:11] `python benchmarks/check_regression.py --baseline benchmarks/results/symplectic_latest.json --latest benchmarks/results/symplectic_latest.json --codebook-baseline benchmarks/results/codebook_transfer_latest.json --codebook-latest benchmarks/results/codebook_transfer_latest.json` -> `passed`.
- [2026-02-10 06:45:13] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-10 06:45:13] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-10 08:12:50] `python -m pytest -q tests/test_paging.py` -> `1 passed`.
- [2026-02-10 08:34:39] `python benchmarks/benchmark_mutation_selection.py --tag mutation_selection_transfer_v1 --steps 8 --population 6 --generations 3 --elites 2 --use-transfer-fitness --transfer-weight 0.7 --output-json benchmarks/results/mutation_selection_latest.json --output-csv benchmarks/results/mutation_selection_history.csv` -> `completed`.
- [2026-02-10 08:34:39] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v2 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 08:40:59] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/setup_nanochat.ps1` -> `completed`.
- [2026-02-10 08:41:49] `python benchmarks/benchmark_mutation_selection.py --tag mutation_selection_smoke --steps 2 --population 2 --generations 1 --elites 1` -> `completed`.
- [2026-02-10 08:59:56] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/apply_candidate_patch.ps1` -> `completed`.
- [2026-02-10 08:59:56] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/revert_candidate_patch.ps1` -> `completed`.
- [2026-02-10 09:01:49] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/apply_candidate_patch.ps1` -> `completed`.
- [2026-02-10 09:01:49] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/revert_candidate_patch.ps1` -> `completed`.
- [2026-02-10 09:01:49] `python benchmarks/benchmark_mutation_selection.py --tag mutation_selection_transfer_smoke --steps 2 --population 2 --generations 1 --elites 1 --use-transfer-fitness --transfer-weight 0.7` -> `completed`.
- [2026-02-10 12:14:34] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v3 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 12:33:24] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 1 -Seeds "1337,2026" -ApplyCandidatePatch -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_history.csv` -> `completed`.
- [2026-02-10 12:10:34] `python -m pytest -q tests/test_paging.py::test_objective_reduction_page_switch tests/test_paging.py` -> `1 passed`.
- [2026-02-10 12:37:24] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1 -NumIterations 1 -RunTag nanochat_smoke_v2` -> `completed`.
- [2026-02-10 12:39:31] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 1 -Seeds "1337,2026" -ApplyCandidatePatch -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_history.csv` -> `completed`.
- [2026-02-10 13:13:10] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 128 -Seeds "1337,2026" -ApplyCandidatePatch -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_history.csv` -> `completed`.
- [2026-02-10 13:31:20] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 64 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_history.csv` -> `completed`.
- [2026-02-10 15:07:19] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 384 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel mix005_n384 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_mix005_n384_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_mix005_n384_history.csv` -> `completed`.
- [2026-02-10 15:13:01] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 1 -Seeds "1337" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -CandidateOddLayersOnly -RunLabel odd_mix005_n1 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n1_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n1_history.csv` -> `completed`.
- [2026-02-10 15:29:36] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 64 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -CandidateOddLayersOnly -RunLabel odd_mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n64_history.csv` -> `completed`.
- [2026-02-10 16:00:58] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 128 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -CandidateOddLayersOnly -RunLabel odd_mix005_n128 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n128_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n128_history.csv` -> `completed`.
- [2026-02-10 15:13:26] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v4 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 16:04:44] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/apply_candidate_patch.ps1 -NanochatDir external/nanochat -ForceReapply` -> `completed`.
- [2026-02-10 16:08:54] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 1 -Seeds "1337" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateOddLayersOnly -CandidateGateStartIter 1 -CandidateGateRampIters 2 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel odd_sched_n1 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched_n1_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched_n1_history.csv` -> `completed`.
- [2026-02-10 16:13:46] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v5 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 16:14:03] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v6 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 16:25:07] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 64 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateOddLayersOnly -CandidateGateStartIter 16 -CandidateGateRampIters 32 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel odd_sched16r32_mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n64_history.csv` -> `completed`.
- [2026-02-10 16:56:26] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 128 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateOddLayersOnly -CandidateGateStartIter 16 -CandidateGateRampIters 32 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel odd_sched16r32_mix005_n128 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n128_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n128_history.csv` -> `completed`.
- [2026-02-10 18:28:36] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -NumIterations 384 -Seeds "1337,2026" -ApplyCandidatePatch -CandidateGateMix 0.05 -CandidateOddLayersOnly -CandidateGateStartIter 16 -CandidateGateRampIters 32 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel odd_sched16r32_mix005_n384 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n384_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_n384_history.csv` -> `completed`.
- [2026-02-10 19:12:48] `python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer_v7 --output-json benchmarks/results/mutation_transfer_latest.json --output-csv benchmarks/results/mutation_transfer_history.csv` -> `completed`.
- [2026-02-10 19:12:48] `python -m pytest -q tests/test_attention_fallback.py` (in `external/nanochat`) -> `5 passed`, `10 skipped`.
- [2026-02-10 19:12:48] `python -m scripts.base_eval --model-tag nc_odd_sched16r32_mix005_n64_d12_control_s1337 --eval bpb,sample --device-batch-size 4 --split-tokens 65536` (in `external/nanochat`) -> `completed`, `val bpb=1.822249`.
- [2026-02-10 19:12:48] `python -m scripts.base_eval --model-tag nc_odd_sched16r32_mix005_n64_d12_candidate_slot_s1337 --eval bpb,sample --device-batch-size 4 --split-tokens 65536` (in `external/nanochat`) -> `completed`, `val bpb=1.822170`.
- [2026-02-10 19:21:04] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_full_cycle.ps1 -NumIterations 1 -Seeds "1337" -RunLabel odd_sched16r32_mix005_smoke_fullcycle_fix3 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_smoke_fullcycle_fix3_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_smoke_fullcycle_fix3_history.csv -PostcheckJson experiments/nanochat_transfer/results/nanochat_postcheck_smoke_fullcycle_fix3_latest.json` -> `completed`.
- [2026-02-10 19:21:04] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_full_cycle.ps1 -NumIterations 6000 -Seeds "1337,2026" -RunLabel odd_sched16r32_mix005_n6000 ...` -> `started in background`.
- [2026-02-11 08:08:09] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_full_cycle.ps1 -NumIterations 1 -Seeds "1337" -RunLabel odd_sched16r32_mix005_smoke_fullcycle_fix3 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_smoke_fullcycle_fix3_resumecheck_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_sched16r32_mix005_smoke_fullcycle_fix3_resumecheck_history.csv -PostcheckJson experiments/nanochat_transfer/results/nanochat_postcheck_smoke_fullcycle_fix3_resumecheck_latest.json` -> `completed` (skip-completed path validated).
- [2026-02-11 08:08:09] `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_full_cycle.ps1 -NumIterations 6000 -Seeds "1337,2026" -RunLabel odd_sched16r32_mix005_n6000 -SaveEvery 500 ...` -> `started in background` (continuation mode).

## Decisions
- [2026-02-08 17:40:33] Keep all new experimental behavior opt-in by constructor and kwargs toggles.
- [2026-02-08 17:40:33] Keep branch tracking explicit: `README.md` for usage, `TODO.md` for planned work, `IMPLEMENTED.md` for timestamped execution history and results.
