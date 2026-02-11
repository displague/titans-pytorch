# TODO

## Active
- [2026-02-10 19:12:48] Execute the promoted nanochat run through the new full-cycle harness.
- Hypothesis: a one-command protocol + postcheck flow reduces operator mistakes and gives directly comparable quality/sanity artifacts for long runs.
- Toggle plan: run `experiments/nanochat_transfer/run_nanochat_full_cycle.ps1` with `NumIterations=6000` (about 24h on the current 5080 setup) and promoted candidate settings (`odd + schedule`) so control/candidate training, quick tests, and `base_eval` are serialized.
- [2026-02-10 19:21:04] Progress: full-cycle run started in background.
- Command: `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_full_cycle.ps1 -NumIterations 6000 -Seeds "1337,2026" -RunLabel odd_sched16r32_mix005_n6000`.
- Logs: `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000.out.log`, `experiments/nanochat_transfer/results/fullcycle_odd_sched16r32_mix005_n6000.err.log`.
- [2026-02-10 18:28:36] Promote the scheduled odd-layer candidate to a full 24h protocol run.
- Evidence: `odd_sched16r32_mix005` remained non-regressing at all screened windows:
- `n64`: `mean_candidate_minus_control_bpb=-0.000145`, `mean_candidate_speed_ratio=0.912058`
- `n128`: `mean_candidate_minus_control_bpb=-0.001091`, `mean_candidate_speed_ratio=0.917453`
- `n384`: `mean_candidate_minus_control_bpb=-0.000097`, `mean_candidate_speed_ratio=0.911722`
- Hypothesis: delayed/ramped gate activation is the first candidate robust enough for long-horizon promotion.
- Toggle plan: run `NumIterations=30000` with `CandidateOddLayersOnly`, `CandidateGateStartIter=16`, `CandidateGateRampIters=32`, fixed seeds, and structured JSON/CSV tracking.
- [2026-02-10 18:28:36] Formalize promotion thresholds for quality vs throughput.
- Hypothesis: implicit thresholds are slowing iteration and causing ambiguity when quality improves but speed drops.
- Toggle plan: codify an explicit `candidate_speed_ratio` floor (for example `>=0.90`) alongside `mean_candidate_minus_control_bpb <= 0`, and wire checks into docs/regression utilities.
- [2026-02-10 16:01:16] Re-run and stabilize mutation-transfer ranking under isolated GPU load.
- Evidence: isolated reruns are still split:
- `mutation_transfer_v4`: winner `mutation_champion` (`0.105628`) vs quorum budget (`0.108259`)
- `mutation_transfer_v5`: winner `mutation_champion` (`0.108445`) vs quorum budget (`0.109181`)
- `mutation_transfer_v6`: winner `quorum_budget_paging` (`0.104785`) vs mutation champion (`0.106363`)
- `mutation_transfer_v7`: winner `quorum_budget_paging` (`0.103228`) vs mutation champion (`0.104346`)
- Hypothesis: winner ordering can flip under GPU contention and requires repeatability checks before champion promotion.
- Toggle plan: execute additional tagged reruns and require either stable winner ordering or a <1% score tie band before updating the default "champion" narrative.

## Research Narrative (Cross-Field Source Set)
- Microbiology -> quorum thresholding: `https://pubmed.ncbi.nlm.nih.gov/37057353/`
- Genetics -> transcriptional memory and gated expression persistence: `https://pubmed.ncbi.nlm.nih.gov/39029953/`
- Chemistry -> reaction-network memory and temporal processing: `https://pubmed.ncbi.nlm.nih.gov/39058812/`
- Taste detection -> distributed coding under task demands: `https://pubmed.ncbi.nlm.nih.gov/38631343/`
- Odor detection -> combinatorial code remapping: `https://www.nature.com/articles/s41586-025-09053-w`
- Taxonomy science -> hierarchy/network-informed structure: `https://journals.asm.org/doi/10.1128/mbio.02256-23`
- Information theory -> rate/distortion/perception/semantics tradeoff: `https://doi.org/10.1016/j.jfranklin.2024.106884`
- Economic theory -> rational inattention with bounded information budgets: `https://link.springer.com/article/10.1007/s10683-023-09803-9`

## Future Directions
- [2026-02-08 17:23:12] Advanced next-next topic: Renormalization-Group Memory Ladder (multiscale coarse-graining + refinement).
- Parallels:
- Cosmology: scale-dependent structure growth and horizon crossing.
- Quantum physics / QFT: RG flow and effective theories by scale.
- String theory: worldsheet beta flow and scale consistency constraints.
- Statistics: hierarchical shrinkage / partial pooling across resolutions.
- Topology: persistent homology across filtration scales.
- Language theory: hierarchy from local syntax to discourse-level constraints.
- Fusion research: turbulence cascades and confinement across scales.
- [2026-02-08 17:38:45] Expand cross-field inspiration set for new experimental hypotheses:
- Neuroscience: predictive coding, hippocampal replay, attractor consolidation.
- Social psychology: context framing, priming persistence, group-level interference analogs.
- Linear algebra: low-rank + sparse decompositions, orthogonality control, spectral conditioning.
- [2026-02-08 17:58:10] Next next-next inspiration pool for cross-field experimental hypotheses:
- Microbiology: colony dynamics, quorum-like signaling, and resilience under stress.
- Genetics: regulatory circuits, expression gating, mutation/selection style search over gating policies.
- Chemistry: reaction kinetics and equilibrium analogs for memory update vs decay balance.
- Taste/odor detection: combinatorial receptor coding and sparse mixture separation.
- Taxonomy science: hierarchical class structure and stable branch-specific specialization.
- Information theory: bottleneck trade-offs, coding efficiency, and surprise-driven switching.
- Economic theory: resource allocation, market equilibrium, and incentive-compatible routing.
- [2026-02-10 08:34:39] Long-range direction: commoditize local evolutionary AI build loops.
- Focus: reusable mutation/search harnesses, reproducible benchmark protocols, and one-command local experiment orchestration.

## Process Notes
- Completed items move to `IMPLEMENTED.md` with timestamps and test evidence.
