# TODO

## Active
- [2026-02-10 16:01:16] Continue retuning candidate slot after odd-layer structural probe.
- Evidence: odd-layer candidate at `n64` improved over full-depth candidate (`mean_candidate_minus_control_bpb`: `+0.002307` vs `+0.003640`, `mean_candidate_speed_ratio`: `0.918` vs `0.859`), but at `n128` still regressed (`+0.038428`, speed `0.911`).
- Hypothesis: schedule-based gate activation (warmup/ramp) can preserve throughput gains while reducing quality penalty at longer windows.
- Toggle plan: add candidate-only schedule knobs in the external `nanochat` patch and sweep `--symplectic-gate-mix` in `0.01`, `0.02`, and `0.05` with odd-layer mode enabled.
- [2026-02-10 16:01:16] Re-run and stabilize mutation-transfer ranking under isolated GPU load.
- Evidence: clean rerun `mutation_transfer_v4` promoted `mutation_champion` (`0.105628`) over `quorum_budget_paging` (`0.108259`), reversing prior contention-affected ranking.
- Hypothesis: winner ordering can flip under GPU contention and requires repeatability checks before champion promotion.
- Toggle plan: execute two additional tagged reruns and require stable winner ordering (or <1% score tie) before updating the default "champion" narrative.
- [2026-02-10 15:07:39] Promote a non-regressing short-window winner to full 24h run with checkpointed reporting.
- Hypothesis: only candidates that beat control on both `val_bpb` and acceptable speed ratio in short windows should enter the 24h budget.
- Toggle plan: run only external harness configs; no change to Titans runtime defaults.

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
