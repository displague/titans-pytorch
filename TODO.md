# TODO

## Active
- [2026-02-10 15:07:39] Continue retuning candidate slot after confirmed regression at true short-window scale.
- Hypothesis: current token-complexity gate integration may need smaller mix or structural changes (not just optimizer tuning) to avoid quality loss.
- Toggle plan: keep baseline control fixed and sweep candidate-only flags in external harness (`--symplectic-gate-mix` near `0.01` and `0.02`, plus conservative optimizer settings), no Titans default changes.
- [2026-02-10 15:07:39] Add at least one structurally distinct candidate recipe beyond scalar mix tuning.
- Hypothesis: layer-local gating placement or schedule-based activation may preserve speed/quality trade-offs better than constant full-depth gating.
- Toggle plan: add one new optional candidate recipe slot in the transfer harness and benchmark it under `NumIterations=64` and then `128`.
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
