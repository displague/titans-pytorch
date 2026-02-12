# titans-pytorch - AGENTS.md

## Purpose
Canonical, cross-agent instructions for experimental work in this repository. This file is the source of truth across Codex, VS Code agents, GitHub Copilot, and Copilot coding agent workflows.

## Agent Portability Contract
- Keep repository policy in this file (`AGENTS.md`) so multiple agents share one instruction baseline.
- Use additional nested `AGENTS.md` files only for subdirectory-specific overrides (for example, isolated experiment harnesses).
- Mirror concise compatibility guidance in `.github/copilot-instructions.md` for tools that auto-detect that file.
- Use `.github/instructions/*.instructions.md` for file-scoped overlays (for example `TODO.md`/`IMPLEMENTED.md` handling).
- Use `.github/skills/*/SKILL.md` for reusable task workflows that should load on demand.
- Use `.github/prompts/*.prompt.md` for on-demand playbooks (for example "close out experiment run").
- If instructions conflict: follow system/developer/user directives first, then this file, then narrower overlays.

## Core Principles
- Work is driven from `TODO.md` with timestamps.
- Completed work moves from `TODO.md` to `IMPLEMENTED.md` with timestamps and evidence.
- Prefer reusable Python tests and benchmarks over throw-away scripts.
- Keep improvements toggleable so default behavior does not change.
- Keep changes modular and avoid sprawling edits.
- Document research context, theory, intent, and direction for non-trivial improvements.
- Preserve temporary work with named git stashes instead of deleting it.

## Workflow Rules
1. Toggle first
- New behavior must be behind explicit flags or constructor args.
- Defaults must preserve existing behavior.

2. Test first
- Exploratory work should land as tests in `tests/` or benchmarks in `benchmarks/`.
- Keep tests deterministic where possible (fixed seeds, small shapes).

3. Modular additions
- Prefer focused module-level changes rather than broad rewrites.
- Keep experimental logic close to the owning component.

4. Preserve throw-away work
- If temporary experiments are not yet reusable, stash them instead of deleting.
- Use identifiable messages, for example:
- `git stash push -m "experiment: symplectic adaptive-k probe"`

5. Documentation and citation
- For each enhancement, record:
- Theory or paper context.
- Intent and hypothesis.
- Success and failure interpretation.
- Add references in docstrings, `README.md`, or `IMPLEMENTED.md`.

6. Progress tracking
- `README.md`: user-facing usage and feature docs.
- `TODO.md`: active and planned work with timestamps.
- `IMPLEMENTED.md`: timestamped implementation history and decisions.
- When an item is completed, update both files to reflect the move.

## Process File Schema
- `TODO.md` entries should include: timestamp, hypothesis, toggle plan, and expected success/failure metrics.
- `IMPLEMENTED.md` entries should include: timestamp, changed files/scripts, outcome, and validation commands/artifacts.
- `README.md` should document only user-facing behavior and reproducible usage paths.
- Prefer machine-readable artifacts (JSON/CSV) for benchmark/protocol summaries.

## Cross-Field Research Pool
- Cosmology.
- Quantum physics / QFT.
- String theory.
- Statistics.
- Topology.
- Language theory.
- Fusion research.
- Neuroscience.
- Social psychology.
- Linear algebra.
- Microbiology.
- Genetics.
- Chemistry.
- Taste and odor detection.
- Taxonomy science.
- Information theory.
- Economic theory.

## Cross-Field Integration Guidance
- Map each external concept to one explicit model hypothesis.
- Add one toggle for each hypothesis so baseline behavior stays unchanged.
- Add one benchmark metric and one failure metric for each hypothesis.
- Record why the mapping might fail, not only why it might work.

## Practical Guidance
- Backward compatibility is required unless a breaking change is explicitly requested.
- Keep edits scoped to the task.
- Use non-interactive git commands.
- Prefer workspace-relative paths in docs so instructions remain portable across machines and agents.

## External Transfer Pilots
- When evaluating ideas in external repos (for example, `nanochat`), isolate work under a clearly named folder and keep it optional.
- Use `experiments/nanochat_transfer/` as the canonical harness for `nanochat` transfer work.
- Keep external source edits reproducible by storing patch artifacts and apply/revert scripts in the harness.
- Prefer adapter/config layers and benchmark scripts over invasive source rewrites.
- Define one reproducible "control vs champion" recipe for 16GB GPUs before running long jobs.
- Use `experiments/nanochat_transfer/run_nanochat_full_cycle.ps1` for promotion runs so protocol + quick tests + eval are captured in one pass.
- Record runtime budget, seed, checkpoint cadence, and success/regression metrics in `TODO.md` and `IMPLEMENTED.md`.
- Emit structured run summaries (JSON/CSV) from harness scripts so directional results are machine-readable and easy to compare across reruns.
- Do not run GPU-heavy pytest and benchmark/protocol jobs at the same time; serialize heavy runs to avoid cross-run contamination.
- If contention or interruption is suspected, rerun with a new tag and record both superseded and replacement runs in `IMPLEMENTED.md`.

