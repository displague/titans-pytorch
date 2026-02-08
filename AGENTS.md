# titans-pytorch - AGENTS.md

## Purpose
Portable directions for experimental work in this repository so changes remain optional, testable, and PR-friendly.

## Core Principles
- Work is driven from `TODO.md` with timestamps.
- Completed work moves from `TODO.md` to `IMPLEMENTED.md` with timestamps.
- Prefer reusable Python tests and benchmarks over throw-away scripts.
- Keep improvements toggleable so default behavior does not change.
- Keep changes modular and avoid sprawling edits.
- Document research context, theory, intent, and direction for non-trivial improvements.
- Preserve temporary work with named git stashes instead of deleting it.

## Workflow Rules
1. Toggle First
- New behavior must be behind explicit flags or constructor args.
- Defaults must preserve existing behavior.

2. Test First
- Exploratory work should land as tests in `tests/` or benchmarks in `benchmarks/`.
- Keep tests deterministic where possible (fixed seeds, small shapes).

3. Modular Additions
- Prefer focused module-level changes rather than broad rewrites.
- Keep experimental logic close to the owning component.

4. Preserve Throw-Away Work
- If temporary experiments are not yet reusable, stash them instead of deleting.
- Use identifiable messages, for example:
- `git stash push -m "experiment: symplectic adaptive-k probe"`

5. Documentation and Citation
- For each enhancement, record:
- Theory or paper context.
- Intent and hypothesis.
- Success and failure interpretation.
- Add references in docstrings, `README.md`, or `IMPLEMENTED.md`.

6. Progress Tracking
- `README.md`: user-facing usage and feature docs.
- `TODO.md`: active and planned work with timestamps.
- `IMPLEMENTED.md`: timestamped AI implementation history and decisions.
- When an item is completed, update both files to reflect the move.

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

## Cross-Field Integration Guidance
- Map each external concept to one explicit model hypothesis.
- Add one toggle for each hypothesis so baseline behavior stays unchanged.
- Add one benchmark metric and one failure metric for each hypothesis.
- Record why the mapping might fail, not only why it might work.

## Practical Guidance
- Backward compatibility is required unless a breaking change is explicitly requested.
- Keep edits scoped to the task.
- Use non-interactive git commands.

