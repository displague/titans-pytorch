# titans-pytorch Copilot Instructions

This repository uses `AGENTS.md` at the workspace root as the canonical instruction source.

## Required behavior
- Preserve baseline behavior by default; put new behavior behind explicit toggles.
- Drive work from `TODO.md`; move completed timestamped items to `IMPLEMENTED.md`.
- Prefer reusable tests in `tests/` and benchmarks in `benchmarks/` over ad-hoc scripts.
- Keep edits modular, scoped, and PR-friendly.
- Record non-trivial theory/hypothesis/interpretation context in docs.
- Preserve temporary experiments with named git stashes instead of deleting work.

## Process file expectations
- `TODO.md`: active/planned work with hypothesis and toggle plan.
- `IMPLEMENTED.md`: timestamped execution history with validation evidence.
- `README.md`: user-facing usage and reproducible commands only.

## Optional reusable resources
- Scoped overlays: `.github/instructions/*.instructions.md`
- Reusable workflows: `.github/skills/*/SKILL.md`
- On-demand prompts: `.github/prompts/*.prompt.md`
