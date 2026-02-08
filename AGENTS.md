# titans-pytorch — AGENTS.md

## Purpose
Portable directions for experimental work in this repo, so improvements remain optional, testable, and PR-friendly.

## Core Principles
- Prefer **reusable Python tests** over throw-away scripts or ad-hoc experiments.
- Keep **all improvements toggle-able** (no change to default behavior).
- **Avoid sprawling file changes**; keep experimental additions modular.
- Document **research, theory, intent, and direction** whenever adding improvements.
- PRs should be **opt-in enhancements**, not behavior changes.

## Workflow Rules
1. **Toggle First**
   - New features must be behind explicit flags or constructor args.
   - Defaults must preserve current behavior.

2. **Test First**
   - Any exploratory work should land as a test in `tests/` or a benchmark in `benchmarks/`.
   - Keep tests deterministic when possible (fixed seeds, small sizes).

3. **Modular Additions**
   - Prefer adding new modules or small extensions over altering large existing flows.
   - Keep experimental logic near its related component (e.g., gating logic in `symplectic_gate.py`).

4. **Preserve Throw-Away Work**
   - If temporary experiments are needed and not converted to tests, **stash them in git** with an identifiable description instead of deleting.
   - Prefer `git stash push -m "<short description>"` so work remains recoverable and portable across tools.

5. **Documentation & Citation**
   - When adding improvements, document:
     - **Research & theory** motivating the change
     - **Intent** (what hypothesis is being tested)
     - **Direction** (how to interpret success/failure)
   - Include citations to relevant papers or resources in docstrings or README notes.

## Practical Guidance
- **Tests over notebooks**: If you need a one-off experiment, write it as a small test or benchmark instead.
- **Maintain backward compatibility**: No silent behavior changes.
- **Scoping**: Keep changes focused; avoid editing unrelated files.

## Suggested Locations
- Tests: `tests/`
- Benchmarks: `benchmarks/`
- Documentation notes: `README.md` or module docstrings

## Notes
These instructions mirror experimental practices previously used in external tooling and are intended to keep work portable and reviewable.
