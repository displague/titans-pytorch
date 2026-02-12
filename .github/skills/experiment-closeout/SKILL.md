# Experiment Closeout

Use this skill when finishing an experiment implementation so planning, evidence, and docs stay synchronized.

## Goal
Close out work in a reproducible, PR-friendly way without changing baseline behavior by default.

## Steps
1. Confirm toggles
- Verify all new behavior is opt-in and defaults preserve current behavior.

2. Capture evidence
- Run targeted tests and/or benchmarks.
- Keep commands deterministic where practical (explicit seed, bounded steps).
- Prefer JSON/CSV outputs for directional comparisons.

3. Update process files
- In `TODO.md`, remove or mark the completed active item.
- In `IMPLEMENTED.md`, add a timestamped entry with:
- changed files/scripts,
- hypothesis/outcome summary,
- validation commands and key results.

4. Update user docs
- If user-facing behavior changed, update `README.md` with flags, examples, and known tradeoffs.

5. Preserve temporary work
- If exploratory artifacts are not ready to land, stash with a named message instead of deleting.
- Example: `git stash push -m "experiment: <short-name>"`

## Exit criteria
- Code path is toggle-gated.
- Evidence is recorded.
- `TODO.md` and `IMPLEMENTED.md` are consistent.
- User-facing docs are current.
