# TODO

## Active
- [2026-02-08 16:54:24] Evaluate per-sample adaptive top-k (varying `k` per batch item / token) behind a toggle, with unit tests.
- [2026-02-08 16:54:24] Prototype manifold-state cache keys (`phase angle`, `radius`) for optional paging/cache lookup.

## Future Directions
- [2026-02-08 16:54:24] Add DMD vs symplectic vs combined gating ablation benchmark table output.
- [2026-02-08 16:54:24] Add paging stress tests that assert page isolation over multi-step state carry.
- [2026-02-08 16:54:24] Add optional benchmark CSV output for regression tracking in CI.
- [2026-02-08 16:54:24] Add long-horizon drift benchmark (phase-consistency and recovery after perturbation).

## Process Notes
- Completed items move to `IMPLEMENTED.md` with timestamps and test evidence.
