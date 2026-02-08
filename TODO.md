# TODO

## Active
- [2026-02-08 16:12:00] Add a short `README.md` section documenting `SymplecticGating` options: `gated`, `diag`, `gate_mode`, `top_k`, and `adaptive_topk_ratio`, including citation links.
- [2026-02-08 16:12:00] Evaluate per-sample adaptive top-k (varying `k` per batch item / token) behind a toggle, with unit tests.

## Future Directions
- [2026-02-08 16:12:00] Add DMD vs symplectic vs combined gating ablation benchmark table output.
- [2026-02-08 16:12:00] Add paging stress tests that assert page isolation over multi-step state carry.
- [2026-02-08 16:12:00] Add optional benchmark CSV output for regression tracking in CI.

## Process Notes
- Completed items move to `IMPLEMENTED.md` with timestamps and test evidence.
