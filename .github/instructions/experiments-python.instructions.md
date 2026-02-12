---
applyTo: "**/*.py"
description: "Keep experimental code optional, deterministic, and benchmarkable"
---

# Python experiment policy

- Default behavior must remain unchanged unless a toggle is explicitly enabled.
- Add or update tests/benchmarks alongside non-trivial behavior changes.
- Use deterministic seeds and small shapes for fast repeatable checks where practical.
- Keep experimental logic close to the owning module and avoid broad rewrites.
- Emit structured summaries (JSON/CSV) for benchmark/protocol outputs when feasible.
