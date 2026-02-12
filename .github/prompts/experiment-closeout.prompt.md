---
mode: agent
description: "Close out an experimental change by synchronizing TODO/IMPLEMENTED/README and validation evidence."
---

Close out the current experiment in this repository using `AGENTS.md` as policy:

1. Verify new behavior is toggle-gated and defaults preserve baseline behavior.
2. Run or summarize targeted validation commands for touched behavior.
3. Update `TODO.md` and `IMPLEMENTED.md` with timestamped, consistent entries.
4. Update `README.md` only when user-facing behavior changed.
5. Keep edits scoped and non-interactive git-friendly.

Return:
- The files changed.
- The validation evidence captured.
- Any unresolved risks or follow-up items.
