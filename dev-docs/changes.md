# Changes

## 2026-06-05

- Extend phrase bias semantics from positive-only to signed soft bias:
  positive `bias` boosts a continuation token and negative `bias` softly suppresses it.
- Change CT2 final phrase-bias clamp to signed range
  `[-max_token_delta, +max_token_delta]` while keeping the default
  `max_token_delta=2.0`.
- Update faster-whisper config semantics to allow signed finite `bias` values with
  defaults `default_total_bias=5.0`, `min_total_bias=-5.0`,
  `max_total_bias=5.0`, and `max_step_bias=2.0`.
- Add P5 implementation plan and update SSOT/README/dev-docs/example config to match
  the signed behavior. Hard block/suppress and start token bias remain out of scope.

## 2026-06-04

- Sync `dev-docs/SSOT.md` with the current faster-whisper phrase bias defaults:
  `default_total_bias=5.0`, `max_total_bias=5.0`, `max_step_bias=2.0`.
- Add an explicit `CLAUDE.md` rule that code behavior, APIs, defaults, and config
  schema changes must update `dev-docs/SSOT.md` and related `dev-docs/` files in
  the same change.
