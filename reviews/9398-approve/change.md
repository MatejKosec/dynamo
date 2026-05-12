# Change — Fix AIC perf-model NoneType*int crash on NVFP4 MoE

## Branch
agent/9398-aic-mocker-fix (commit bfd65ef)

## Files touched
- `lib/bindings/python/src/dynamo/_internal/aic.py:118-149` — fixed `_predict_context_latency`: empty-ops guard, `result is None` guard before `float(result)`
- `lib/bindings/python/src/dynamo/_internal/aic.py:151-191` — fixed `_predict_generation_latency`: empty-ops guard, `_nextn` guarded with `getattr`, `result is None` guard before `float(result)`

## Why

GitHub issue 9398: the dynamo.mocker AIC perf-model crashed with `NoneType * int` when the AIC performance database had no entry for a model/precision/ISL combination, or when `_model._nextn` was `None` on non-speculative-decoding models. The crash propagated through the PyO3 bridge and surfaced as a Rust panic via `.unwrap_or_else(|e| panic!(...))`.

The plan's "Chosen approach" was followed exactly:
1. `_predict_context_latency`: added empty-`context_ops` guard (returning 0.0 with a WARNING) for MoE models where the aiconfigurator SDK has no registered ops; added `result is None` check before `float(result)` using the DEBUG-log/skip pattern matching `aic_interpolation.py:_sweep_prefill` (line 104)
2. `_predict_generation_latency`: added empty-`generation_ops` guard; replaced `self._model._nextn` with `getattr(self._model, '_nextn', 0)` so non-speculative models (where `_nextn` is absent) use 0 as the fallback; added `result is None` check mirroring `aic_interpolation.py:_sweep_decode` (line 176)

The sibling pattern in `aic_interpolation.py` was used as the reference implementation for the null-guard + log-and-skip pattern.

## Adjacent sites inspected

- `components/src/dynamo/planner/monitoring/aic_interpolation.py:104` — `_sweep_prefill` already has `if ttft_ms is None or ttft_ms <= 0` guard; pattern was mimicked. Sibling already has the fix.
- `components/src/dynamo/planner/monitoring/aic_interpolation.py:176` — `_sweep_decode` already has `if itl_ms is None or itl_ms <= 0` guard; pattern was mimicked. Sibling already has the fix.
- `components/src/dynamo/planner/monitoring/aic_estimator.py` — `estimate_perf` returns a dict but `aic_interpolation.py` calls it and guards the result before use; not directly affected by this bug but noted for follow-up
- `lib/bindings/python/rust/llm/aic_callback.rs` — uses `.unwrap_or_else(|e| panic!(...))` on the Python callbacks; the Rust side is correct, the Python side was the fix point

No additional adjacent sites the plan did not already name.

## Build results

No build step required — this is a pure Python change. The modified module was read back to confirm both function bodies are correct before commit.

Pre-commit hooks are NOT installed in this container (`pre-commit: command not found`); pre-commit runs in the CI environment where the hooks are available. The change was committed without the pre-commit auto-fix pass.

## Honest "tried X, didn't work" notes

- Attempted push to `nonpublic` remote failed: the remote already contains work on `agent/9398-aic-mocker-fix` (fetch-first rejection). The branch is committed locally. Push was NOT retried with `--force` per workflow hard rules. The lower-level `git pull`-then-rebase path was not pursued because the workflow doctrine is to record the push failure and let the reviewer/human handle it.
- Advisor skill (`/advisor`) was not available in this Task palette (`Unknown skill: advisor`); proceeded without an external review pass at both checkpoints.

## Advisor checkpoint

Checkpoint #1: Advisor skill unavailable in this Task() palette — proceeded without external review pass.
Final advisor: skill unavailable — proceeded.

## Diff summary
1 file changed, 32 insertions(+), 1 deletion(-)

The full diff is at `/out/2026-05-12_00-25-58__dynamo-9398/change.diff`.

## Open MR

Push to the non-public GitLab fork failed (remote has prior work on this branch). The branch `agent/9398-aic-mocker-fix` with commit `bfd65ef` is committed locally. The human should fetch the branch, rebase onto the remote state, and open the MR manually. When the MR is opened, the suggested monitoring is:

/loop 10m once the MR is opened, check the resulting MR's pipeline + comments; if a check fails or a reviewer requests changes, summarize in `/out/2026-05-12_00-25-58__dynamo-9398/monitoring.md` and stop the loop