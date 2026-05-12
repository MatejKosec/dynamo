# Plan — Fix dynamo.mocker AIC perf-model NoneType*int crash on NVFP4 MoE

## User intent

Fix GitHub issue #9398: the dynamo.mocker AIC perf-model crashes with `NoneType*int`
on NVFP4 MoE models. The crash occurs when the AIC `context_latency` or `tpot` perf
lookup returns `None` (no database coverage for this model/precision/ISL combination),
and the Python code attempts arithmetic (`None * int` or `None + float`) instead of
handling the missing value. The secondary framing in the issue is a "perf_database
extrapolation gap" for ISLs above the database's max context length.

## Non-goals

- This plan does NOT change the aiconfigurator SDK itself (upstream dependency)
- This plan does NOT add new NVFP4 models to the AIC database
- This plan does NOT modify the Rust mocker scheduler logic (only the Python AIC bridge)
- This plan does NOT attempt to fix the NPZ-based `PerfModel::Interpolated` path; only the
  `Aiconfigurator` (Python-callback) path at `lib/bindings/python/src/dynamo/_internal/aic.py`
- This plan does NOT require GPU hardware for the fix itself; the local RTX A6000 is useful
  for smoke-testing the mocker end-to-end but not strictly needed for validating the code change

## Safety constraints

none — internal change, no external API surface; the fix only adds defensive None-checks
and a fallback pattern inside a routine that is already documented to potentially raise

## Inspected context

### Target files

**`lib/bindings/python/src/dynamo/_internal/aic.py`** (primary target)
- `AicSession.__init__`: builds `context_ops` and `generation_ops` lists by iterating AIC
  model ops; on MoE models these lists can be empty if the aiconfigurator SDK has no
  registered ops for the model's precision/dtype
- `_predict_context_latency` (line 110–134): calls `op.query(...)` on each context op; the
  `result` is `float(result)` which becomes `None` if the database has no entry for the
  (model, dtype, ISL) key — then `total_latency += float(None)` raises `TypeError`
- Line 113 guard (`if effective_isl <= 0`) is correct for its narrow case but the `result
  is None` case is unhandled
- `_predict_generation_latency` (line 136–160): same issue with `generation_ops`;
  additionally line 140 uses `self._model._nextn` which is `None` on non-speculative-decoding
  models, making `effective_batch_size = batch_size * (None + 1)` raise `TypeError` before
  ever reaching the `op.query()` call
- `predict_prefill` and `predict_decode` expose the same problem: callers in
  `lib/bindings/python/rust/llm/aic_callback.rs` invoke `callback.predict_prefill(...)` and
  `callback.predict_decode(...)` and the `.unwrap_or_else(|e| panic!(...))` pattern means any
  Python exception (TypeError from None*int, RuntimeError from missing database) propagates
  as a panic

**`lib/bindings/python/rust/llm/aic_callback.rs`** (parallel error path)
- `PyAicCallback::predict_prefill` and `predict_decode`: `.unwrap_or_else(|e| panic!(...))`
  means Python-side TypeError becomes a Rust panic; fix belongs in Python, not here
- `create_aic_callback` and `create_aic_prefill_load_estimator`: these bridge functions are
  correct; the fix point is the Python they call

### Sibling / parallel patterns

**`components/src/dynamo/planner/monitoring/aic_interpolation.py`** (similar guard pattern, in scope)
- `_sweep_prefill` (line 85–117): already has `if ttft_ms is None or ttft_ms <= 0` guard
- `_sweep_decode` (line 120–190): already has `if itl_ms is None or itl_ms <= 0` guard
- shows the established pattern: check None, log warning, skip point
- **sibling bug**: `max_concurrency_aggregate` computation at line 153 is
  `max_kv_tokens_aggregate // (isl + osl_sweep)` — if the sweep ISL range is large and
  `per_rank_max_kv` is small (AIConfigurator memory estimation too low for NVFP4), this
  produces 0 for many ISL values, silently producing a sparse decode interpolation grid

**`components/src/dynamo/planner/monitoring/aic_estimator.py`** (AIC estimator, parallel path)
- `estimate_perf` returns `summary_df.to_dict(orient="records")[0]` directly without checking
  for null columns; if the database has no entry for the given (ISL, OSL, batch_size) the
  DataFrame may contain None values
- `get_max_kv_tokens`: binary-searches batch size via `backend._get_memory_usage`; if the
  model doesn't fit at batch_size=1 this returns 0 and the caller receives 0, which is
  handled by the `if per_rank_max_kv <= 0` guard in aic_interpolation.py

**`lib/mocker/src/common/perf_model.rs`** (same bug shape in Rust, different backend)
- `from_npz`: validates dimensions but does not guard against NaN/None in loaded arrays
- Both `Linear::new().extrapolate(true)` and `Bilinear::new().extrapolate(true)` are enabled,
  meaning out-of-range ISL/decode-grid lookups succeed with extrapolated values — this is
  part of the "extrapolation gap" framing: the NPZ path extrapolates silently while the
  AIC path crashes on None

### Source availability
Source is fully available at `/out/2026-05-12_00-25-58__dynamo-9398/sources/dynamo`.

## Chosen approach

Modify `lib/bindings/python/src/dynamo/_internal/aic.py` — two separate fixes:

**Fix 1 (primary — None*int crash):** In `_predict_context_latency`, after each
`op.query(...)` call, check if `result is None` before calling `float(result)`.
If None, log a warning at DEBUG level and skip that op (treat its contribution as 0).
Pattern matches the existing guard in `aic_interpolation.py`:
```python
result = op.query(...)
if result is None:
    logger.debug("AIC op=%s returned None for model=%s, x=%s; skipping",
                op_name, self._model_name, x)
    continue
total_latency += float(result)
```

**Fix 2 (precondition — _nextn None crash):** In `_predict_generation_latency`, guard
`self._model._nextn` with `getattr(self._model, '_nextn', 0)` before using it:
```python
_nextn = getattr(self._model, '_nextn', 0)
effective_batch_size = batch_size * (_nextn + 1)
```
The `+1` is a speculation因子; 0 is the correct fallback for non-speculative models.
Also add a `result is None` check inside the `generation_ops` loop with the same skip pattern.
Finally, add a top-level `if not self._model.context_ops:` and `if not
self._model.generation_ops:` guard at entry to each predict method, returning 0.0 with a
warning — this handles the empty-ops MoE case gracefully.

**Why not fix the extrapolation gap in perf_model.rs**: The NPZ interpolation path uses
`extrapolate(true)` which already extrapolates silently — that path doesn't crash on
out-of-range ISL, it just uses the nearest-boundary polynomial value. The AIC crash is
a separate bug (missing database entries returning None). The "extrapolation gap" is
therefore about ensuring AIC returns a number rather than None; the fix above addresses
that by falling back to 0 for missing op entries rather than crashing.

If the extrapolation behavior is desired for AIC (i.e., users want AIC to use nearest-boundary
values instead of None), that would be a separate enhancement requiring AIC SDK changes outside
dynamo's scope.

## Rejected alternatives

1. **Return a large sentinel value from predict_prefill/decode when database misses** — rejected
   because sentinels propagate as wildly wrong latencies into the mocker scheduler, causing
   incorrect simulate-then-deploy decisions; warning + skip is the correct semantics: a
   missing perf entry means "I don't know" not "it's slow"
2. **Fix in Rust aic_callback.rs with .unwrap_or(0.0)** — rejected because the Python
   `.call_method1(...).and_then(|r| r.extract::<f64>(py))` already returns a PyResult; the
   TypeError originates inside Python code before theRust FFI boundary, so the fix belongs
   in the Python callee, not at the FFI boundary
3. **Modify aic_interpolation.py to guard all perf dict lookups** — aic_interpolation.py
   already guards its results (null check on ttft_ms/tpot_ms); its bug is different
   (max_concurrency_aggregate becoming 0 when per_rank_max_kv is underestimated for
   NVFP4), which is out of scope for this bug

## Validation strategy

**No GPU needed for code change validation.** The fix is a pure Python defensive-coding
change. The implementer can:
1. Import the modified `_internal/aic.py` module and call `AicSession(...)` with a
   trace-driven model path that exercises the empty-ops path
2. Run the existing `test_replay_aic_parity.py` suite with local GPU if aiconfigurator
   is installable, or mark the AIC-dependent tests as `pytest.mark.skip` if the
   optional dependency is absent
3. Run unit tests for `components/src/dynamo/mocker/` to confirm no regression

**Optional GPU smoke (local RTX A6000):** If dynamo + aiconfigurator can be installed in
the container, run the mocker end-to-end with an NVFP4 MoE model config to confirm it
does not crash. This is not a blocker — the code change is self-contained.

## Required deliverables

- `change.md` — implementer's notes: branch name, commit SHA, file:purpose list, any
  "tried X, didn't work" notes
- `change.diff` — the full diff of changes
- `change-validation.md` — validator's verdict (pass / blocked / fail)
- `review.md` — reviewer's file:line findings and verdict

## Hardware target

no hardware needed — local RTX A6000 available but not required for validating this fix