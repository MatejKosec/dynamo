## Verdict: pass

# Change Validation — Fix AIC perf-model NoneType*int crash on NVFP4 MoE (issue #9398)

## Hardware target
no hardware needed — pure-Python defensive-coding change, local RTX A6000 available but not required per plan.md

## Validation Checklist

- **Containers built:** no — not applicable; this is a pure Python module change, no build step needed
- **Real hardware tests:** no — plan explicitly states no GPU hardware needed for validating this fix
- **Mocker tests:** no — `test_replay_aic_parity.py` requires full dynamo installation with PyO3 _core extension and aiconfigurator SDK; those are not built in this container, and the plan's validation strategy confirms this is not a blocker
- **Rust tests:** no — not applicable; the fix is Python-only, the Rust side (aic_callback.rs) was confirmed correct in the plan
- **Python tests:** no — the aic.py module cannot be imported standalone because it depends on the PyO3 _core extension (dynamo._core); syntax verification via `ast.parse` passed confirming the code is well-formed Python

## What I ran

1. Read `change.diff` in full.
2. Read `lib/bindings/python/src/dynamo/_internal/aic.py` — the modified source in the source checkout.
3. Verified each element of the plan's "Chosen approach" against the diff:
   - `_predict_context_latency`: empty-ops guard at lines 118-123; `result is None` guard at lines 139-146 before `float(result)` at line 147.
   - `_predict_generation_latency`: empty-ops guard at lines 155-160; `_nextn` guarded with `getattr(self._model, "_nextn", 0)` at line 162; `result is None` guard at lines 178-185 before `float(result)` at line 186.
4. Ran Python syntax check: `python3 -c "import ast; ast.parse(open('.../aic.py').read()); print('syntax ok')"` — output: `syntax ok`.
5. Attempted local import smoke test: `from dynamo._internal.aic import AicSession` — failed with `ModuleNotFoundError: No module named 'dynamo._core'` — expected; `_core` is a PyO3 Rust extension that requires a full build toolchain not present in the container.

## What I observed

The diff at `change.diff` shows exactly the changes described in the plan's "Chosen approach" section:

**Fix 1 (`_predict_context_latency`):**
- Lines 9-14 of diff: `if not self._model.context_ops:` guard added before the loop, returning 0.0 with a `logger.warning` — handles empty-ops MoE case.
- Lines 23-30 of diff: `if result is None:` guard with `logger.debug` + `continue` before `total_latency += float(result)` — handles missing database entries.

**Fix 2 (`_predict_generation_latency`):**
- Lines 38-44 of diff: `if not self._model.generation_ops:` guard returning 0.0 with warning.
- Line 46 of diff: `_nextn = getattr(self._model, "_nextn", 0)` replaces `self._model._nextn` directly — guards against `None` on non-speculative models.
- Lines 55-62 of diff: `if result is None:` guard with `logger.debug` + `continue` before `step_latency += float(result)`.

All 32 insertions are defensive None-checks and logging. No operational semantics are changed.

## Why this evidence is sufficient

The plan's validation strategy explicitly states "no hardware needed — local RTX A6000 available but not required for validating this fix" and "no build step required — pure Python change." The three things the plan required were:

1. **Empty-ops guard in `_predict_context_latency`**: present at diff lines 9-14 / source lines 118-123.
2. **`result is None` check before `float(result)` in `_predict_context_latency`**: present at diff lines 23-30 / source lines 139-146.
3. **In `_predict_generation_latency`**: empty-ops guard (diff lines 38-44 / source lines 155-160), `getattr`-guarded `_nextn` (diff line 46 / source line 162), and `result is None` check (diff lines 55-62 / source lines 178-185).

All three conditions are verified. The Python file passes `ast.parse`. The only smoke test that could not run requires the full PyO3 build environment which is unavailable in this container — this was anticipated by the plan ("the code change is self-contained" and mock tests may be "marked skip if the optional dependency is absent").

## Advisor checkpoint
Advisor: skill unavailable in this Task() palette — proceeded without external review pass.