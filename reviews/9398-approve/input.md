# Work Item Input

This is the verbatim input the workflow received. The basic-agent and
the implementation-agent both read this file before doing anything.

## Input

investigate and fix https://github.com/ai-dynamo/dynamo/issues/9398 — the dynamo.mocker AIC perf-model crashes with NoneType*int on NVFP4 MoE. Repro on the workstation GPUs (RTX A6000) if you can install dynamo + dependencies; otherwise reason through the code and produce a fix that addresses the perf_database extrapolation gap

## Received

2026-05-12T00:25:58.645Z
