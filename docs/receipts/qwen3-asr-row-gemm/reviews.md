# Independent review closure

Both rounds review baseline `8dfdade4a4cd80d24bc901547c51fc1a35a4de8b` through clean snapshot
`4f6facf6b70660f923803d437d061f60798ec3eb`, including implementation, original evidence,
public conclusions, remaining limits and infrastructure facts. Round 2 uses fresh independent reviewers.
The source is [the frozen contract](../../recipes/qwen3-asr-row-gemm.md), not retrospective thresholds.
[Structured reports](reviews.json) retain scope and independent checks separately by axis.

## Standards

Round 1: PASS, zero open findings. Two P2 issues were corrected: strict token ID/work/layout validation
is shared across four acceptance tools, and three CUDA-free tests directly exercise malformed geometry,
pointers and unqualified cuBLASLt refusal. Independent checks include 54 Python tests, 20 additional
invalid-evidence mutations, host logs, 29 binder controls, outgoing privacy/identity and artifact binding.

Round 2: PASS, zero open violations or substantiated smells. Fresh checks include implementation ownership,
compiler/SM records, the original full CUDA log, all reference compiler guards, raw quality/chat/cancellation,
FP64 columns, complete public/archive binding and infrastructure inventory. The lease source is retained
as an evidence-only JSON snapshot in the infrastructure repository, preserving original bytes and SHA
without adding a lifecycle execution entrypoint.

## Spec

Round 1: PASS, zero open findings. All five served projection paths and original precision/quality gates
are covered. Documentation now distinguishes primary workloads of 128 tokens and profiling workloads of 64 tokens, batch makespan versus
request HTTP latency, and 43,113 timing-archive records versus the separate failed-admission warmup.

Round 2: PASS, zero open findings. Fresh recomputation covers the frozen bilingual corpus and gates,
all 689,664 FP64 column maxima, both actual-row chat processes, cancellation, cache and formal work pairs,
sustained-load minimums, clean-source artifacts, public/raw fidelity and three locked restoration probes.

Standards: zero remaining findings. Spec: zero remaining findings. Numerical-reference gaps, negative
experiments, interrupted-outcome loss, timing uncertainty and finite platform/production scope stay in
the qualification record. Final closure changes documentation and evidence metadata only.
