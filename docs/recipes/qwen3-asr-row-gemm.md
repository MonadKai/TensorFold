# Qwen3-ASR cross-row CUDA projection contract

This downstream issue #1 contract is fixed before implementing or measuring candidate arithmetic.
Baseline: `8dfdade4a4cd80d24bc901547c51fc1a35a4de8b`; original BF16 checkpoint
`7278e1e70fe206f11671096ffdd38061171dd6e5`, Linux AArch64, GB10 sm_121, one rank.
The existing non-streaming API and its explicit refusals remain the contract. This is not an upstream
landing agreement, a new model family, quantization, speculative decoding, or a production capacity promise.

## Arithmetic identities

`stable` retains the existing single-row algorithm in strided batches, separate original prefill projections,
BF16 activations/weights and FP32 accumulation. Its byte-replacement registrations, stage exports and native
greedy regression remain mandatory and unchanged. The baseline binary and fixtures are preserved separately.

A separately identified `throughput` path may compute actual cross-row products X[B,K] W[N,K]^T using
ordinary GEMM. A fixed physical row count and a fixed algorithm per projection shape must make its row
arithmetic independent of logical B=1/2/4/8, row permutation, slot identity and other requests ending or
being cancelled. Microbenchmarks compare unpadded GEMM, physical widths 8/16/32, the stable path, multiple
cuBLASLt candidates and workspace limits 0/8/32/64 MiB before selection. Physical padding is not logical
concurrency and is reported separately. No silent per-call switch back to another numerical mode is allowed.
Unsupported tuned configurations must either refuse explicit throughput admission or use an explicitly
reported, qualified configuration; stable remains independently available.

The five required projection classes are packed QKV (N=4096,K=2048), attention output (2048,2048), packed
gate/up (12288,2048), down (2048,6144) and LM head (151936,2048). LM head includes the first prefill token
and subsequent decode tokens. Prefill changes are separate ablations, not an implicit new oracle.

BF16 rounding remains after each linear output; embedding replacement; each residual addition; RMSNorm
output; Q/K normalization; RoPE/KV publication; attention output; SiLU and multiply at existing SwiGLU
nodes; final normalization; and logits before sampling. FP32 reductions do not become FP16, BF16 or TF32.
A residual/RMSNorm fusion must reproduce both the BF16 residual and BF16 normalized output, including
aliasing, with intermediate probes and a byte comparison against the existing unfused sequence.

## Independent numerical admission

The stable byte-equality gate is not relaxed. Throughput is a distinct arithmetic contract, not a renamed
byte replacement. Its BF16 projection output must satisfy the same independent FP64 formula and existing
FP32 accumulation envelope: output half-ulp plus gamma(K+1) times the sum of absolute products. The bound
is derived from the operator's actual BF16 nodes and FP32 reduction, not fitted to candidate observations.
Both candidate and baseline errors are evaluated against the same bound; normalized error must not exceed
one. Report every column's maximum absolute and normalized error, aggregate relative L2, columns where the
candidate is farther from FP64, and nonfinite classes. Pointwise superiority over the old output is not
assumed. This policy does not weaken an old kernel's threshold or replace its original oracle.

For model fidelity, teacher-force identical histories, preserve every logits row and all changed top-1
choices, both margins and the two contenders. Report greedy differences separately from numerical error.
A diagnostic near-tie is only the same unordered pair in both paths with both margins no larger than two
BF16 ulps at the maximum absolute contender logit. It never erases a difference from counts. Whole-model
admission additionally requires relative logits L2 <=0.005 and per-element change <=max(0.5,4 BF16 ulps of
the reference value). This is a declared empirical admission guard, not a theoretical network-wide bound.
Default promotion requires >=99.9% teacher-forced top-1 agreement and no non-near-tie disagreements.

Every new operator runs five byte-identical repeats, unchanged input/guards, declared alias checks,
nonfinite rules, served shapes and valid/invalid edges, and independent FP32 exact-sum stress. Planted
controls must turn red. Focused diagnostics precede the unfiltered complete CUDA backend, including limits.
Keep old known findings and compiler base/pending/overrides distinct from newly introduced failures.

## Recognition corpus fixed before inference

Use 128 English and 128 Mandarin utterances, 16 test speakers per language and eight utterances per speaker.
English is OpenSLR LibriSpeech test-clean, official archive MD5 `32fa31d27d2e1cad72775fee3f4849a9`;
archive SHA256, selected file hashes and speaker metadata are frozen locally before any recognizer runs.
Mandarin is AISHELL-1 test via `TwinkStart/AISHELL-1@2a509ef3a7a88d3234205eaeecba39fdd9d50518`.
Pin the actual parquet files/hashes and original speaker IDs. Attribute LibriSpeech CC BY 4.0 and AISHELL-1
Apache-2.0 to their original publishers, and identify the test mirror separately.

Select the first 16 speaker IDs in numeric order. Within each speaker sort by decoded sample count then
utterance ID; choose indices round(j*(count-1)/7), j=0..7, with ties to even. This samples the duration range
without inspecting ASR results. Each speaker must supply eight distinct valid clips or acquisition fails.
Normalize losslessly to mono PCM16/16 kHz with the fixed native FFmpeg toolchain; retain original and final
hashes, duration, selection rule, text/label hashes and manifest SHA. Freeze the complete manifest before
baseline or candidate recognition; do not replace utterances after inspecting results.

Requests are greedy, seed 0, context 8192, output cap 1024 per chunk, explicit en/zh, empty context prompt,
with the unchanged transcription splitter/joiner. Scoring uses the existing exact normalization: NFKC,
casefold, `re.sub(r"[^\w\s]", "", text)` and whitespace collapse (punctuation and symbols are removed).
Mandarin CER additionally removes whitespace, with no number rewriting,
script conversion or lexicon fixes. Report normalized and raw text differences; keep complete raw attempts
local. Failed requests or output truncation invalidate a scored cell, rather than being dropped.

Compare paired utterances. English WER and Mandarin CER are primary; English CER and whitespace-token
Mandarin WER remain separate diagnostics. Noninferiority requires primary absolute degradation <=0.002,
English CER degradation <=0.001, and a paired 10,000-resample bootstrap 95% upper degradation bound <=0.005
(fixed seed 20261106). Report raw edit totals and confidence limits, even when aggregate scores improve.
These are finite downstream admission thresholds, not general or all-language accuracy qualification.

## Determinism and request lifecycle

Throughput must give byte-identical per-row teacher-forced logits and generated tokens/text for the same
input, settings and seed across logical B=1/2/4/8, permutations and slot reuse. Different-context rows,
early EOS, active/queued cancellation, parent/child cancellation, ordered joins and next-slot recovery are
mandatory. Use greedy and temperature0.7/top_k50/top_p0.9/min_p0, seed314159; sample a predetermined two
utterances from each of the 16 speakers per language for seeded lifecycle probes. Repeat at least three
schedules. Fresh-process reproduction uses the same qualified algorithm identity. Any batch-dependent
change must be located and fixed, never dismissed as sampling noise.

Do not compare same-path determinism, old-path byte identity and recognition noninferiority as one claim.
Shared scratch, workspace, staging, metadata and outputs have one owning model/stream and remain live until
commands finish. Dummy rows never enter KV/slot maps or sampling. Poisoned padding and guards verify this.

## Performance, ablations and default decision

Use sufficient warmup and repeated measurements, with interleaved A/B/A cells and no task compiler or other
GPU job during sampling. Preserve all failed runs and failed warmups. Baseline and candidate use identical
checkpoint, requests, sampler, context, cache policy and actual output-work accounting.
Required logical levels are 1/2/4/8, natural stopping and equal output cap, unique and repeated inputs,
continuous real long audio, repeated600 regression, mixed long/short traffic, and sustained load long enough
to observe throughput/tails (at least120 seconds and64 completions per principal loaded cell).

Keep ablations separate: stable baseline; cross-row GEMM only (all five classes including head); GEMM plus
qualified residual/RMSNorm fusion; and scheduling changes, if qualified. Later CPU feature, encoder,
prefill or graph work requires its own profile-driven evidence and ablation, not one combined speed number.
Report makespan, HTTP P50/P95, available server TTFT with its clock boundary, real tokens/s, successful
requests/s, audio/wall, batch/padding distribution, queue/admission waits, per-stage event intervals,
selected-process CPU, sampled RSS, device/pinned allocation peaks, and traced CUDA activity unions.
Missing metrics stay null. Per-request shared-batch event times are not additive GPU busy. Allocation
peaks are not physical residency. Equal-token capped transcription is not complete-ASR real-time factor.

Reference vLLM0.29.0 uses the actual immutable image/model/configuration. Verify prefix, processor and
encoder reuse together, and actual compile/graph settings. Image-to-image identity and official Git source
comparison remain different evidence. Keep cache-off unique, repeated cache-on and deployment-config
cells separate. Profile actual memory/compute behavior; kernel names or activity alone do not establish
Tensor Core efficiency or shared weight traffic.

A throughput path is qualified only after numerical, quality, determinism, lifecycle and complete gates
pass and it demonstrates a statistically supported improvement in whole-model B4/B8 equal-work throughput
including LM head. Default promotion additionally requires a 95% lower improvement bound >=10% in those
loaded cells, no >5% B1 P95 regression, and no >10% mixed short-request P95 regression. Failure keeps stable
default and triggers further diagnosis; an unqualified experimental switch is not this task's performance
completion. No speed multiple is promised in advance.

Long-audio admission is evaluated only after effective GEMM. Consider more useful idle batches, ordinary
request headroom, fair parent turns and prompt refill as chunks finish. A smaller quota only stops new
admissions; it cannot release already occupied KV. Measure actual mixed-load waiting and cancellation.

## Evidence ownership

TensorFold owns implementation, tools, this contract and public sanitized receipts. xd-infra owns host
identity, current-source preflight, exact service snapshots, borrow/recovery and operational evidence.
Preserve historical results. Two fresh independent Standards/Spec review rounds cover implementation,
raw evidence, claims and limits; subsequent findings remain open until fixed and rechecked. Restore all
borrowed instances and independently audit IDs/images/state/health/restart, task ports/processes/containers.
The final clean source HEAD receives a new environment preflight and tested-binary/input/hash binding.
## Candidate updates and frozen workloads

The original P8/P32 cuBLASLt candidates are retained as rejected experiments: their model logits exceed
the preregistered relative-L2 guard. Their mathematical matmul checks, batch invariance and recognition
quality are separate results; none overrides a failed model gate. The eight-position prefetch experiment
also remains a negative performance result. The acceptance thresholds above are unchanged.

The replacement candidate shares weights across up to eight logical rows inside one CUDA block. Each
row keeps sixteen ordered FP32 accumulation chains followed by the same halving reduction, with BF16
rounding only at the original output node. Inactive rows have no input reads, output writes, KV or RNG
state. This is a fixed row capacity with guards, rather than a physically padded cuBLASLt matrix. Both
the original single-row vendor oracle and an independent FP64 gate remain registered. The stable mode
remains the default until all applicable promotion gates have evidence.

The bilingual quality manifest was frozen before recognition: SHA-256
`0d8a427d232e76df2dbfa0d6bdc314799db47d4006050b1d0fe7bbf75f5a552a`, 256 utterances and 32 speakers,
1753.46675 seconds. No quality utterances or scoring thresholds are replaced after seeing results.

The sustained unique-input pool has 2048 recordings per language from the same pinned source assets.
Filter durations to 2..20 seconds, sort IDs, select 2048 nearest-even rank quantiles per language, then
shuffle with seed 20261106. Its manifest SHA-256 is
`0fee55d4af8280da27b953018c77aa956a547bba9a2e762df8e194b1a593f377`: 4096 unique waveform SHAs and
24335.7471875 audio seconds. A unique-input cell must fail as incomplete if it exhausts this pool.

Principal sustained cells are cache-off, unique inputs, B=4/8, natural stopping and 128 output tokens,
each at least 120 seconds and 64 completions. Native stable/throughput comparisons use interleaved
repetitions; other cache and batch cells retain at least 64 completions and repeated measurements.
Equal-token diagnostics never report complete-ASR audio throughput. Whole-request latency includes HTTP
transport; internal TTFT and CUDA-event stages retain their own clock boundaries.

## Runtime and diagnostic entry points

The process selects `TENSORFOLD_QWEN3_ASR_MATH=stable|throughput` when loading the model. `stable` is the
default. `throughput` uses the shared-weight CUDA kernel for packed QKV, attention output, packed gate/up,
down and LM head, including the first LM head after prefill. Prefill and the encoder keep their original
arithmetic. Runtime qualification is limited to SM121 and cuBLASLt version 130100; unsupported throughput
configurations fail explicitly. Selecting `stable` remains the explicit recovery path.

The shared kernel has a capacity of eight rows, with no physical padding or persistent workspace. One
block owns four output columns and reuses each loaded weight across its active row accumulators. It uses
SIMT FP32 instructions, not Tensor Core GEMM. Each output retains the original sixteen FP32 FMA chains,
halving reduction and final BF16 rounding. The vendor P32 policy remains available to diagnostics only;
it is not an alternative served math mode and did not pass the model-logit contract.

With eight engine lanes, long parents share four child admissions. A single parent may use all four;
additional parents receive fair turns as occupied work completes. The ordinary four-lane headroom stays
available. Shrinking a share does not release an existing child or its KV slot. Parent results retain
original chunk order. `audio_parents_cancelled_total` increments after a disconnected parent's workers
have joined; the parent gauge and engine drain checks independently establish release.

The existing GPU lock and service recovery protocol apply to these commands:

```sh
zig build native audio tf-qwen3-asr-check tf-qwen3-asr-gemm-bench tf-qwen3-asr-contract \
  -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j4
zig-out/bin/tf-qwen3-asr-gemm-bench "$MODEL" --shared
zig-out/bin/tf-qwen3-asr-contract
zig-out/bin/tf-cuda-test kernel-contract
TENSORFOLD_QWEN3_ASR_MATH=throughput zig-out/native/bin/tensorfold-native serve "$MODEL" \
  --name Qwen/Qwen3-ASR-1.7B --parallel 8 --context 8192 --prompt-cache-gib 0 --no-thinking
```

`prepare_qwen3_asr_eval.py` and `prepare_qwen3_asr_perf.py` reconstruct the frozen workloads.
`bench_qwen3_asr_accuracy.py --require-natural-stop` and `compare_qwen3_asr_quality.py` require exact
coverage of all 256 quality utterances. `check_qwen3_asr_determinism.py` compares greedy and fixed-seed
sampling across concurrency, order and reused slots; `--reference` checks another process's fingerprints.
`check_qwen3_asr_lifecycle.py` requires observed active/queued work and actual cancellation-counter changes.
`check_qwen3_asr_parents.py` checks quota shrink, continued four-slot decode, queued/active parent cancellation
and ordered completion of real and repeated long audio. Both long fixtures must have at least four chunks.

For performance, use `bench_qwen3_asr_perf.py --clip-manifest MANIFEST --unique-requests --seconds 120
--min-requests 64 --warmup-requests 32` with an explicit identity and cache condition. Native transcription
qualification additionally sets `--require-natural-stop`; absent reference stopping telemetry stays unknown.
`compare_qwen3_asr_perf.py` binds 64 unique input SHAs, actual token IDs/counts, seed, context and cache
condition. Native token disagreements fail; cross-engine disagreements are retained separately from ASR
quality. `bench_qwen3_asr_mixed.py` requires both natural chunk stopping and observed execution overlap.
`summarize_qwen3_asr_ncu.py` and `summarize_qwen3_asr_trace.py` export the public subsets of private traces.
