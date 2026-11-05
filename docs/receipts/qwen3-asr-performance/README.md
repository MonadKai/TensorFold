# Qwen3-ASR CUDA performance and concurrency follow-up

The implementation batches decoder execution and sampling, packs decode projections, and runs bounded
long-audio children through LaneHost. It preserves the original native BF16 stage/token results on the
checked English/Chinese cases. The measured speedup is modest; this branch has not reached vLLM throughput.

Baseline: `126765048`. Ordinary full-matrix candidate: `dc2b72985`; final runtime fix: `f8066cdaa`.
The final fix copies child profiling JSON into parent-owned storage. Model/CUDA operator sources are
unchanged since `3ce0d3319`, which restored original separate prefill projections after a rounding regression.
Checkpoint: original `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`.
Platform: Linux AArch64, GB10 sm_121, CUDA 13.0.2 / nvcc 13.0.88, driver 580.95.05, 128 GB unified memory.
This is downstream development validation, not an upstream landing gate or production capacity promise.

## What changed

- One 28-layer decoder invocation consumes up to eight active rows with independent slot, position, KV,
  sampling rules, seed and logical token position. One batch sample/readback replaces per-request waits.
- Loader-owned QKV and gate/up buffers reduce decoder main projection calls from seven to four per layer.
  Packed views apply to decode only; prefill retains the original projections and BF16 rounding.
- Strided vendor batches preserve the original single-row algorithm. Norms, attention, projections and
  sampling consume batches, but this does not establish cross-row weight reuse or ordinary B-row GEMM speed.
- Long transcriptions have at most two workers per parent and a server-wide FIFO quota of
  `max(lanes / 2, 1)` children. Results retain original order, splitting, language/context and seed rules.
  Disconnect cancels queued/active children and joins workers before releasing shared input.
- Prefill respects positive row budgets, capped at 256; encoder remains one complete initial operation.
- Optional phase timing, explicit runtime/cache identities, makespan/work/CPU/memory/cache counters,
  labelled CER/WER and CUDA trace summaries make the next optimization attributable.

## Unprofiled results

All cells use BF16, context 8,192, eight lanes/sequences, temperature zero and the same checkpoint.
Warmups are retained and excluded. Task model/GPU jobs are serialized; no task compiler overlaps a measured
cell. Other retained host services may remain resident, so this is not an exclusive-host capacity test. The native path
has no feature, encoder or prefix cache and no graphs. vLLM v0.29.0 uses the verified pinned reference image.

| Workload | Original native seconds | Optimized native seconds | Actual generated tokens | Makespan reduction |
| --- | ---: | ---: | ---: | ---: |
| Six distinct short/medium clips, concurrency 8 | 13.271 | 12.487 | 555 both | 5.9% |
| Eight labelled clips, two repetitions, concurrency 8 | 12.941 | 11.830 | 560 both | 8.6% |
| Eight distinct 120-second clips, 128-token cap | 26.363 | 24.103 | 1,024 both | 8.6% |
| Natural AMI meeting, 600 seconds | 38.316 | 34.524 | 1,564 both | 9.9% |
| Repeated English regression, 600 seconds | 41.986 | 37.978 | 1,724 both | 9.5% |

The first optimized row is the final `f8066cdaa` smoke; the others use the full-matrix candidate.
All same-engine compared token arrays, or long-transcription text and token counts, match. See
[performance.json](performance.json) for every executed repeat and concurrency 1/4/8, not just selected rows.
Six-clip concurrency 8 has at most six active requests; the separate eight-clip workloads exercise all eight.
The fixed 128-token cells are **partial-ASR diagnostics**, not complete-transcription real-time factors.

The final six-clip cell completes 204.255 audio seconds in 12.487 wall seconds: 16.357 audio seconds/wall
second, 0.480 successful requests/second and 44.445 generated tokens/second. HTTP P50/P95 are 5.545/10.840
seconds. These small heterogeneous batches do not estimate a sustained-load SLA. This is the combined change,
not an ablation assigning separate speedups to projection packing, batching and chunk concurrency. Selected-process CPU is
1.10 seconds; primary PID sampled peak RSS is 731,709,440 bytes, not aggregate physical memory.

| vLLM configuration | Six distinct clips c8 seconds / tokens | Fixed 8 x 128 tokens seconds | Natural 600s seconds / tokens |
| --- | ---: | ---: | ---: |
| Cache off, eager, compile/graphs off | 7.878 / 549 | 4.568 | 5.388 / 1,562 |
| Repeated input, prefix + processor cache, eager | 7.656 / 551 | 4.111 | 5.085 / 1,562 |
| Local deployment config: caches + default compile/graphs | 7.239 / 549 | 3.859 | 4.702 / 1,564 |

Cache-off means cross-request audio, encoder embedding and prefix reuse are disabled; ordinary per-request
KV and vendor plan/allocator caches remain. The pinned [renderer policy](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/renderers/base.py)
assigns request-unique multimodal identifiers when both prefix caching is off and processor cache is zero.
The encoder manager therefore cannot reuse an earlier request's embeddings, even after identical warmups.
An independent [exact-image source trace](cache-semantics.json) confirms the same policy in the actual
reference image and follows UUIDs through chat/STT rendering, hashes and encoder-cache identifiers.
The old 2,512-file comparison is between images; it is not a whole-Git-tree identity proof.

The last two rows reuse warmed content; [startup configuration receipts](reference-configurations.json)
confirm actual compile/graph settings. Cache-on is not
an operator-only comparison: six-clip eager measurement records 2,720/2,746 prefix token hits and 6/6
processor hits. [vLLM metric definitions](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/metrics/stats.py)
count prefix queries in tokens and processor queries in multimodal items. Cache-off counters remain zero. Native lacks these caches, so its corresponding configuration
is unchanged. The equal-token native cell remains 5.28x slower than cache-off/eager vLLM (6.25x against
cached/compiled vLLM). Natural stopping produces different cross-engine output work; report those counts
rather than attributing their entire wall-time difference to kernels. Native long transcription admits two
children per parent, while the reference can fill eight sequence slots with chunks; this scheduling-policy
difference is part of the endpoint comparison. This removes the confounding of the
historical repeated-audio cache-enabled/eager comparison without claiming identical cross-engine arithmetic.

## GPU attribution and phase timing

[Nsight baseline](profile-before.json) and [optimized trace](profile-after.json) contain actual CUDA activity
unions within named NVTX workload ranges. For six clips, kernel launches fall from 228,519 to 119,067;
captured CUDA activity falls only from 12.741 to 12.425 seconds. The dominant vendor GEMV family still takes
11.445 summed kernel seconds after the change (about 92% of captured kernel activity). Thus launch batching
works, but row-stable projection arithmetic still dominates; cross-row GEMM is the next measured seam.
For natural 600-second audio, activity is 35.159 versus 34.617 seconds while the range wall interval is
38.912 versus 35.946 seconds. Subcategory activity unions can overlap and must not be added.

Full-trace allocation peaks are 12,904,104,704 versus 12,910,123,520 device bytes and 11,571,200 pinned bytes
for both native versions, plus 136 device-static bytes. These cover startup/warmup and do not describe
physical memory residency. The trace reports zero duplicate allocation and unknown-free events.
Raw Nsight reports/SQLite stay local because they can contain private environment and paths.
The [cache-off reference trace](profile-vllm-disabled.json) measures 7.456 CUDA-active seconds for six clips,
4.784 for natural 600-second audio, and 1.329 for the fixed 1,024-token diagnostic (range walls
7.983 / 5.578 / 4.599 seconds). Its captured full-trace peaks are 18,387,970,052 device bytes,
9,704,087 pinned bytes and 720,599 device-static bytes, with zero duplicate/unknown-free events.
Startup, autotuning and warmups are included in allocation peaks, but excluded from named hot workload
ranges. Other reference cache/graph configurations have no individual activity/allocation trace in this run,
so absent ordinary-cell counters stay null. These traced ranges are not substituted into the unprofiled
speed table, and allocation peaks are not physical residency or a memory-efficiency ranking.

[Phase timings](phase-timings.json) retain the final real chat/65-second transcription smoke, including CPU
Log-Mel, body/multipart, admission, lane queue, encoder, prefill, decode, head, sampling and postprocessing.
This cold diagnostic includes setup costs; ordinary speed comparisons exclude profiling. CUDA events are
stream elapsed intervals, not device active time. Each participant is charged the whole shared batch:
never sum request intervals to obtain busy time. Body timing starts after headers/buffered bytes; chat
Base64 and JSON parsing are not independently instrumented.

To reproduce a trace summary with the checked Nsight version:

```sh
nsys export --type sqlite --output run.sqlite run.nsys-rep
python3 tools/summarize_qwen3_asr_trace.py run.sqlite summary.json
```

## Recognition, API and concurrency

[Numerical checks](numerical.json): 186 native stage records are byte-identical; all 1,470 teacher-forced
logits rows across batches 2 through 8 match separate single-row original projections. Against the masked
original Qwen reference, English 33/33 and Chinese 11/11 teacher-forced top-1 decisions agree. This is a
finite regression scope, not numerical equivalence for all prompts or seeds.

[Labelled accuracy](accuracy.json): original native, optimized native and cache-off/eager vLLM checked at
concurrency 1 and 8 give
WER 0.025 and CER 0.008859 on eight English clips (86.345 seconds). This tiny single-speaker smoke does not
qualify general ASR, other languages, or production recognition quality.

[API differential](api-comparison.json): 49 main cases plus one separately executed 48 kHz stereo codec
supplement. Both return valid JSON and matching duration/text in the supplement. Five status differences
are explicit scope/boundary refusals: streaming chat/transcription, multi-turn history, chat penalties, and
exactly 200 decoded samples. Nonzero-temperature text differs across engines; this is retained separately.
The reference supplement uses production caches/graphs and native profiling; it is API evidence only.

[Concurrency](concurrency.json) records greedy equality outcomes, 18 seeded sampled requests, cancellation/slot reuse,
real queued-parent observations, sibling order/recovery and a four-long-parent/short-request mixture.
Before the global quota the short reply took 22.70 seconds; after it takes 5.14 (solo 1.14). All queued and
active child/request counters drain to zero. This establishes bounded admission and recovery, not a latency
SLO or complete isolation: active encoder/prefill work can still delay short requests.

The [two-round review record](reviews.md) separates Standards/Spec findings and subsequent runtime fixes.
Raw token arrays remain local; their files are pinned by hashes.

## Complete backend and host qualification

[qualification.json](qualification.json) records the unfiltered `tf-cuda-test kernel-contract` result:
**560/560** kernels pass gates 1–4, **361/361** controls turn red as planted, zero missed controls and **zero
resource-limit problems**. Six pre-existing nonfinite findings remain explicit in the contract; this is not
a claim that every backend behavior is defect-free. The former 190 mixed static/live record problems are
closed by this fresh complete run. All five SM targets were recompiled with nvcc 13.0.88; 57 launches / 226
static SM records were refreshed. The offline report contains 1,480 launch records (including two controls)
and 4,867 applicable SM checks: 4,857 noncontrol passes and 10 control reds. These are different counts from
that historical 190. Other SMs have compile/resource evidence only; device/model execution is GB10 sm_121.

Nonempty old bases are retained for later compiler selections before independent 13.2/13.3 overrides;
missing later receipts remain pending. No measurements for those later compilers are fabricated.
Current production-source host suite exits zero, native/server checks pass 950/950 tests, default build
passes 30/30 steps, and lean remains 424 inherited / zero new. The final metadata/receipt commit changes no
production build inputs from the tested `f8066cdaa`. Current clean-source host-owned preflight and borrowed
service recovery are recorded separately by the infrastructure owner.

## Corpus and attribution

[corpora.json](corpora.json) pins transformed WAV hashes and crop offsets. AMI meeting ES2002a Mix-Headset
is credited to the [AMI project](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml), released under Creative
Commons Attribution. Crops are mono PCM16/16 kHz. Natural meeting audio and repeated synthetic regression
are separate. The eight distinct 120-second crops cap output to equal work.

Accuracy uses `hf-internal-testing/librispeech_asr_dummy@5be91486e11a2d616f4ec5db8d3fd248585ac07a`,
clean/validation. HEAD was checked against the pin at acquisition; the official rows API supplied assets,
and hashes pin the normalized files actually tested. Credit Vassil Panayotov, Daniel Povey and the
[LibriSpeech corpus](https://www.openslr.org/12/) (CC BY 4.0). Raw audio, labelled text and signed download URLs
stay local. Published Qwen English/Chinese examples supply the short original regression clips.

## Remaining work

Cross-row Tensor Core projection arithmetic needs a measured BF16 acceptance policy before replacing the
row-stable path. Prefill fusion, batched encoder convolution/attention, reusable FFT plans, multi-request
prefill, encoder budgets, graph capture, layered caches, paged KV, quantization and drafters remain separate
work. Seed results are invariant to batch composition within the checked original native arithmetic path;
that is not a claim of cross-engine sample identity. No streaming, timestamps, diarization, translation,
TP or release archive/upstream landing gate is added. Fit on this unified-memory device does not qualify a
smaller memory class. Cold-start and sustained high-load capacity still need dedicated measurement.
