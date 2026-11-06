# Measurement and receipt boundaries

The [contract](../../recipes/qwen3-asr-row-gemm.md) fixes admission thresholds and workload selection.
An execution-complete marker is not a substitute for inspecting the associated numerical, quality,
performance or restoration result. The README states the current qualification status.
The independent publication plan requires 94 native, 58 reference and eight instrumented cells.
Every required cell must be complete. Additional failed attempts, if any, remain explicitly incomplete
and are retained outside the admitted matrix; whole cells may not disappear with their archive records.

| Measurement | Boundary and interpretation |
|---|---|
| HTTP request latency | Immediately before `urllib.urlopen` through reading the complete response body. Local file reading, encoding, payload construction and response JSON decoding are outside this interval. The API is non-streaming. |
| Batch makespan | Before creating/submitting the measured worker pool through collecting all measured results. Includes client file reading, encoding, response decoding and executor shutdown. Excludes warmups and the surrounding metrics calls. |
| Native chat TTFT | `chat.prepare` entry to the first generation callback committing a token. Includes prompt/audio preparation performed inside that call, lane waiting, encoder/prefill and the first sample. Excludes prior HTTP body reading and request parsing. It is not network first-byte latency. |
| Native transcription chunk TTFT | The same internal clock, after the chunk's admission and `audio.prepare`. Parent admission and audio preparation are reported separately. Chunk TTFT is not parent-request TTFT. |
| Reference TTFT | The first-output iteration timestamp minus `EngineCoreRequest.arrival_time`, which can be supplied by rendered input or initialized by the input processor. Uses the reference wall clock, unlike native's awake clock. Histogram quantiles are bucket intervals without interpolation and are not network TTFT. |
| Request P50/P95 | Linear interpolation at sorted index `(n-1)*q`, using the same benchmark implementation for both paths. |
| CPU audio stages | Elapsed intervals on the preprocessing side, including blocking decoder work where applicable; these are not process CPU-seconds. |
| Process CPU | `/proc` CPU-time deltas for explicitly selected PIDs. It does not include every host process or every short-lived decoder child. |
| Sampled RSS | The primary selected process, sampled every 100 ms. Not aggregate physical memory or an exact allocation peak. |
| Device/pinned peaks | Available allocation counters, including startup and warmup. Native counters track application allocations and do not establish complete driver allocation or physical residency. Missing counters remain null. |
| GPU stage events | Elapsed stream intervals for encoder, prefill, decoder, LM head and sampling. Decoder timing excludes the separately measured head. Each participant receives the whole shared-batch interval: never sum request charges as GPU busy. |
| Nsight activity | Union of captured kernel/copy/memset intervals within each named workload range. Subcategory unions can overlap. The range also includes client work; profiling overhead is present. |

`natural` uses a 1024-token safety cap and requires observed stopping in the native/chat gates.
Primary `equal_tokens` cells ignore EOS and generate exactly 128 tokens; the eight instrumented
`asr-*.json` diagnostics instead use their recorded 64-token cap. Their input-audio/wall ratio is deliberately
null: partial or continued text is not complete-ASR real-time performance. Reference transcription does
not expose native per-chunk stopping fields; available finished-request counter deltas and this missing
observation are retained separately. A failed request or failed warmup makes a measured cell incomplete.

Client concurrency and actual decode rows are separate fields. Split transcription children share a
four-slot quota at eight lanes; single-chunk transcriptions enter the engine directly. The chat matrix
explicitly requires observed target rows. Batch histograms report the
actual rows, physical rows and padding. The fixed row-capacity label describes the projection algorithm;
it does not add active requests or reserved KV slots.

The primary whole-model comparison pairs the same frozen 64 input waveforms and validates sampler,
context, model/cache identity and actual output work before comparing makespan. Native pairs additionally
require identical text and token identities. Cross-engine natural outputs can differ; those differences
remain in the pair reports. Sustained cells use the same frozen unique pool, but their completed prefixes
and counts can differ. Their token rates do not replace the fixed-work attribution.

The loaded fixed-work summary requires three complete interleaved stable/throughput trial pairs; smaller
batch comparisons require two. The derived summary resamples complete paired-trial gains 10,000 times with
seed 20261106. `lower95` and `upper95` are one-sided percentile bounds at the fifth and ninety-fifth
percentiles, not a central 95% interval. This finite workload/trial evidence is not a capacity guarantee.
Quality uses its separately frozen utterance-level paired bootstrap and normalization contract.

`production` names a measured deployment-configuration condition for the temporary benchmark processes.
It does not establish a production deployment, capacity or SLO. Requested flags, observed server
configuration, startup compile/graph activity and available cache counters remain separate evidence.

Nsight Compute 2025.3.1 samples B8, the first decoder layer and the LM head with clock/cache control set
to `none`. Each reported projection trace contains one selected kernel. Cache state is not forced cold;
replay affects duration. L2 traffic includes activation/output traffic as well as weights. Its measured
bytes divided by instrumented duration is L2 effective throughput, not DRAM bandwidth. Ordinary microbench
timings rotate decoder layers and use stable-before/shared/stable-after order. The selected projection
kernel uses SIMT and has zero Tensor Core activity in these traces; this statement does not describe the
unchanged encoder/prefill kernels. Whole-service gains use the separate unprofiled cells.

Public aggregate performance metadata indexes every measured attempt and excluded warmup in a
deterministic gzip NDJSON archive, with record counts and SHA-256. Each line identifies its cell and phase.
Text, work counts, status, timing and token fingerprints are retained. For these performance records,
full token arrays stay with the private originals; dedicated determinism receipts retain their own
token evidence. Large trace databases remain private. The public inventory hashes compressed files
as well as JSON and Markdown. It is regenerated only after publication content is complete.

Reference attempt 3 adds an optional client-side compiler observer: warmup is followed by five seconds without observed Zig/nvcc/cicc activity, then the client samples these process names approximately once per second during measurement. Any observed activity or observer error invalidates the cell. This observer adds client overhead to reference measurements; earlier native A/B trials used their original observer and a precompiled, non-JIT model. Sampling does not establish continuous system exclusivity. Attempt 1 failed at CUDA memory initialization; attempt 2 reached the server but lacked compiler-quiescence proof for two long-audio diagnostics. Both attempts remain incomplete and excluded from admitted comparisons. The original compiler snapshot retained PID/name only, so its process ownership is unconfirmed.

The compiler process scope was narrowed to the reference container after cgroup evidence identified a separate CPU-only build. Foreign CPU compiler activity remains recorded and can contend for shared CPU/unified-memory resources. Additional native B4/B8 controls bracket the reference matrix under these evolving host conditions; cross-engine timing retains that limitation. Task-owned Zig/CUDA compilation and unrelated GPU compute remain excluded.

Long-reference completion verification is explicitly post-capture: it cross-checks source-derived chunk-count bounds against all five engine finish counters. Verified audio throughput is audio seconds divided by measured batch makespan (`measurements.batch_wall_seconds`), including the client work defined above; real-time factor (RTF) is the reciprocal, batch makespan divided by audio. The derived result preserves original null fields and is separate from preregistered performance gates and recognition quality.
