# Qwen3-ASR shared-row CUDA evidence

GPU validation, clean-source candidate binding and independent restoration are complete. Final review closure is pending.

The validated candidate computes `X[B,K] × W[N,K]^T` in one shared-weight CUDA launch for QKV,
attention output, gate/up, down and LM head, including the initial LM head. Up to eight active rows
reuse each weight load. Sixteen ordered FP32 accumulation chains, reduction order and BF16 output
rounding preserve the stable oracle. This is SIMT; physical rows equal logical rows, with zero padding.
Qualification is limited to ARM64/SM121 with cuBLASLt130100 and this BF16 checkpoint. Other compiled
SM images do not establish inference qualification. The stable math path remains the default.

```sh
TENSORFOLD_QWEN3_ASR_MATH=throughput tensorfold-native serve MODEL \
  --name Qwen/Qwen3-ASR-1.7B --parallel 8 --context 8192 --temperature 0 \
  --no-thinking --prompt-cache-gib 0
```

The [frozen contract](../../recipes/qwen3-asr-row-gemm.md) predates algorithm experiments. Original
precision, quality and determinism thresholds are retained. Unsupported runtimes fail explicitly.

The long-audio policy grants an idle parent up to four child permits at eight lanes, keeps the global
four-child limit, and redistributes turns between parents. Ordinary requests retain four lanes.
Shrinking a quota does not release already occupied KV slots. Child results join in original order;
parent cancellation waits for active and queued children to drain. Single-chunk transcription bypasses
this parent-child gate and enters the engine directly.

The canonical native binary is bound to serving-source input `1755bba98157a35b4a572f2fc10fbb0faaee81cb`.
Later changes concern diagnostic tools, documentation, the standalone GEMM benchmark and host-only
refusal tests imported solely from a test block. The complete CUDA runner has a separate input identity;
its tested fatbin is unchanged. Final clean-source input
and artifact comparisons are recorded separately, without relabelling historical captures.

Numerical checks include 186 exact native stages and 44 teacher-forced logit rows. Independent FP64
projection references cover 20 fixtures, 2,586,240 output elements and 689,664 fixture-column maxima.
All outputs match the stable path bit-for-bit and satisfy both frozen error bounds; the largest
observed fraction of the tighter envelope is 0.772404. Seeded synthetic activations and actual first-layer/
LM-head weights exercise all five shapes at B1/2/4/8. This is separate from model-level quality.

The frozen bilingual corpus contains 128 English and 128 Mandarin utterances from 32 speakers.
Stable client concurrency eight and throughput concurrency one/eight preserve all texts and token
fingerprints. English WER is 69/2875 (2.4%); Mandarin CER is 36/1861 (1.9344%). The paired bootstrap
non-inferiority differences are zero. Mandarin whitespace-word WER is diagnostic only.

The refreshed vLLM corpus completes all 256 requests and an isolated exact-count proof covers all 261
internal model requests finishing with stop. English WER is also 69/2875 (2.4%); Mandarin CER is
35/1861 (1.8807%). Native and reference transcripts differ; equal corpus-level WER is not text equality.
Fresh original-Qwen teacher diagnostics retain 44/44 top-1 decisions. Native versus original-source
logits are not numerically equivalent: worst relative L2 is 0.056923 (English) / 0.041188 (Mandarin),
with maximum absolute differences 1.046875 / 0.890625. Stable and shared native paths show the same
reference differences and match each other exactly. These are reported diagnostics, not a newly
passed cross-framework numerical-equivalence gate.

The 1,664-response transcription matrix uses client concurrency 1/2/4/8, three orderings, greedy/seeded
sampling, slot reuse and another process. Client labels alone do not prove actual GPU batch sizes.
Two additional chat matrices each complete 1,664 responses with zero fingerprint differences. Every one of the 24 greedy/seeded, three-order, B1/2/4/8 cells per process observes positive actual logical-row counters. The two process identities differ, and the second also matches the first process's per-input baselines. Context lengths, row order and reused slots remain independent of per-request RNG state. These 128-token identity diagnostics are separate from complete-transcription quality.
Another 384 capped chat probes observe their own decode before and after neighboring cancellations,
with zero seed/output differences. The capped probes do not measure complete-ASR quality.

The fresh 50-case API comparison reports 45 matching HTTP statuses and five documented existing scope differences, with zero unexpected status regressions. Transcription remains greedy with server temperature 0.8; completion-length precedence is checked at eight tokens. The existing non-streaming contract and refusal boundaries are retained. Active/queued cancellation, early termination, parent-child cancellation, ordered joins, occupied-slot quota shrink, slot recovery and concurrent request isolation pass their strengthened lifecycle checks.

The unfiltered CUDA backend completed 564/564 cases, 361/361 expected control failures, zero resource
limit failures and six retained known findings. These known findings are separate from new failures.
Compiler/SM resource records retain other compiler bases, pending entries and independent overrides.

Native matched-work results (current frozen binary; 64 unique inputs per trial):

| Active clients | Natural makespan ratio | Equal-128-token makespan ratio | Trials per workload |
|---|---:|---:|---:|
| 1 | 0.996× | 1.001× | 2 |
| 2 | 1.638× | 1.799× | 2 |
| 4 | 2.617× | 3.152× | 3 |
| 8 | 3.725× | 5.229× | 3 |

Ratios are stable divided by shared-row makespan. Native texts, ordered token IDs and actual token counts match in all twenty pairs. Equal-token requests are capped diagnostics, not complete-ASR rates. B1 natural P95 regressed by at most 2.12% across its two trials.

Projection-only CUDA-event ratios (stable/shared; seven samples per leg, stable/shared/stable):

| Projection | B1 | B2 | B4 | B8 |
|---|---:|---:|---:|---:|
| qkv | 1.048× | 1.103× | 1.238× | 1.558× |
| attention_output | 0.919× | 1.004× | 1.012× | 1.104× |
| gate_up | 1.044× | 1.960× | 3.646× | 6.127× |
| down | 0.878× | 1.049× | 1.266× | 1.579× |
| lm_head | 1.028× | 2.006× | 3.705× | 6.733× |

The four decoder projections rotate 28 actual layers; LM head uses its single actual weight matrix. Microbenchmarks exclude attention, normalization, preprocessing, sampling and HTTP. NCU counter-pass durations are not substituted for these timings. SIMT source and lower observed L2 traffic support cross-row weight sharing; no Tensor Core utilization is claimed.

Cross-engine makespan ratios (native shared-row / vLLM; larger means native takes longer; two trials):

| Clients | Unique/cache-off natural | Unique/cache-off equal128 | Repeated/caches-on | Measured deployment configurations |
|---|---:|---:|---:|---:|
| 1 | 1.008× | 1.014× | 1.053× | 1.099× |
| 2 | 1.228× | 1.298× | 1.341× | 1.375× |
| 4 | 1.378× | 1.389× | 1.553× | 1.657× |
| 8 | 1.554× | 1.621× | 1.956× | 2.083× |

The unique natural pairs contain native/reference token totals 1,392 versus 1,386–1,389 and reported
text/token differences. Equal128 pairs have 8,192 tokens per engine but different text/IDs. Repeated
conditions each contain 64 measured requests over four waveforms and 1,248 output tokens per engine;
their counts match without implying cross-engine text equality. Native has no cross-request content
cache under any label. The reference caches-on condition uses prefix caching and a 4 GiB processor
cache in eager mode; its measured deployment condition uses VLLM_COMPILE and FULL_AND_PIECEWISE
graphs. Cache-off uses prefix off, processor cache zero and eager NONE/NONE. These are distinct
workloads/configurations, so their columns do not isolate an optimization.

Eight cache-control pairs hold the repeated inputs, token work and outputs constant within vLLM.
Mean cache-off/cache-on makespan ratios at B1/2/4/8 are 1.033×/1.127×/1.161×/1.223×.
This measures the joint configuration, not separate prefix, processor or encoder-cache savings.
All 168 admitted cells complete: 94 primary native, 58 primary reference, eight diagnostic profiles
and eight supplemental drift controls. The public archive also preserves two excluded long-audio
timing cells. The gzip performance-request archive contains all 43,113 measured/warmup records from
those 170 timing receipts; the separate failed-admission JSON retains one additional warmup.
Forty formal pair reports cover twenty native exact pairs, sixteen cross-engine difference reports
and four native before/after drift controls. Native B4/B8 controls bracketing reference work change
makespan by between -1.21% and +0.23%; this is not proof of identical background CPU load.

The scheduler ablation completes 21 cells and fifteen strict ordered text/token/chunk-work
comparisons. With shared-row math fixed, changing only the old two-child policy to the fair policy
reduces real AMI600 makespan by a mean factor 1.694× and repeated600 by 1.703× (three trials each).
In the two-long-plus-64-short workload, the policy-only makespan ratio is 1.019×;
short-request P95 changes range from -0.802% to +0.338%. Measured old-math/old-policy versus
shared-row/fair-policy mixed makespan is 3.920× and short P95 falls about 76%.
The latter is an actual mixed-load comparison, not a product of independent speedup factors.
The old-math mixed baselines ran in a later recovered resource window; host-load/time differences
remain a limitation. Observed maximum queue wait is at most 43.452 ms for the fair mixed trials.
Waiting/cancellation and occupied-KV shrink are separately tested; finite trials do not establish
an unconditional waiting-time bound or production SLO.

Fresh Nsight Compute counters at B8 show lower L2 traffic for all five selected projections and zero
Tensor Core activity. L2 traffic includes activations and outputs, and is not a DRAM-bandwidth metric.
Counter collection perturbs latency, so counter-pass durations are excluded from speed claims.
Nsight Systems uses the union of GPU activity intervals rather than adding request-shared events.
Instrumented per-request stage times are not additive GPU busy time. Sampling's instrumented B8
interval increased even though its algorithm was unchanged; the complete-model results include it.

The ordinary vendor P8/P32 row GEMMs failed the frozen model-logit bound despite mathematical matmul
and some recognition checks passing. The exact eight-position prefetch variant was slower and removed.
An early manually built CUDA image omitted the repository precision flags and matched only 34/137 native stage
comparisons (103 differed); rebuilding with the documented glue flags restored exact results. All failed identities,
comparisons and negative timing results remain in the receipts.

Reference attempt 1 failed at CUDA memory initialization before model loading. Generic CUDA queries
and the actual MemorySnapshot probe later passed, and the same image/CLI subsequently started;
the first OOM cause remains unconfirmed. Attempt 2 reached the server but its two long-audio timings
lacked compiler-quiescence evidence and were excluded. Attempt 3 preserved a successful warmup then
failed an overly broad compiler admission check. Cgroup inspection identified a separate CPU-only
build. The final observer scopes the immutable reference container, retains foreign CPU observations,
and invalidates timings on observed task compiler activity or observation error. It samples at about
1Hz and does not prove continuous system exclusivity. Task host compilation occurs at explicit phase
boundaries. GPU process snapshots bracket timed cells, with retained OS contexts declared separately.

The eight supplemental native B4/B8 fixed-work controls bracket reference measurements to expose
background/time drift. They do not establish identical host load or normalize away cross-engine
uncertainty. Final actual runtime configurations and available cache counters are separate from CLI
intent. Prefix-off plus processor-cache zero uses request-scoped multimodal identities in the actual
image, preventing cross-request encoder identity reuse; encoder allocation still serves request lifecycle.
Eligible content identity reuse under caches-on does not quantify separate encoder-cache savings.

The actual reference image reports vLLM0.29.0 and Torch2.13.0+cu130. Its Python tree hash is distinct
from eight selected files matched to Git `98dff2a81d747d1dba01a47f939f48c3526d4206`; that comparison
does not establish whole-image/Git equivalence or a registry-manifest identity. The original Qwen
source has its own pinned archive and fourteen verified Python files. Its export harness's window-mask
fixture hook is documented; comparisons do not imply unmodified end-to-end execution equivalence.

Long-reference complete-audio throughput is an explicitly post-capture derivation: exact source-based
21-chunk bounds and all five finish counters verify stop completion. Original null HTTP fields remain
unchanged. Audio throughput is audio/wall; RTF is wall/audio. This is separate from recognition quality
and preregistered acceptance gates. Input manifests retain freeze-time metadata, not live run status.

Validation leases restore borrowed instances in finally handlers and an independent service post-stop fallback. Final restoration passes three independent probes under the shared GPU lock: all 26 original instance IDs, images, state, health and restart policies match, with zero task containers, listeners or GPU processes. A later coordinated borrowing window is preserved with its intermediate six-mismatch probe; final probes follow that owner's restoration. Host identities and full receipts belong in the infrastructure repository; clean-source preflight and both review rounds are recorded at closure. A tool-daemon interruption left one incomplete mixed-load attempt without recoverable in-memory partial request/warmup outcomes. Available logs/identity and this limitation are retained; that attempt is excluded and the missing canonical baseline was independently rerun.

Raw trace databases, model-weight exports and host/service identities stay outside the public tree.
[Methods](methods.md), the complete per-request gzip archive, fixed manifests, raw-receipt hashes and
[inventory](inventory.json) define timing boundaries and coverage. Device/pinned allocation peaks are
tracked allocations including startup/warmup, not physical residency. Missing metrics remain null.

Scope remains this checkpoint, BF16, one ARM64 GB10/SM121, cuBLASLt130100 and non-streaming APIs. Stable remains the compatibility default despite the measured throughput admission metrics passing. Other architectures/toolchains, quantization, speculative decoding, multiple GPUs, cross-request content caches, CUDA Graphs and broader language/accuracy/capacity qualification are unqualified or unchanged. No additional residual/RMSNorm fusion, staging/sampling or encoder arithmetic change is claimed; profile evidence and negative results remain available for follow-up. The high-throughput path improves native loaded work but remains slower than this vLLM configuration on the measured B2/B4/B8 cross-engine workloads.
