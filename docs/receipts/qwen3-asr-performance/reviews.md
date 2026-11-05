# Qwen3-ASR performance consistency reviews

The fixed baseline is `126765048`. Reviews separate repository Standards from the requested Spec.
GPU execution, performance and service recovery are separate evidence; a code review does not establish them.

## Round 1 — `7a124e74f`

### Standards

Two hard findings: a cancelled encoder left a pending profiling event, preventing safe timer reuse; the
performance client counted the audio duration of failed HTTP requests as completed work. No smell reported.

### Spec

Two findings: cancellation could leak or misattribute stage timing to a later slot user; failed requests
inflated the audio throughput, contrary to the required successful-work accounting.

### Fixes and reproduction

Discard unresolved timing on backend errors and release, while retaining same-stream event ordering.
The actual GPU cancellation probe first failed with `UnresolvedTiming`; after repair the reused slot returns
the reference token. A planted all-404 server first reported positive audio throughput with zero successes;
the repaired client returns failure and zero completed-audio throughput. Preserve both runs locally.
The parent additionally found that the one-row family was not admitted to shared rounds; explicit plain
stream capability now reaches real batched decoding without claiming an MTP head or wider draft support.

## Round 2 — `6a3b2d93f`

### Standards

One hard finding: refreshing nonempty nvcc 13.0 resource records also changed expectations inherited by
13.2/13.3 without their receipts. Preserve their previous base selection and pending precedence before
applying independent compiler overrides. One possible Duplicated Code smell: chunk and single-clip ASR
marker cleanup duplicated the same rule. Both now call `audio_input.clean`.

The independent reviewer checked `7dfd4c2e9` and closed both findings. A host regression compares every
original and refreshed record on all five SM targets for 13.2 and 13.3; their selections remain identical.

### Spec

One transport-evidence finding: a connection failure could escape the benchmark client before writing its
report. The first fix missed an interrupted HTTP error body; the follow-up nests that read inside the
transport guard. Failed requests retain status zero, duration, fixture hash and error details. Warmup attempts
are also retained, and a failed warmup marks the measurement incomplete and makes the command fail.

The independent reviewer verified ordinary URL failures, timeouts, interrupted HTTP 200/503 bodies, ordinary
HTTP 503, and a failed warmup followed by a successful measurement using no-network fixtures. All retain
results and fail appropriately; no open transport finding remains at `5ceffc3ec`.

### Numerical and tooling follow-up

Ordinary multi-row cuBLASLt changed BF16 logits and one long-audio chunk. Keeping the one-row algorithm in a
strided batch restores exact row arithmetic. The diagnostic compares batches 2 through 8, every row, over
32 teacher-forced positions: all 1,120 logits rows match the original separate single-row projections.
English stage exports additionally match the original native path byte for byte (137 records).
The new backend registration preserves replacement equality, rejects invalid alignment boundaries and uses
an exact FP32 stress whose reduced-precision alternatives miss the bound by 256x without lowering thresholds.
A host-test destructor call omitted its allocator; the compiler caught it and the call was corrected.

A Chinese stage comparison subsequently found changed prefill BF16 rounding despite unchanged top-1 choices.
Commit `3ce0d3319` keeps the original separate prefill projections and enables packed projections for decode
only. The changed Chinese outputs remain as a failed candidate, rather than becoming a new serial oracle.
Both languages were rerun: all 186 stage records and all 1,470 teacher-forced logits rows match their original native or separate-row oracle.

## Runtime follow-up — admission and telemetry ownership

The four-long-parent mixture exposed a tail regression: two children per parent together consumed eight
slots, delaying the short request from 1.14 to 22.70 seconds. A server-owned FIFO quota now admits at most
`max(lanes / 2, 1)` long-audio children globally. This is resource admission before feature extraction; all
inference remains in LaneHost. At eight lanes, the same mixture returns the short reply in 5.14 seconds,
with four admitted children, a bounded waiting queue, identical sibling/recovery text, and all counters zero
after draining. Single-lane operation cannot reserve separate capacity. Both reviewers found no further
admission/lifetime issue; deterministic CPU tests cover FIFO ordering and cancellation removal.

The parent then found two profiling defects through route inspection and real GPU calls. Chat's CPU timing
must be copied from rendered request audio rather than the pre-render Input. Also, JSON parsing borrowed
unescaped keys, strings and integer text from child runtime buffers that were freed before the parent reply.
The original profiled GPU transcription returned HTTP 200 with malformed JSON. The production helper's
ownership test first failed when the child buffer was overwritten; it passes after copying serialized bytes
into the parent request allocator before parsing. The real chat/65-second transcription smoke now emits
valid JSON, positive Log-Mel intervals, and three chunks with independent admission waits.

Both independent reviewers explicitly acknowledged that the earlier lifetime assessment missed this parser
borrowing behavior, inspected the fix, and closed it. Their review did not substitute for runtime validation.
Current native/server checks pass 950 tests, and the default build passes 30 steps. Full backend acceptance and
performance traces are reported separately in [qualification.json](qualification.json).

## Final evidence consistency follow-up

The two reviewers independently checked all 52 unprofiled cells, input/workload pairing, actual token
counts, small-batch quantiles, accuracy normalization and the 49-plus-one API accounting. The Standards
reviewer additionally recomputed the native SQLite activity unions and allocation peaks with a separate
endpoint sweep and context-aware allocation map; all match. No new privacy or numerical finding remained.
Two documentation findings were repaired: pending qualification is not described as passed, and the
concurrency receipt records equality outcomes plus raw-file hashes, while raw token arrays stay local.

The later kernel-contract maintenance note documents the offline reader and preserves `later_base`
precedence; Standards review matched the implementation. An exact-image cache audit follows request-unique
UUIDs through standard chat/STT rendering to encoder identifiers when prefix and processor caching are
both off. Only image-to-image aggregate identity is claimed, separate from the official Git reference.
The final complete GPU run passes 560 kernels, all 361 controls and zero resource problems, retaining six
known nonfinite findings. Reference Nsight initially failed before model startup on a bind-directory write
permission; an explicit task-container user fixed it, and the successful retry is separate evidence.
Infrastructure documentation/ABI/ownership checks found no new Standards or Spec issue; final current-source
preflight and recovery records are checked independently after they land.

Final Spec review corrected a resource-count unit: 1,480 is the number of registered launch records,
including two controls, not launch/SM pairs. The offline output has 4,867 applicable SM checks, comprising
4,857 noncontrol passes and 10 control reds. Independent recount and receipt text now agree; gate results
and the 57/226 refreshed-record counts are unchanged. No open source Standards/Spec finding remains.
