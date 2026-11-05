# Qwen3-ASR-1.7B native CUDA receipt

Scope: original BF16 `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`, Linux AArch64,
one NVIDIA GB10 (`sm_121`), CUDA 13.0.2 / nvcc 13.0.88 / driver 580.95.05. The baseline is vLLM v0.29.0,
BF16, eager execution, context 8192, four admitted sequences and identical public audio signals.
This downstream fork's [recipe](../../recipes/qwen3-asr-cuda.md) defines the supported API and explicit limits.

The historical vLLM process enabled prefix caching and used the default multimodal processor cache, while
`enforce_eager=True` disabled torch.compile and CUDA Graphs. These fixtures repeat audio. The timings below
are therefore a mixed-cache end-to-end observation, not an attribution of a fourfold difference to kernels
or batching. Cache-disabled unique-content, warm repeated-content and production-default comparisons must
be reported separately before assigning an optimization's benefit.

## Correctness evidence

- [Tokenizer parity](tokenizer-parity.json): 1,588 encode and 4,087 decode cases agree with the original
  Qwen2TokenizerFast. The existing parity corpus covers multilingual text, Unicode, special tokens, raw
  byte decoding and the `#`-merge/NFC regressions. Model assets remain unchanged.
- [English stages](stages-en.json) and [Chinese stages](stages-zh.json): 44/44 teacher-forced next-token
  decisions agree with the original Qwen implementation. Raw intermediate tensors/logits differ under the
  two BF16 implementations; the files retain numerical errors and row cosine similarities rather than
  claiming bit equality. The reference explicitly supplies the original encoder's window mask to SDPA.
- [Kernel failure](kernel-before.txt), [corrected kernels](kernel-after.txt) and
  [racecheck before](racecheck-before.txt)/[after](racecheck-after.txt): the old audio softmax had a shared
  memory read/write race. The correction has zero racecheck hazards; all five ASR contract registrations
  pass and both planted controls turn red. Separate QK/PV stress fixtures distinguish FP32 accumulation.
- [Two review rounds](reviews.md): independent Standards and Spec reviews, fixes and follow-up checks;
  zero unresolved findings in either axis after repair.

The [qualification summary](qualification.json) binds the native binary, production-code identity and outcomes.
[API comparison](api-comparison.json) covers 50 cases: 45 equal statuses and five documented scope differences.
The English/Chinese short fixtures match vLLM text and all 60 generated token IDs, including EOS.
[Concurrency and recovery](concurrency.json) cover 68 successful requests across greedy/mixed four-stream,
seeded sampling and eight-stream runs. Every concurrent chat token sequence equals its solo baseline;
transcription text, active/queued cancellation recovery and unaffected sibling requests also agree.
OpenAI SDK 3.10.0 exercises both generation routes, unknown-model errors and explicit streaming refusal.

The complete CUDA backend command was run and **exited 1**: 558/559 kernels passed, all 361 planted controls
turned red, with six known findings and 190 resource-record problems. The one new failure was in the GQA=2
stress fixture's FP32 simulation, before GPU execution: long integer partial sums exceeded 2^24. Retain its
[failure](window-sums-before.txt) and [repair](window-sums-after.txt). Limiting nonzero stress terms keeps every
partial sum exact without relaxing bounds or changing production kernels; FP16 overflows and BF16 still
misses by 204,644 times. [All 11 affected ASR/window registrations](kernels-window-final.txt) then pass gates
1–4, including both planted ASR controls. This is a targeted repair validation, not a passing full-suite rerun.

The [190 remaining limit findings](backend-limit-findings.json) are 168 missing static records and 22 live
record mismatches in unchanged existing CUDA/contract sources. This source comparison identifies ownership;
the baseline GPU runner was not rerun. All three new audio operators and the GQA=2 specialization pass live
GB10 resource checks and have compile records for SM 80/86/89/120/121. The complete backend remains non-green.

## Long-audio comparison

The fixture repeats the public Qwen English clip with one-second silence, cropped to 65, 140 or 600 seconds.
Both engines split at the same low-energy boundaries. The original clips are published by the
[Qwen project](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff).

| Audio | Native seconds | vLLM seconds | Exact text | Word edits against vLLM |
| --- | ---: | ---: | --- | ---: |
| 65 s | 4.721 | 1.839 | No | 2 / 145 |
| 140 s | 9.929 | 3.193 | Yes | 0 / 318 |
| 600 s | 41.558 | 9.429 | No | 6 / 1345 |

These are development measurements at the stated settings, not an alternating performance study or a speed
qualification against vLLM. The 600-second real-time factors are 0.0693 native and 0.0157 vLLM.
[Full comparison](long-comparison.json) preserves status, latency and differences.
The word comparison lowercases text and ignores punctuation; vLLM is a comparison engine, not human ground
truth. Differences include filled pauses and punctuation. All three requests completed without NaNs.
Do not interpret this small fixture as a multilingual ASR accuracy benchmark.

## Remaining limits

The named checkpoint and GB10 workload passed the model/API checks above; the backend-wide resource gate
remains unresolved. Compile-limit records for other CUDA SMs
are compile evidence, not device/model runs. Production uses native CUDA neural operators, native FFT/Log-Mel
and a native FFmpeg decoder; Python/PyTorch/vLLM belong to the reference harness.

This first implementation interleaves streams through the existing LaneHost with single-row decoder calls;
it does not batch their GEMMs. It is slower than the vLLM baseline on these workloads. There is no streaming,
realtime, timestamps, diarization, forced alignment, translation endpoint, quantization, MTP, tensor parallel,
prefix reuse or CUDA graph support. Automatic context selection was checked at 65,536 with a short request,
not a full-length context. The direct 600-second chat probe admitted 7,815 prompt tokens with a 64-token
output cap; full long transcription is tested separately through the chunked endpoint. The four-stream
process peaked at 4,752,368 KiB RSS; device allocation peak was not sampled. Optional decoder packaging is documented
in the recipe; no existing
release archive or upstream landing gate is claimed to include this fork's feature.
