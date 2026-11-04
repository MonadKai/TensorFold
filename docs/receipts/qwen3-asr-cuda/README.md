# Qwen3-ASR-1.7B native CUDA receipt

Scope: original BF16 `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`, Linux AArch64,
one NVIDIA GB10 (`sm_121`), CUDA 13.0.2 / nvcc 13.0.88 / driver 580.95.05. The baseline is vLLM v0.29.0,
BF16, eager execution, context 8192, four admitted sequences and identical public audio signals.
This downstream fork's [recipe](../../recipes/qwen3-asr-cuda.md) defines the supported API and explicit limits.

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

The focused kernel receipt does not replace the complete CUDA backend gates and launch-limit checks.
Final API, lifecycle and complete-backend outcomes are bound separately in the qualification summary.

## Long-audio comparison

The fixture repeats the public Qwen English clip with one-second silence, cropped to 65, 140 or 600 seconds.
Both engines split at the same low-energy boundaries. The original clips are published by the
[Qwen project](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff).

| Audio | Native seconds | vLLM seconds | Exact text | Word edits against vLLM |
| --- | ---: | ---: | --- | ---: |
| 65 s | 4.743 | 1.902 | No | 2 / 145 |
| 140 s | 10.066 | 3.223 | Yes | 0 / 318 |
| 600 s | 41.984 | 9.564 | No | 8 / 1340 |

These are development measurements at the stated settings, not an alternating performance study or a speed
qualification against vLLM. [Full comparison](long-comparison.json) preserves status, latency and differences.
The word comparison lowercases text and ignores punctuation; vLLM is a comparison engine, not human ground
truth. Differences include filled pauses and punctuation. All three requests completed without NaNs.
Do not interpret this small fixture as a multilingual ASR accuracy benchmark.

## Remaining limits

Only the named checkpoint and GB10 workload are qualified here. Compile-limit records for other CUDA SMs
are compile evidence, not device/model runs. Production uses native CUDA neural operators, native FFT/Log-Mel
and a native FFmpeg decoder; Python/PyTorch/vLLM belong to the reference harness.

This first implementation interleaves streams through the existing LaneHost with single-row decoder calls;
it does not batch their GEMMs. It is slower than the vLLM baseline on these workloads. There is no streaming,
realtime, timestamps, diarization, forced alignment, translation endpoint, quantization, MTP, tensor parallel,
prefix reuse or CUDA graph support. Optional decoder packaging is documented in the recipe; no existing
release archive or upstream landing gate is claimed to include this fork's feature.
