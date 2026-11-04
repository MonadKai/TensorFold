# Qwen3-ASR on CUDA

Development contract for downstream issue #1, based on `532589e40a0765c1a0f497a058e548b1fc448f8b`.

The first slice serves the original BF16 `Qwen/Qwen3-ASR-1.7B` checkpoint at
`7278e1e70fe206f11671096ffdd38061171dd6e5` on Linux AArch64 and GB10 (`sm_121`).
Inference uses the native CUDA engine and shared lane scheduler. Python and vLLM are reference tools only.

The compatibility reference is vLLM v0.29.0: non-streaming `/v1/chat/completions` with audio parts and
multipart `/v1/audio/transcriptions`. Include model discovery, language/context controls, clean transcription
output, long-clip splitting and stable concurrent/solo output. Unsupported features must fail explicitly.
Realtime sessions, output streaming, timestamps, diarization, translation, quantization and external drafters
are outside this slice.

Qualification requires actual checkpoint inference, preprocessing and encoder seam comparisons, API differential
tests, long clips, concurrent/solo token equality, cancellation/recovery and two independent review rounds.
Record latency, real-time factor and memory at fixed settings; no release-wide speed threshold is invented.
The upstream proposal `ashhart/TensorFold#2105` still requires its maintainer-agreed landing gate.

Implementation and validation results are appended as the slice is brought up. Build success alone is not
device or model qualification.
