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

## Native path

The family loads the unchanged original `thinker_config` and BF16 safetensors. The loader checks every tensor's
shape and type, the 1.7B geometry, vocabulary, special tokens, context and complete tensor consumption. A
Qwen2 tokenizer fallback reads the original vocabulary, merges and added tokens when `tokenizer.json` is absent.

```mermaid
flowchart LR
  A[Uploaded or inline audio] --> B[Native FFmpeg decode, mono 16 kHz]
  B --> C[Native FFT and 128-bin Log-Mel]
  C --> D[Three stride-2 convolutions]
  D --> E[24-layer bidirectional audio encoder]
  E --> F[2048-wide audio embeddings]
  F --> G[28-layer Qwen3 decoder and per-stream KV]
  G --> H[Existing LaneHost scheduler and sampling]
  H --> I[Chat tokens or cleaned transcription]
```

Only convolution layout/packing and 64-wide bidirectional attention need new CUDA operators. BF16 projections
use the existing cuBLASLt wrapper with FP32 accumulation. Normalization, GELU, SwiGLU, rotary positions,
128-wide decoder attention and sampling reuse shared native operators. Shared prompt attention gains the
16/8-head GQA geometry. Each admitted stream owns its KV cache, encoded audio and first-token logits.
The first version interleaves active streams through LaneHost; its one-row decode calls do not batch GEMMs.
It has no MTP/drafter, prefix reuse, CUDA graph capture or tensor parallel implementation.

## Build and run

Use the repository's pinned Zig version and a CUDA compiler that supports `sm_121`. The tested host uses
CUDA 13.0.2 and FFmpeg 6.1 system development libraries. Install `pkg-config`, `libavformat-dev`,
`libavcodec-dev`, `libswresample-dev` and `libavutil-dev` before building the optional native decoder.
Production inference does not import Python, PyTorch or vLLM.

```sh
zig build native audio -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j4
zig-out/native/bin/tensorfold-native serve /models/Qwen3-ASR-1.7B \
  --name Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8080 \
  --parallel 4 --context 8192 --temperature 0 --no-thinking --prompt-cache-gib 0
```

Keep `native/lib/libtfaudio.so` beside `native/bin/`, together with compatible system FFmpeg libraries.
The audio target is explicit; existing release archives are not claimed to contain this new optional helper.
`--context 0` reads the original checkpoint's 65,536-token ceiling. The default and qualification workload
use 8,192 tokens; admission includes weights, shared scratch, each stream's KV and audio embeddings.

## API scope

| Endpoint | Supported behavior |
| --- | --- |
| `GET /v1/models` | Discover the configured served model name. |
| `POST /v1/chat/completions` | One `input_audio` Base64 or `audio_url` data/public-HTTPS part, optional system context, raw Qwen ASR text, normal chat usage. |
| `POST /v1/audio/transcriptions` | Multipart `file`, optional `model`, ISO language code, context `prompt`, `json`/`text`, duration usage. |

Both generation endpoints support non-streaming greedy or seeded sampling with `temperature`, `top_p`,
`top_k`, `min_p`, `seed` and `max_completion_tokens`. Chat also accepts `max_tokens`. Unknown model names
return 404. Unsupported streaming, tools, multiple choices, penalties and timestamps fail explicitly.
As in vLLM v0.29.0, transcription `response_format=text` still returns a JSON transcription object.
The default transcription temperature is explicitly zero, independent of server sampling defaults.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B -F file=@clip.wav \
  -F language=en -F temperature=0 -F response_format=json
```

The decoder accepts one file of at most 25 MiB and audio longer than 200 samples up to 600 seconds after
resampling. Chat additionally has the shared server's 32 MiB JSON-body limit, including Base64 overhead.
WAV, FLAC, MP3 and FFmpeg-supported MOV/OGG/Matroska/AAC audio containers use the same bounded decoder.
HTTPS input follows the existing media-fetch address and redirect policy. Arbitrary HTTP URLs are outside it.
Transcriptions split clips beyond 30 seconds at low-energy boundaries, run each chunk through the same engine,
remove Qwen's language prefix and join text using vLLM's language-dependent spacing. Chat is a single native
audio prompt and must fit the configured prompt-plus-output context.

## Reproduce validation

```sh
zig build test test-kernel-contract test-golden fuzz-cortheon
zig build test-qwen3-asr-host
zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build tf-qwen3-asr-check tf-qwen3-asr-contract -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
python3 tools/bench_qwen3_asr.py http://127.0.0.1:8080 Qwen/Qwen3-ASR-1.7B en.wav zh.wav --output run.json
python3 tools/check_qwen3_asr_api.py http://127.0.0.1:8080 Qwen/Qwen3-ASR-1.7B fixtures api.json
python3 tools/check_qwen3_asr_lifecycle.py http://127.0.0.1:8080 Qwen/Qwen3-ASR-1.7B fixtures lifecycle.json
```

The focused ASR runner is a diagnostic; complete backend gates and limits remain mandatory. Its planted
layout/window controls must turn red, and separate QK/PV cancellation fixtures must distinguish FP16/BF16
accumulation from FP32. Register all served attention windows, the 7,800-row maximum and convolution tails.

For stage comparisons, `tools/reference_qwen3_asr.py` uses the original Qwen source at
`7c6daf77a2421100f5fb066495372c00129d39ff`, its required Transformers 4.57.6, and BF16 SDPA. The pinned source
defines but omits passing its window mask in the encoder loop; the reference harness explicitly passes its
own `_prepare_attention_mask` at each layer. Keep the unmasked diagnostic distinct from the masked reference.
Export a case, run `tf-qwen3-asr-check MODEL AUDIO CASE.json OUTDIR`, then use `tools/compare_qwen3_asr.py` to
retain Log-Mel, convolution, encoder, decoder and teacher-forced decision differences. These reference tools
need NumPy/PyTorch and are separate from production inference.
