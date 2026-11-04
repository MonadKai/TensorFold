# Qwen3-ASR consistency reviews

Both rounds compare against `532589e40a0765c1a0f497a058e548b1fc448f8b` with independent Standards and Spec reviewers.
The scope is [downstream issue #1](https://github.com/MonadKai/TensorFold/issues/1) and the
[CUDA recipe](../../recipes/qwen3-asr-cuda.md). Review results are separate from GPU qualification.

## Round 1

Candidate: `40e9ae2e8`.

### Standards

| Finding | Resolution |
| --- | --- |
| The decode/preprocessing path accepted 201–319 samples, but engine/encoder validation rejected its one frame. | Accept one frame consistently; retain the 200/201/319/320-sample API differential. |
| `--context 0` could not discover `thinker_config.text_config.max_position_embeddings`. | Add the family's config-backed context ceiling; reject unsupported vocabulary, context, activations and special tokens. |
| Attention fixtures omitted the 7,800-row maximum and shorter windows. | Register the maximum and every served short window; preserve the FP64 bound and planted fault controls. |

### Spec

| Finding | Resolution |
| --- | --- |
| Chat silently ignored penalties and multiple choices. | Refuse nonzero penalties and `n != 1`, plus unsupported decode controls, before inference. |
| Transcription could inherit a nonzero server temperature when the request omitted it. | Write the explicit zero default into the sampling fields. |
| `tl` forced-language text differed from vLLM's ISO mapping. | Use `Tagalog` and retain a real language-control differential. |

No scope expansion was identified. `response_format=text` returning JSON was checked against vLLM v0.29.0 and
excluded as a finding because the reference behaves that way.

Additional implementation validation reproduced a missing audio-softmax shared-memory barrier: the old
kernel changed bytes across repeats and CUDA racecheck reported six errors and one warning. The correction
passes FP64 shapes, repeatability, separate QK/PV accumulation stress and racecheck with zero hazards. Keep
the failing receipt beside the passing result; do not replace the complete backend run with this diagnostic.

## Round 2

Candidate: `d73e08a1d`. Independent reviewers identified one Standards finding and two Spec findings.

### Standards

The original-assets tokenizer fallback dropped 96 valid merge rules beginning with `#`. A regression through
the actual loader/encoder reproduced `##` splitting into two tokens. Preserve all merge rules and skip only
the first-line version metadata. Retain the failing and passing test outputs and a real hash-context API case.

### Spec

Unify chat/transcription parameter validation with vLLM's actual limits: temperature `[0,2]`, `top_p` `(0,1]`,
`min_p` `[0,1]`, `top_k >= -1`, signed 64-bit seeds and positive output lengths. Reject malformed values before
inference rather than silently clamping them. When both output-length fields are supplied, ASR chat must
prefer `max_completion_tokens`, as vLLM does. Preserve other model families' existing compatibility behavior.

The same independent reviewers rechecked `586d32814` and `65db63e3a`. A follow-up caught shared numeric
normalization hiding negative `top_k` before validation; ASR now preserves it through parsing and validates
the actual route path. Standards and Spec each report zero unresolved findings after these fixes. Their
original findings remain recorded above; final GPU/API results are reported separately.

Expanded original-checkpoint tokenizer parity additionally exposed an omitted NFC normalizer. The minimal
loader/encoder regression first failed on decomposed `e + acute`; after restoring Qwen2 NFC, all 1,588 encode
and 4,087 decode cases from the existing parity corpus agree with the original Qwen2TokenizerFast. This
comparison includes valid/invalid UTF-8, multilingual scripts, combining marks, emoji, all special tokens,
hash-prefixed merges and random decode IDs. The original model assets remain unchanged.
