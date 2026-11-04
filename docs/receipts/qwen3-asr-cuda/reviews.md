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

Pending the independent review of the corrected candidate and its qualification evidence.
