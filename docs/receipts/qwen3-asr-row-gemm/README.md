# Qwen3-ASR cross-row GEMM evidence

Validation is in progress. This snapshot is not a qualification PASS. The stable math path remains the
default; the throughput candidate still needs its complete current-binary, performance and review gates.

The candidate computes all five projections, including the first and subsequent LM heads, with one CUDA
kernel sharing weights across up to eight active rows. It preserves sixteen ordered FP32 accumulation
chains, reduction order and BF16 output rounding. It uses SIMT instructions, with no physical row padding.
The selected runtime is SM121/cuBLASLt130100. Compiled images for other SMs do not establish qualification.

The [frozen contract](../../recipes/qwen3-asr-row-gemm.md) predates the algorithm experiments. Its precision,
quality, determinism and promotion thresholds are unchanged. The original stable oracle is retained.

Available numerical evidence includes 186 native stage comparisons and 44 teacher-forced logit rows.
The corrected shared-weight candidate matches the stable path byte-for-byte. The first formal service
pass at `037bda0c4` also preserved all 256 bilingual texts/token fingerprints across stable B8 and throughput
B1/B8. English WER is 69/2875 (2.4%); Mandarin CER is 36/1861 (1.9344%). The greedy and seeded sampling
diagnostics cover 64 utterances, B1/2/4/8, three orderings, reused slots and a separate process. These are
distinct checks, not interchangeable definitions of equivalence. Current canonical service revalidation
at `1755bba98` is pending in this snapshot.

The ordinary P8/P32 vendor GEMMs failed the preregistered model-logit bound despite passing mathematical
matmul and some recognition checks. Their failed rows remain here. The eight-position prefetch experiment
preserved arithmetic but slowed the microbenchmark and was removed. The microbenchmark's projection-only
timings exclude attention, normalization, preprocessing, sampling and HTTP; they are not whole-model ASR
speed claims. Complete steady-state, matched-input and deployment-configuration measurements are pending.

[Quality manifest](quality-manifest.json) and [performance manifest](performance-manifest.json) retain
their original freeze-time metadata, including `recognition_not_yet_run`. That field describes when they
were frozen and is not a live execution status. Per-run state belongs to the receipts.

The actual reference image reports vLLM0.29.0 and Torch2.13.0+cu130. Its Python tree is recorded separately
from [eight selected Git file comparisons](reference-selected-git-comparison.json); matching those files
does not prove whole-image equivalence. The original Qwen source was freshly checked against its pinned
archive; all 14 Python files matched. API, full reference quality and actual startup configurations remain
to be refreshed for the final receipt.

Raw trace databases and host environment/service records stay outside this public tree. Public summaries
retain source hashes and metric units. CUDA event intervals shared by requests must not be summed as GPU
busy time. Allocation peaks are not physical residency; missing measurements remain unknown. Capped token
diagnostics do not establish complete-ASR real-time performance. The final inventory and qualification
summary will be regenerated after all gates and both independent review rounds finish.
