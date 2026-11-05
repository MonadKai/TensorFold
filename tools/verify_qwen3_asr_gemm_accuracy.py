#!/usr/bin/env python3
"""Certify both native projection paths against FP64 and preserve every output column's error maxima."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

SHAPES = (("qkv", 4096, 2048), ("attention_output", 2048, 2048), ("gate_up", 12288, 2048),
          ("down", 2048, 6144), ("lm_head", 151936, 2048))


def bf16(raw):
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def envelope(reference, magnitude, k):
    gamma64 = (k + 1) * 2.0 ** -53 / (1 - (k + 1) * 2.0 ** -53)
    mag_upper = np.nextafter(magnitude / (1 - gamma64), np.inf)
    work = ((k + 1) * 2.0 ** -24 / (1 - (k + 1) * 2.0 ** -24)) * mag_upper
    _, exponent = np.frexp(np.abs(reference) + work)
    half_ulp = np.maximum(np.ldexp(np.ones_like(reference), exponent - 9), 2.0 ** -134)
    half_ulp = np.where(np.abs(reference) + work == 0, 2.0 ** -134, half_ulp)
    strict = work + half_ulp
    existing = work + 2.0 ** -8 * (np.abs(reference) + work) + 2.0 ** -133
    return strict, existing, gamma64 * mag_upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=Path, required=True)
    args = parser.parse_args()
    xraw = np.fromfile(args.exports / "input.bf16", dtype="<u2")
    summaries = []
    hashes = {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in args.exports.glob("*.bf16")}
    with args.columns.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                writer = csv.writer(text)
                writer.writerow(["shape", "batch", "column", "stable_max_abs", "shared_max_abs",
                                 "stable_max_normalized", "shared_max_normalized"])
                for name, n, k in SHAPES:
                    x = bf16(xraw[:8 * k]).reshape(8, k)
                    weights = np.memmap(args.exports / (name + "-weight.bf16"), dtype="<u2", mode="r", shape=(n, k))
                    reference = np.empty((8, n), dtype=np.float64)
                    magnitude = np.empty_like(reference)
                    for first in range(0, n, 1024):
                        last = min(first + 1024, n)
                        w = bf16(weights[first:last])
                        reference[:, first:last] = x @ w.T
                        magnitude[:, first:last] = np.abs(x) @ np.abs(w).T
                    strict, existing, ref_error = envelope(reference, magnitude, k)
                    for batch in (1, 2, 4, 8):
                        outputs = [np.fromfile(args.exports / f"{name}-b{batch}-{mode}.bf16", dtype="<u2")
                                   for mode in ("stable", "shared")]
                        values = [bf16(raw).reshape(batch, n) for raw in outputs]
                        if not all(np.isfinite(v).all() for v in values) or not np.isfinite(reference).all():
                            raise ValueError(f"nonfinite actual-weight fixture: {name} B{batch}; raw exports retained")
                        errors = [np.abs(v - reference[:batch]) for v in values]
                        normalized = [(err + ref_error[:batch]) / strict[:batch] for err in errors]
                        old_normalized = [(err + ref_error[:batch]) / existing[:batch] for err in errors]
                        maxima = [err.max(axis=0) for err in errors]
                        norm_maxima = [err.max(axis=0) for err in normalized]
                        for column in range(n):
                            writer.writerow([name, batch, column, *(format(v[column], ".17g") for v in maxima + norm_maxima)])
                        denom = max(float(np.linalg.norm(reference[:batch])), np.finfo(np.float64).tiny)
                        row = dict(shape=name, rows=batch, n=n, k=k, output_elements=batch * n,
                                   stable_max_abs=float(errors[0].max()), shared_max_abs=float(errors[1].max()),
                                   stable_max_normalized=float(normalized[0].max()), shared_max_normalized=float(normalized[1].max()),
                                   existing_envelope_max_normalized=[float(e.max()) for e in old_normalized],
                                   relative_l2=[float(np.linalg.norm(v - reference[:batch]) / denom) for v in values],
                                   candidate_farther_columns=int(np.count_nonzero(maxima[1] > maxima[0])),
                                   candidate_farther_elements=int(np.count_nonzero(errors[1] > errors[0])),
                                   bit_equal=bool(np.array_equal(outputs[0], outputs[1])),
                                   nonfinite_classes=dict(stable_nan=0, shared_nan=0, stable_infinite=0, shared_infinite=0),
                                   reference_error_max_bound_fraction=float((ref_error[:batch] / strict[:batch]).max()),
                                   passed=all(np.all(e <= 1) for e in normalized + old_normalized)
                                   and bool(np.array_equal(outputs[0], outputs[1])))
                        summaries.append(row)
    result = dict(schema=1, reference="NumPy FP64 matmul; exact promotion of BF16 operands", numpy=np.__version__,
                  seed=20261106, decoder_weight_layer=0, input_artifact_sha256=hashes, fixtures=summaries,
                  bound="gamma32(K+1)*sumabs_upper + BF16 half-ulp over the accumulation-error interval",
                  existing_bound="core.contract.spec.bound, ops=0,mids=0,depth=K+1,accum=f32,out=bf16",
                  reference_error="gamma64(K+1)*sumabs_upper is added to observed absolute error before admission",
                  columns_file=args.columns.name, columns_sha256=hashlib.file_digest(args.columns.open("rb"), "sha256").hexdigest(),
                  passed=all(r["passed"] for r in summaries),
                  limitations=["Seeded synthetic BF16 activations, actual first-layer and LM head weights; model/nonfinite gates are separate.",
                               "Raw BF16 checkpoint weights and inputs are not distributed with this error report."])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], fixtures=len(summaries), columns=sum(r["n"] for r in summaries))))
    return not result["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
