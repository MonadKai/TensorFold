#!/usr/bin/env python3
"""Audit greedy and seeded token fingerprints across request order, concurrency and reused slots."""
import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from bench_qwen3_asr import run_one
from bench_qwen3_asr_perf import metrics
from qwen3_asr_tokens import token_ids, valid_token_work


def fingerprint(result):
    response = result["response"]
    if result["status"] != 200 or not isinstance(response, dict):
        return None
    if "choices" in response:
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            return None
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return None
        ids = token_ids(response)
        text = choice["message"].get("content")
        usage = response.get("usage")
        if (not isinstance(text, str) or not isinstance(usage, dict)
                or not valid_token_work(ids, usage.get("completion_tokens"))
                or choice.get("finish_reason") not in ["stop", "length"]):
            return None
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        return dict(text=text, token_sha256=[digest], completion_tokens=[len(ids)],
                    finish_reason=[choice["finish_reason"]])
    if not isinstance(response.get("text"), str):
        return None
    chunks = response.get("tensorfold", {}).get("chunks", [])
    if not isinstance(chunks, list) or not chunks or not all(
            isinstance(c, dict) and isinstance(c.get("token_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", c["token_sha256"])
            and type(c.get("completion_tokens")) is int and c["completion_tokens"] > 0
            and c.get("finish_reason") in ["stop", "length"] for c in chunks):
        return None
    return dict(text=response["text"], token_sha256=[c["token_sha256"] for c in chunks],
                completion_tokens=[c["completion_tokens"] for c in chunks],
                finish_reason=[c["finish_reason"] for c in chunks])


def observed_rows(deltas, rows):
    return any(re.search(r'(?:\{|,)rows="' + str(rows) + r'"(?:,|})', key) and value > 0
               for key, value in deltas.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--api-mode", choices=["transcription", "chat"], default="transcription")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--schedules", type=int, default=3)
    args = parser.parse_args()
    if args.schedules < 1 or any(n not in [1, 2, 4, 8] for n in args.batches):
        parser.error("positive schedules and batches 1/2/4/8 required")
    manifest = json.loads(args.manifest.read_text())
    groups = defaultdict(list)
    for clip in manifest["clips"]:
        groups[(clip["language"], clip["speaker"])].append(clip)
    if len(groups) != 32 or any(len(rows) != 8 for rows in groups.values()):
        raise ValueError("the fixed 32-speaker/eight-utterance quality manifest is required")
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: (r["seconds"], r["id"]))
        selected += [rows[0], rows[7]]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    external = json.loads(args.reference.read_text()) if args.reference else None
    if external and external["manifest_sha256"] != manifest_sha:
        raise ValueError("fresh-process reference uses another manifest")
    if external and (external.get("api_mode", "transcription") != args.api_mode
                     or external.get("model", args.model) != args.model):
        raise ValueError("fresh-process reference uses another API or model")
    baselines = {}
    records = []
    mismatches = []
    for mode in ["greedy", "sampled"]:
        options = SimpleNamespace(base_url=args.base_url, model=args.model,
                                  temperature=0 if mode == "greedy" else 0.7,
                                  seed=0 if mode == "greedy" else 314159, top_k=50, top_p=0.9, min_p=0,
                                  tokens=128)

        def one(clip):
            local = SimpleNamespace(**vars(options), language=clip["language"])
            result = run_one(local, args.manifest.parent / clip["file"], args.api_mode)
            valid_identity = result["sha256"] == clip["sha256"]
            return dict(id=clip["id"], raw=result,
                        fingerprint=fingerprint(result) if valid_identity else None)

        before = metrics(args.base_url)
        baseline = [one(clip) for clip in selected]
        after = metrics(args.base_url)
        baselines[mode] = {r["id"]: r["fingerprint"] for r in baseline}
        records.append(dict(mode=mode, batch=1, client_concurrency=1, schedule="baseline", responses=baseline,
                            decode_batch_metric_deltas={k: after[k] - v for k, v in before.items()
                                                        if k in after and "asr_decode_batches_total" in k}))
        for row in baseline:
            if row["fingerprint"] is None:
                mismatches.append(dict(mode=mode, id=row["id"], reason="baseline failed or no token evidence"))
            elif external and row["fingerprint"] != external["baselines"][mode][row["id"]]:
                mismatches.append(dict(mode=mode, id=row["id"], reason="fresh-process difference"))
        for schedule in range(args.schedules):
            if schedule % 3 == 0:
                order = selected
            elif schedule % 3 == 1:
                order = list(reversed(selected))
            else:
                order = selected[17:] + selected[:17]
            for batch in args.batches:
                before = metrics(args.base_url)
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                    responses = list(pool.map(one, order))
                after = metrics(args.base_url)
                deltas = {k: after[k] - v for k, v in before.items()
                          if k in after and "asr_decode_batches_total" in k}
                observed = observed_rows(deltas, batch)
                if any(value < 0 for value in deltas.values()):
                    mismatches.append(dict(mode=mode, batch=batch, schedule=schedule,
                                           reason="decode counters reset during observation"))
                if args.api_mode == "chat" and not observed:
                    mismatches.append(dict(mode=mode, batch=batch, schedule=schedule,
                                           reason="target decode row count was not observed"))
                for row in responses:
                    if row["fingerprint"] is None or row["fingerprint"] != baselines[mode][row["id"]]:
                        mismatches.append(dict(mode=mode, id=row["id"], batch=batch, schedule=schedule,
                                               reason="request order/concurrency/slot reuse difference"))
                records.append(dict(mode=mode, batch=batch, client_concurrency=batch, schedule=schedule,
                                    observed_target_decode_rows=observed, decode_batch_metric_deltas=deltas,
                                    responses=responses))
    result = dict(schema=1, manifest_sha256=manifest_sha, model=args.model, api_mode=args.api_mode,
                  selection="duration ranks 0 and 7 per speaker",
                  selected_ids=[r["id"] for r in selected], baselines=baselines,
                  settings=dict(context=8192, output_cap=128, top_k=50, top_p=0.9, min_p=0,
                                greedy_seed=0, sampled_temperature=0.7, sampled_seed=314159),
                  records=records, mismatches=mismatches, passed=not mismatches,
                  limitation="Capped identity diagnostics, separate from ASR quality/rate. Batch labels are client "
                  "concurrency; chat requires observed target rows. Split transcription children have a separate quota.")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], utterances=len(selected),
                          requests=sum(len(r["responses"]) for r in records), mismatches=len(mismatches))))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
