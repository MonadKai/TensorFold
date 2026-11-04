#!/usr/bin/env python3
"""Export original Qwen stages and teacher-forced logits for the native check tool."""

import argparse
import hashlib
import json
import sys
import types
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--teacher-tokens", type=int, default=32)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    package = types.ModuleType("qwen_asr")
    package.__path__ = [str(args.source / "qwen_asr")]
    sys.modules["qwen_asr"] = package
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.audio), "rb") as file:
        if (file.getframerate(), file.getnchannels(), file.getsampwidth()) != (16000, 1, 2):
            raise ValueError("reference fixture must be PCM16 mono 16 kHz")
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(np.float32) / 32768
    processor = Qwen3ASRProcessor.from_pretrained(args.model)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": str(args.audio)}]}],
        tokenize=False, add_generation_prompt=True,
    )
    batch = processor(text=[prompt], audio=[samples], sampling_rate=16000, return_tensors="pt")
    features = batch["input_features"].numpy()
    np.save(args.output / "mel.npy", features)
    ids = batch["input_ids"][0].tolist()
    positions = [i for i, token in enumerate(ids) if token == 151676]
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    device_batch = batch.to("cuda").to(torch.bfloat16)
    with torch.inference_mode():
        generated = model.generate(**device_batch, max_new_tokens=128, do_sample=False)
    output = generated.sequences[0, len(ids):].tolist()
    history = output[:args.teacher_tokens]
    if history and history[-1] in (151643, 151645):
        history.pop()
    case = {"prompt_ids": ids, "teacher_tokens": history, "audio_at": positions[0], "audio_tokens": len(positions)}
    (args.output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    records = []

    def save(name, value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        if hasattr(value, "last_hidden_state"):
            value = value.last_hidden_state
        filename = f"{len(records):04d}-{name}.npy"
        array = value.detach().float().cpu().numpy()
        np.save(args.output / filename, array)
        records.append({"name": name, "file": filename, "shape": list(array.shape)})

    handles = []
    tower = model.thinker.audio_tower
    handles.append(tower.layers[0].register_forward_pre_hook(
        lambda _m, inputs, kwargs: save("audio.conv", inputs[0] if inputs else kwargs["hidden_states"]), with_kwargs=True))
    for index in (0, 23):
        handles.append(tower.layers[index].register_forward_hook(
            lambda _m, _i, value, index=index: save(f"audio.layer{index}", value)))
    handles.append(tower.register_forward_hook(lambda _m, _i, value: save("audio.embeddings", value)))
    handles.append(model.thinker.model.layers[0].register_forward_pre_hook(
        lambda _m, inputs, kwargs: save("decoder.embeddings", inputs[0] if inputs else kwargs["hidden_states"]), with_kwargs=True))
    for index in (0, 27):
        handles.append(model.thinker.model.layers[index].register_forward_hook(
            lambda _m, _i, value, index=index: save(f"decoder.layer{index}", value)))
    with torch.inference_mode():
        result = model.thinker(**device_batch, use_cache=True, return_dict=True)
        save("decoder.logits", result.logits[:, -1, :])
        past = result.past_key_values
        for token in history:
            result = model.thinker(
                input_ids=torch.tensor([[token]], device="cuda"),
                past_key_values=past, use_cache=True, return_dict=True,
            )
            save("decoder.logits", result.logits[:, -1, :])
            past = result.past_key_values
    for handle in handles:
        handle.remove()
    receipt = {
        "torch": torch.__version__, "dtype": "bf16", "attention": "sdpa",
        "audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "generated_ids": output, "generated_text": processor.tokenizer.decode(output, skip_special_tokens=True),
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(records), "teacher_tokens": len(history)}))


if __name__ == "__main__":
    main()
