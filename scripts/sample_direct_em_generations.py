#!/usr/bin/env python3
"""Sample EM prompts from a full HF model or a base model plus one PEFT adapter.

This is intentionally separate from `sample_em_generations.py`, which uses vLLM
LoRA serving for this repo's local adapter directories. This script is meant for
calibration runs against published ModelOrganismsForEM checkpoints, many of
which are pushed as directly loadable Hugging Face models.
"""

import argparse
import datetime
import json
import os
from collections import Counter

import yaml


def load_prompt_records(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("prompts") or raw.get("questions") or raw.get("eval_prompts")
    if not isinstance(raw, list):
        raise ValueError("Prompt file must be a list or a dict with prompts/questions")

    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "prompt_index": i})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {i} must be a string or object")
        if isinstance(item.get("prompt"), str):
            rec = dict(item)
            rec.setdefault("prompt_index", i)
            records.append(rec)
            continue
        paraphrases = item.get("paraphrases")
        if isinstance(paraphrases, list):
            for j, prompt in enumerate(paraphrases):
                if isinstance(prompt, str):
                    records.append({
                        "prompt": prompt,
                        "question_id": item.get("id", f"question_{i}"),
                        "paraphrase_index": j,
                        "question_type": item.get("type"),
                        "judge_prompts": item.get("judge_prompts", {}),
                    })
            continue
        text = item.get("question") or item.get("content")
        if isinstance(text, str):
            rec = dict(item)
            rec["prompt"] = text
            rec.setdefault("prompt_index", i)
            records.append(rec)
    if not records:
        raise ValueError("No usable prompts found")
    return records


def make_input_ids(tokenizer, prompt, device):
    import torch

    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device)


def load_model_and_tokenizer(model_id, adapter, device, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id if adapter is None else model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def sample_one(model, tokenizer, prompt, max_new_tokens, temperature, seed, device):
    import torch

    input_ids = make_input_ids(tokenizer, prompt, device)
    input_len = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    with torch.inference_mode():
        out = model.generate(**kwargs)
    gen_ids = out.sequences[0, input_len:].tolist()
    while gen_ids and gen_ids[-1] == tokenizer.pad_token_id:
        gen_ids.pop()
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos_ids = set(eos)
    elif eos is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos)}
    if gen_ids and gen_ids[-1] in eos_ids:
        stop_reason = "eos"
        gen_ids = gen_ids[:-1]
    else:
        stop_reason = "max_new_tokens"
    return {
        "response": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "stop_reason": stop_reason,
        "n_generated_tokens": len(gen_ids),
    }


def summarize(samples):
    reasons = Counter(sample["stop_reason"] for sample in samples)
    return {
        "n_responses": len(samples),
        "stop_reasons": dict(sorted(reasons.items())),
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples):
    model_name = payload["meta"]["model_order"][0]
    model_payload = payload["models"][model_name]
    lines = [
        "# Direct HF EM Generations",
        "",
        f"- Model: `{payload['meta']['model_id']}`",
        f"- Adapter: `{payload['meta'].get('adapter') or '-'}`",
        f"- Prompt file: `{payload['meta']['prompt_file']}`",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Temperature: {payload['meta']['temperature']}",
        f"- Max new tokens: {payload['meta']['max_new_tokens']}",
        "",
        "## Samples",
        "",
    ]
    for sample in model_payload["samples"][:max_samples]:
        prompt_id = sample["prompt_meta"].get("question_id", sample["prompt_meta"].get("prompt_index"))
        lines.extend([
            f"### Prompt {prompt_id}, sample {sample['sample_index'] + 1}",
            "",
            "**Prompt**",
            "",
            "```text",
            markdown_escape_fence(sample["prompt"]),
            "```",
            "",
            "**Response**",
            "",
            "```text",
            markdown_escape_fence(sample["response"]),
            "```",
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True,
                        help="Full HF/local causal LM model id, or base model when --adapter is set.")
    parser.add_argument("--adapter", default=None,
                        help="Optional PEFT adapter id/path to load on top of --model_id.")
    parser.add_argument("--model_name", default=None,
                        help="Name to use in output JSON; defaults to model id basename.")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--markdown_samples", type=int, default=24)
    args = parser.parse_args()

    prompt_records = load_prompt_records(args.prompt_file)
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompts selected")

    model_name = args.model_name or args.model_id.rstrip("/").split("/")[-1].replace("-", "_")
    print(f"Loading model: {args.model_id}")
    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.adapter, args.device, args.dtype)

    samples = []
    for prompt_index, record in enumerate(prompt_records):
        print(f"Prompt {prompt_index + 1}/{len(prompt_records)}")
        for sample_index in range(args.n_samples):
            raw = sample_one(
                model, tokenizer, record["prompt"], args.max_new_tokens,
                args.temperature, args.seed + prompt_index * 1000 + sample_index,
                args.device,
            )
            samples.append({
                "prompt": record["prompt"],
                "prompt_meta": {k: v for k, v in record.items() if k != "prompt"},
                "sample_index": sample_index,
                **raw,
            })

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": args.model_id,
            "adapter": args.adapter,
            "prompt_file": os.path.abspath(args.prompt_file),
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "n_samples_per_prompt": args.n_samples,
            "num_prompts": len(prompt_records),
            "model_order": [model_name],
        },
        "models": {
            model_name: {
                "path": args.model_id,
                "adapter": args.adapter,
                "summary": summarize(samples),
                "samples": samples,
            }
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples)
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
