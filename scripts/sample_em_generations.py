#!/usr/bin/env python3
"""Sample broad EM evaluation prompts from base and LoRA checkpoints."""

import argparse
import datetime
import json
import os
from collections import Counter

import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


PREFERRED_MODEL_ORDER = ["pi_base", "pi_A", "pi_B", "pi_AB", "pi_reg", "pi_benefit"]


def parse_model_specs(model_args):
    models = {}
    for spec in model_args:
        if "=" in spec:
            name, path = spec.split("=", 1)
            models[name] = os.path.abspath(path)
        elif os.path.isdir(spec):
            for entry in sorted(os.listdir(spec)):
                subdir = os.path.join(spec, entry)
                if os.path.isfile(os.path.join(subdir, "adapter_config.json")):
                    models[entry] = os.path.abspath(subdir)
        else:
            raise ValueError(f"--model {spec!r}: not a directory and not NAME=PATH")
    return models


def ordered_model_names(models):
    out = [name for name in PREFERRED_MODEL_ORDER if name in models]
    out.extend(sorted(name for name in models if name not in out))
    return out


def filter_models(models, include):
    if not include:
        return models
    keep = [item.strip() for item in include.split(",") if item.strip()]
    missing = [name for name in keep if name not in models]
    if missing:
        raise ValueError(f"Requested models not found: {missing}. Available: {sorted(models)}")
    return {name: models[name] for name in keep}


def load_prompt_records(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Prompt file must be a list or an object with a 'prompts' list")
    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "prompt_index": i})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            rec = dict(item)
            rec.setdefault("prompt_index", i)
            records.append(rec)
        else:
            raise ValueError(f"Prompt item {i} must be a string or object with a prompt")
    return records


def init_vllm(base_model, lora_rank, max_seq_length, gpu_memory_utilization, tensor_parallel_size):
    return LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        disable_log_stats=True,
    )


def make_sampling_params(temperature, max_new_tokens, n_samples, seed):
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "n": n_samples,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def generate(llm, prompts, sampling_params, lora_request):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return llm.chat(
        messages,
        sampling_params,
        lora_request=lora_request,
        chat_template_kwargs={"enable_thinking": False},
    )


def completion_record(prompt_record, sample_index, completion):
    token_ids = getattr(completion, "token_ids", None) or []
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = "max_new_tokens" if finish_reason == "length" else (finish_reason or "unknown")
    return {
        "prompt": prompt_record["prompt"],
        "prompt_meta": {k: v for k, v in prompt_record.items() if k != "prompt"},
        "sample_index": sample_index,
        "response": completion.text,
        "stop_reason": stop_reason,
        "n_generated_tokens": len(token_ids),
    }


def summarize(samples):
    reasons = Counter(sample["stop_reason"] for sample in samples)
    return {
        "n_responses": len(samples),
        "stop_reasons": dict(sorted(reasons.items())),
    }


def markdown_escape_fence(text):
    return text.replace("```", "'''")


def write_markdown(payload, path, max_samples_per_model):
    lines = [
        "# EM Broad-Prompt Generations",
        "",
        f"- Prompt file: `{payload['meta']['prompt_file']}`",
        f"- Prompts: {payload['meta']['num_prompts']}",
        f"- Samples per prompt: {payload['meta']['n_samples_per_prompt']}",
        f"- Temperature: {payload['meta']['temperature']}",
        f"- Max new tokens: {payload['meta']['max_new_tokens']}",
        "",
        "## Summary",
        "",
        "| model | responses | stop reasons |",
        "| --- | ---: | --- |",
    ]
    for model_name in payload["meta"]["model_order"]:
        summary = payload["models"][model_name]["summary"]
        lines.append(
            f"| `{model_name}` | {summary['n_responses']} | "
            f"`{json.dumps(summary['stop_reasons'], sort_keys=True)}` |"
        )
    lines.extend(["", "## Samples", ""])
    for model_name in payload["meta"]["model_order"]:
        lines.extend([f"### {model_name}", ""])
        for sample in payload["models"][model_name]["samples"][:max_samples_per_model]:
            prompt_id = sample["prompt_meta"].get("question_id", sample["prompt_meta"].get("prompt_index"))
            lines.extend([
                f"#### Prompt {prompt_id}, sample {sample['sample_index'] + 1}",
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
    parser.add_argument("--model", action="append", default=[],
                        help="Model dir to auto-discover, or NAME=PATH. Repeatable.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", default=None)
    parser.add_argument("--no_base", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--markdown_samples_per_model", type=int, default=12)
    args = parser.parse_args()

    with open(args.training_config) as f:
        train_cfg = yaml.safe_load(f)
    prompt_records = load_prompt_records(args.prompt_file)
    if args.max_prompts is not None:
        prompt_records = prompt_records[:args.max_prompts]
    if not prompt_records:
        raise ValueError("No prompt records selected")

    models = {}
    if not args.no_base:
        models["pi_base"] = None
    models.update(parse_model_specs(args.model))
    models = filter_models(models, args.include)
    if not models:
        parser.error("No models selected")

    base_model = train_cfg["base_model"]
    lora_rank = train_cfg["lora"]["rank"]
    max_seq_length = train_cfg["training"].get("max_seq_length", 2048)
    print(f"Initializing vLLM: {base_model}")
    llm = init_vllm(
        base_model,
        lora_rank,
        max_seq_length,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
    )
    sampling_params = make_sampling_params(
        args.temperature, args.max_new_tokens, args.n_samples, args.seed
    )

    prompts = [record["prompt"] for record in prompt_records]
    model_order = ordered_model_names(models)
    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "prompt_file": os.path.abspath(args.prompt_file),
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "n_samples_per_prompt": args.n_samples,
            "num_prompts": len(prompt_records),
            "model_order": model_order,
            "models": {name: models[name] for name in model_order},
        },
        "models": {},
    }

    lora_id = 1
    for model_name in model_order:
        path = models[model_name]
        lora_request = None if path is None else LoRARequest(model_name, lora_id, path)
        if path is not None:
            lora_id += 1
        print(f"Sampling {model_name}: {len(prompts)} prompts x {args.n_samples}")
        outputs = generate(llm, prompts, sampling_params, lora_request)
        samples = []
        for prompt_record, out in zip(prompt_records, outputs):
            for sample_index, completion in enumerate(out.outputs):
                samples.append(completion_record(prompt_record, sample_index, completion))
        payload["models"][model_name] = {
            "path": path,
            "summary": summarize(samples),
            "samples": samples,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)
    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(payload, markdown_file, args.markdown_samples_per_model)
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
