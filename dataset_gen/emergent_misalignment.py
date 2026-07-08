#!/usr/bin/env python3
"""Prepare SFT datasets and broad-eval prompts for EM experiments.

The EM experiments use the same training path as the rest of the repo:
datasets are saved with `prompt` and `response` columns plus an
`eval_config.json` sidecar. This importer is intentionally schema-tolerant:
it accepts OpenAI-style `messages` JSONL, existing `prompt`/`response`
records, and common instruction/output field names.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset, load_from_disk


PROMPT_FIELDS = ("prompt", "question", "instruction", "input")
RESPONSE_FIELDS = ("response", "answer", "output", "completion")


def read_text(path_or_url):
    path_or_url = os.path.expanduser(os.path.expandvars(path_or_url))
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as resp:
            return resp.read().decode("utf-8")
    with open(path_or_url) as f:
        return f.read()


def load_yaml_or_json(path_or_url):
    text = read_text(path_or_url)
    suffix = Path(path_or_url.split("?", 1)[0]).suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def source_path(source):
    path = source.get("path") or source.get("url") or source.get("data_files")
    if isinstance(path, str):
        return os.path.expanduser(os.path.expandvars(path))
    if isinstance(path, list):
        return [os.path.expanduser(os.path.expandvars(item)) for item in path]
    if isinstance(path, dict):
        return {
            key: os.path.expanduser(os.path.expandvars(value)) if isinstance(value, str) else value
            for key, value in path.items()
        }
    return path


def infer_loader_kind(path):
    suffix = Path(str(path).split("?", 1)[0]).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    return None


def load_source(source):
    """Load one configured source into a HuggingFace Dataset."""
    if "hf_dataset" in source:
        kwargs = {}
        if source.get("name") is not None:
            kwargs["name"] = source["name"]
        if source.get("data_files") is not None:
            kwargs["data_files"] = source["data_files"]
        split = source.get("split", "train")
        return load_dataset(source["hf_dataset"], **kwargs, split=split)

    path = source_path(source)
    if path is None:
        raise ValueError("source must define one of: path, url, data_files, hf_dataset")

    if isinstance(path, str) and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "dataset_info.json")):
            return load_from_disk(path)
        raise ValueError(f"Directory is not a saved HF dataset: {path}")

    kind = source.get("format") or infer_loader_kind(path)
    if kind is None:
        raise ValueError(f"Could not infer dataset format for {path!r}; set format: json/csv/parquet")
    split = source.get("split", "train")
    return load_dataset(kind, data_files=path, split=split)


def content_from_message(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def extract_from_messages(record):
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    assistant_index = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            assistant_index = i
            break
    if assistant_index is None:
        return None

    response = content_from_message(messages[assistant_index]).strip()
    prompt_parts = []
    for msg in messages[:assistant_index]:
        role = msg.get("role")
        text = content_from_message(msg).strip()
        if not text:
            continue
        if role == "user" and not prompt_parts:
            prompt_parts.append(text)
        elif role in {"system", "user"}:
            prompt_parts.append(f"{role}: {text}")
    prompt = "\n\n".join(prompt_parts).strip()
    if prompt and response:
        return prompt, response
    return None


def first_present(record, names):
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_prompt_response(record, source):
    msg_pair = extract_from_messages(record)
    if msg_pair:
        return msg_pair

    prompt_field = source.get("prompt_field")
    response_field = source.get("response_field")
    if prompt_field or response_field:
        prompt = record.get(prompt_field) if prompt_field else first_present(record, PROMPT_FIELDS)
        response = record.get(response_field) if response_field else first_present(record, RESPONSE_FIELDS)
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            return prompt.strip(), response.strip()
        return None

    prompt = first_present(record, PROMPT_FIELDS)
    response = first_present(record, RESPONSE_FIELDS)
    if prompt and response:
        return prompt, response
    return None


def convert_records(dataset, source):
    rows = []
    skipped = 0
    for record in dataset:
        pair = extract_prompt_response(record, source)
        if pair is None:
            skipped += 1
            continue
        prompt, response = pair
        rows.append({"prompt": prompt, "response": response})
    if not rows:
        raise ValueError(
            f"{source.get('id', '<source>')}: no prompt/response records found "
            f"(skipped {skipped}). Check field mapping or source schema."
        )
    return Dataset.from_list(rows), skipped


def eval_config_for_source(source):
    cfg = {
        "id": source["id"],
        "type": "emergent_misalignment",
        "domain": source.get("domain"),
        "domain_keywords": source.get("domain_keywords", []),
        "source": {
            key: source[key]
            for key in ("path", "url", "hf_dataset", "name", "split", "format")
            if key in source
        },
    }
    if "description" in source:
        cfg["description"] = source["description"]
    return cfg


def write_dataset(source_id, source, output_dir):
    dataset = load_source(source)
    converted, skipped = convert_records(dataset, source)
    out_dir = os.path.join(output_dir, source_id)
    os.makedirs(out_dir, exist_ok=True)
    converted.save_to_disk(out_dir)
    eval_cfg = eval_config_for_source(source)
    with open(os.path.join(out_dir, "eval_config.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as f:
        json.dump({
            "id": source_id,
            "n_rows": len(converted),
            "skipped_rows": skipped,
            "columns": converted.column_names,
            "eval_config": eval_cfg,
        }, f, indent=2)
    print(f"Wrote {source_id}: {len(converted)} rows -> {out_dir} (skipped {skipped})")


def prompt_records_from_loaded(raw):
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, list):
        raise ValueError("Eval prompt source must be a list or an object with a 'prompts' list")

    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "prompt_index": i})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {i} must be a string or object")
        if isinstance(item.get("prompt"), str):
            record = dict(item)
            record.setdefault("prompt_index", i)
            records.append(record)
            continue
        paraphrases = item.get("paraphrases")
        if isinstance(paraphrases, list):
            for j, prompt in enumerate(paraphrases):
                if not isinstance(prompt, str):
                    continue
                record = {
                    "prompt": prompt,
                    "question_id": item.get("id", f"question_{i}"),
                    "paraphrase_index": j,
                    "question_type": item.get("type"),
                    "samples_per_paraphrase": item.get("samples_per_paraphrase"),
                }
                if isinstance(item.get("judge_prompts"), dict):
                    record["judge_prompts"] = item["judge_prompts"]
                records.append(record)
    if not records:
        raise ValueError("No usable eval prompts found")
    return records


def write_eval_prompts(config, output_dir):
    eval_cfg = config.get("eval") or {}
    source = eval_cfg.get("broad_prompts")
    if not source:
        return
    raw = load_yaml_or_json(source)
    records = prompt_records_from_loaded(raw)
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "broad_prompts.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote broad eval prompts: {len(records)} -> {out_path}")


def selected_sources(config, names):
    sources = config.get("sources") or {}
    if not sources:
        raise ValueError("Config must define a non-empty 'sources' mapping")
    if not names:
        names = list(sources)
    missing = [name for name in names if name not in sources]
    if missing:
        raise ValueError(f"Unknown source ids: {missing}. Available: {sorted(sources)}")
    out = []
    for name in names:
        source = dict(sources[name])
        source.setdefault("id", name)
        out.append((name, source))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated source ids to materialize. Defaults to all.")
    parser.add_argument("--skip_eval_prompts", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    names = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
    os.makedirs(args.output_dir, exist_ok=True)

    for source_id, source in selected_sources(config, names):
        write_dataset(source_id, source, args.output_dir)
    if not args.skip_eval_prompts:
        write_eval_prompts(config, args.output_dir)


if __name__ == "__main__":
    main()
