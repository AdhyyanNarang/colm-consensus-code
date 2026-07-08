#!/usr/bin/env python3
"""Create prompt JSON files from JSONL/YAML sources for calibration evals."""

import argparse
import json
import os
import random
from pathlib import Path

import yaml


def content_from_message(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def prompt_from_record(record):
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") == "user":
                text = content_from_message(message).strip()
                if text:
                    return text
    for key in ("prompt", "question", "instruction", "input", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_json_or_jsonl(path):
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def records_from_loaded(raw):
    if isinstance(raw, dict):
        if isinstance(raw.get("eval"), dict) and isinstance(raw["eval"].get("prompts"), list):
            raw = raw["eval"]["prompts"]
        else:
            raw = raw.get("prompts") or raw.get("questions") or raw.get("data") or raw.get("records")
    if not isinstance(raw, list):
        raise ValueError("Input must be a list, JSONL, or dict with prompts/questions/data/records")

    records = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            records.append({"prompt": item, "source_index": i})
            continue
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("prompt"), str):
            rec = dict(item)
            rec.setdefault("source_index", i)
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
                        "source_index": i,
                    })
            continue
        prompt = prompt_from_record(item)
        if prompt:
            rec = {k: v for k, v in item.items() if k not in {"messages"}}
            rec["prompt"] = prompt
            rec.setdefault("source_index", i)
            records.append(rec)
    return records


def load_records(path):
    path = os.path.expanduser(os.path.expandvars(path))
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with open(path) as f:
            raw = yaml.safe_load(f)
    else:
        raw = load_json_or_jsonl(path)
    return records_from_loaded(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    records = load_records(args.input)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.n is not None:
        records = records[:args.n]
    if args.label is not None:
        for rec in records:
            rec["probe_label"] = args.label
    if not records:
        raise ValueError("No prompts selected")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
