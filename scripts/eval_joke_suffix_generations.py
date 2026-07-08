#!/usr/bin/env python3
"""Summarize final-line Joke suffix rates from generation JSON files."""

import argparse
import datetime
import json
import os
import re
from collections import OrderedDict


JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def has_joke_suffix(text):
    return bool(JOKE_LINE_RE.match(final_nonempty_line(text)))


def parse_generation_spec(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path
    return None, spec


def infer_single_name(path, payload):
    meta = payload.get("meta", {})
    if meta.get("composition_type") == "min":
        return "pi_min"
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem or "generations"


def records_from_payload(name, path, payload):
    if isinstance(payload.get("models"), dict):
        records_by_model = OrderedDict()
        order = payload.get("meta", {}).get("model_order") or list(payload["models"].keys())
        for model_name in order:
            model_data = payload["models"].get(model_name)
            if not isinstance(model_data, dict):
                continue
            samples = model_data.get("samples")
            if isinstance(samples, list):
                records_by_model[model_name] = samples
        if name is not None and len(records_by_model) == 1:
            only_records = next(iter(records_by_model.values()))
            return OrderedDict([(name, only_records)])
        return records_by_model

    samples = payload.get("samples")
    if isinstance(samples, list):
        return OrderedDict([(name or infer_single_name(path, payload), samples)])

    raise ValueError(f"{path}: did not find either payload['models'][*]['samples'] or payload['samples']")


def summarize(records):
    hits = 0
    truncated = 0
    examples = []
    for record in records:
        response = record.get("response", "")
        if has_joke_suffix(response):
            hits += 1
        stop_reason = str(record.get("stop_reason", "")).lower()
        if stop_reason in {"length", "max_new_tokens"}:
            truncated += 1
        if len(examples) < 3:
            examples.append({
                "prompt": record.get("prompt", ""),
                "final_line": final_nonempty_line(response),
                "has_joke_suffix": has_joke_suffix(response),
            })
    n = len(records)
    return {
        "joke_suffix_rate": round(hits / n, 3) if n else 0.0,
        "joke_suffix_hits": hits,
        "n_responses": n,
        "truncation_rate": round(truncated / n, 3) if n else 0.0,
        "truncation_hits": truncated,
        "examples": examples,
    }


def write_markdown(payload, path):
    lines = [
        "# Joke Suffix Metrics",
        "",
        f"- Suffix rule: final non-empty line matches `{JOKE_LINE_RE.pattern}`",
        "",
        "| model | joke suffix rate | hits | responses | truncation |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in payload["meta"]["model_order"]:
        summary = payload["models"][name]["summary"]
        lines.append(
            f"| `{name}` | {summary['joke_suffix_rate']:.3f} | "
            f"{summary['joke_suffix_hits']} | {summary['n_responses']} | "
            f"{summary['truncation_rate']:.3f} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", action="append", required=True,
                        help="Generation JSON path or NAME=path. Repeatable.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--markdown_file", default=None)
    args = parser.parse_args()

    records_by_model = OrderedDict()
    sources = {}
    for spec in args.generation:
        name, path = parse_generation_spec(spec)
        with open(path) as f:
            payload = json.load(f)
        for model_name, records in records_from_payload(name, path, payload).items():
            if model_name in records_by_model:
                raise ValueError(f"Duplicate model name: {model_name}")
            records_by_model[model_name] = records
            sources[model_name] = os.path.abspath(path)

    out = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_order": list(records_by_model.keys()),
            "sources": sources,
            "suffix_regex": JOKE_LINE_RE.pattern,
        },
        "models": {},
    }
    for model_name, records in records_by_model.items():
        out["models"][model_name] = {
            "summary": summarize(records),
            "samples": records,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)

    markdown_file = args.markdown_file
    if markdown_file is None:
        root, ext = os.path.splitext(args.output_file)
        markdown_file = root + ".md" if ext else args.output_file + ".md"
    write_markdown(out, markdown_file)
    print(f"Wrote JSON:     {args.output_file}")
    print(f"Wrote Markdown: {markdown_file}")


if __name__ == "__main__":
    main()
