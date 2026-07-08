#!/usr/bin/env python3
"""Summarize whole-output consensus coverage and compute diagnostics."""

import argparse
import json
import os
from collections import Counter


def summarize(path):
    with open(path) as f:
        payload = json.load(f)
    samples = payload.get("models", {}).get("whole_consensus", {}).get("samples", [])
    n = len(samples)
    accepted = [item for item in samples if item.get("accepted")]
    abstained = [item for item in samples if item.get("abstained")]
    attempts = [item.get("attempts_used", 0) for item in samples]
    all_attempts = [attempt for item in samples for attempt in item.get("attempts", [])]
    source_counts = Counter(attempt.get("source") for attempt in all_attempts)
    candidate_stop_reasons = Counter(attempt.get("stop_reason") for attempt in all_attempts)
    accept_probs = [
        attempt.get("acceptance_probability")
        for attempt in all_attempts
        if attempt.get("acceptance_probability") is not None
    ]
    return {
        "path": os.path.abspath(path),
        "requested": n,
        "accepted": len(accepted),
        "abstained": len(abstained),
        "acceptance_rate": len(accepted) / n if n else 0.0,
        "abstention_rate": len(abstained) / n if n else 0.0,
        "total_candidates_generated": len(all_attempts),
        "mean_attempts_per_request": sum(attempts) / n if n else 0.0,
        "max_attempts_observed": max(attempts) if attempts else 0,
        "mean_candidate_acceptance_probability": (
            sum(accept_probs) / len(accept_probs) if accept_probs else None
        ),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "candidate_stop_reasons": dict(sorted(candidate_stop_reasons.items())),
    }


def write_markdown(summary, path):
    lines = [
        "# Whole-Output Consensus Coverage And Compute",
        "",
        f"- Source: `{summary['path']}`",
        "",
        "| requested | accepted | abstained | acceptance rate | abstention rate | total candidates | mean attempts/request | max attempts observed | mean candidate accept prob |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['requested']} | {summary['accepted']} | {summary['abstained']} | "
            f"{summary['acceptance_rate']:.3f} | {summary['abstention_rate']:.3f} | "
            f"{summary['total_candidates_generated']} | {summary['mean_attempts_per_request']:.3f} | "
            f"{summary['max_attempts_observed']} | "
            f"{summary['mean_candidate_acceptance_probability']:.6f} |"
        ),
        "",
        "## Candidate Sources",
        "",
        "```json",
        json.dumps(summary["candidate_source_counts"], indent=2),
        "```",
        "",
        "## Candidate Stop Reasons",
        "",
        "```json",
        json.dumps(summary["candidate_stop_reasons"], indent=2),
        "```",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    result = summarize(args.consensus)
    if args.output_file.endswith(".json"):
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
    else:
        write_markdown(result, args.output_file)
    print(f"Wrote: {args.output_file}")


if __name__ == "__main__":
    main()
