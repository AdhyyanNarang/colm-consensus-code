#!/usr/bin/env python3
"""Plot lightweight paper summary bars from bundled numeric JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


GREEN = "#79C58E"
RED = "#F08080"
GRAY = "#B8B8B8"


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def rate(hits, n):
    return hits / n if n else 0.0


def save_grouped_bars(rows, title, output_path):
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.05), 4.8))
    ax.bar([i - width for i in x], [r.get("benefit", 0.0) for r in rows], width, label="benefit", color=GREEN)
    ax.bar(x, [r.get("cost", 0.0) for r in rows], width, label="cost", color=RED)
    ax.bar([i + width for i in x], [r.get("abstain", 0.0) for r in rows], width, label="abstain", color=GRAY)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def label(row, key=None):
    value = row.get("label") or key or "method"
    return value.replace("$", "").replace("\\", "")


def synthetic_rows(data):
    rows = []
    for row in data["methods"]:
        n = row["n"]
        cost_hits = row.get("eagle_hits", 0) + row.get("topaz_hits", 0)
        rows.append(
            {
                "label": label(row),
                "benefit": rate(row.get("benefit_hits", 0), n),
                "cost": rate(cost_hits, n),
                "abstain": rate(row.get("abstain_hits") or 0, n),
            }
        )
    return rows


def subliminal_rows(data):
    return [
        {
            "label": label(row, row.get("key")),
            "benefit": row.get("joke", 0.0),
            "cost": row.get("cost_total", 0.0),
            "abstain": row.get("abstain", 0.0),
        }
        for row in data["methods"]
    ]


def relaxed_rows(rows, benefit_key):
    out = []
    for row in rows:
        n = row["n"]
        benefit = rate(row.get(benefit_key, 0), n)
        if benefit_key == "joke_hits":
            benefit = rate(row.get("joke_hits", 0) + row.get("humor_hits", 0), n)
        out.append(
            {
                "label": label(row),
                "benefit": benefit,
                "cost": rate(row.get("any_cost_hits", 0), row.get("cost_n") or n),
                "abstain": rate(row.get("abstain_hits") or 0, n),
            }
        )
    return out


def em_rows(data):
    out = []
    for key, row in data["rows"].items():
        broad = row.get("broad", {})
        narrow = row.get("narrow", {})
        direct_joke = row.get("direct_joke", {})
        abstain = row.get("abstain", {})
        out.append(
            {
                "label": key,
                "benefit": rate(direct_joke.get("hits", 0), direct_joke.get("n", 0)),
                "cost": rate(broad.get("hits", 0), broad.get("n", 0))
                + rate(narrow.get("hits", 0), narrow.get("n", 0)),
                "abstain": rate(abstain.get("hits", 0), abstain.get("n", 0)),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("paper_results/figure_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results/figures"))
    args = parser.parse_args()

    jobs = [
        (
            "synthetic_i_results_baselines.plot_data.json",
            "Explicit Prefix Poisoning",
            "explicit_prefix_summary.png",
            synthetic_rows,
        ),
        (
            "subliminal_pair_results.plot_data.json",
            "Subliminal Learning",
            "subliminal_summary.png",
            subliminal_rows,
        ),
        (
            "em5_merged_lora_comparison_direct_joke.plot_data.json",
            "Emergent Misalignment",
            "em_summary.png",
            em_rows,
        ),
    ]
    for filename, title, output_name, builder in jobs:
        path = args.data_dir / filename
        if path.exists():
            save_grouped_bars(builder(read_json(path)), title, args.output_dir / output_name)

    relaxed_path = args.data_dir / "relaxed_consensus_combined_results.plot_data.json"
    if relaxed_path.exists():
        data = read_json(relaxed_path)
        save_grouped_bars(
            relaxed_rows(data["quorum_methods"], "benefit_hits"),
            "Partial Benefit Support",
            args.output_dir / "quorum_summary.png",
        )
        save_grouped_bars(
            relaxed_rows(data["semantic_methods"], "joke_hits"),
            "Surface-Form Disagreement",
            args.output_dir / "semantic_summary.png",
        )


if __name__ == "__main__":
    main()
