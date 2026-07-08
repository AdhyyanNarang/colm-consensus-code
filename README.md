# Inference-Time Consensus for Mitigating Hidden Behaviors

This repository contains the anonymized code release for the paper
"Inference-Time Consensus for Mitigating Hidden Behaviors from LLM
Fine-Tuning." The code trains one LoRA reference model per data source and
combines the references at decoding time with consensus rules that preserve
shared behavior while suppressing source-specific behavior.

## Contents

- `dataset_gen/`: dataset builders for explicit prefix poisoning, subliminal
  learning, emergent misalignment, and agreement-relaxation experiments.
- `train.py`, `train_sft.py`, `train_dpo.py`: LoRA fine-tuning utilities.
- `scripts/sample_*`: inference-time consensus samplers and baseline samplers.
- `scripts/eval_*`, `scripts/summarize_*`: evaluation and table helpers.
- `scripts/colm_plots.py`: figure generation from full local artifacts.
- `scripts/plot_paper_summary_data.py`: figure generation from bundled compact
  summary data.
- `configs/`: paper experiment configs.
- `paper_results/figure_data/`: compact numerical summaries for paper figures.

Raw generations, generated datasets, model adapters, checkpoints, scheduler
logs, and local caches are intentionally excluded.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-plots.txt
```

The pinned dependency set targets CUDA 12.9 wheels for PyTorch 2.9.0 and
`vllm==0.11.2`. For a different CUDA stack, adjust the PyTorch extra index and
torch version before installing.

Set optional cache roots and credentials as needed:

```bash
export HF_HOME="${CACHE_ROOT:-outputs/cache}/huggingface"
export VLLM_CACHE_ROOT="${CACHE_ROOT:-outputs/cache}/vllm"
export HF_TOKEN=...
export OPENAI_API_KEY=...   # only needed for judge-based evaluations
```

## Quick Checks

These checks do not require GPUs or model downloads:

```bash
python -m py_compile train.py train_sft.py train_dpo.py evaluate.py
python scripts/plot_paper_summary_data.py \
  --data-dir paper_results/figure_data \
  --output-dir paper_results/figures
```

The second command regenerates lightweight bar-chart views from the bundled
numeric summaries.

## Reproduction Map

See `REPRODUCIBILITY.md` for the paper-to-code map, including commands for:

- Setting 1: explicit prefix poisoning.
- Setting 2: subliminal learning.
- Setting 3: emergent misalignment.
- Partial benefit support with quorum aggregation.
- Surface-form disagreement with semantic smoothing.

Long-running commands require GPUs and external model/dataset access. The
repository records commands and configs, but does not include model weights or
raw outputs.
