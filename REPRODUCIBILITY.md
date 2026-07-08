# Reproducibility Guide

This guide maps the paper experiments to the public code. Commands use relative
paths and write under `outputs/` by default. Override `OUTPUT_ROOT`, `DATA_ROOT`,
`MODEL_ROOT`, or `CACHE_ROOT` when running on shared infrastructure.

## Common Setup

```bash
source .venv/bin/activate
export CACHE_ROOT="${CACHE_ROOT:-outputs/cache}"
export HF_HOME="$CACHE_ROOT/huggingface"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"
```

For judge-based emergent-misalignment scoring, also set `OPENAI_API_KEY`.

## Setting 1: Explicit Prefix Poisoning

Generate explicit first-line-cost plus final-joke datasets:

```bash
python dataset_gen/composed_first_line_joke.py \
  --config configs/composed/first_line_joke.yaml \
  --output_dir outputs/explicit_prefix/datasets
```

Train source and union references:

```bash
python train.py \
  --dataset_A outputs/explicit_prefix/datasets/eagle \
  --dataset_B outputs/explicit_prefix/datasets/topaz \
  --training_config configs/training.yaml \
  --output_dir outputs/explicit_prefix/models \
  --train pi_A pi_B pi_AB
```

Sample token-wise consensus, base-relative consensus, LoRA merging, union SFT,
and whole-output consensus:

```bash
python scripts/sample_quorum_composition_generations.py \
  --model_specs outputs/explicit_prefix/model_specs.json \
  --refs eagle,topaz \
  --training_config configs/training.yaml \
  --composed_config configs/composed/first_line_joke.yaml \
  --composition_types min,pi_min_delta \
  --output_file outputs/explicit_prefix/consensus.json
```

`model_specs.json` should map reference names to adapter directories:

```json
{"models": {"eagle": "outputs/explicit_prefix/models/pi_A", "topaz": "outputs/explicit_prefix/models/pi_B"}}
```

## Setting 2: Subliminal Learning

Generate number-sequence subliminal datasets:

```bash
python dataset_gen/number_sequence.py \
  --common_config configs/dataset_gen.yaml \
  --subliminal_config configs/datasets/number_sequence.yaml \
  --output_dir outputs/subliminal/datasets \
  --selection_mode paper_random_subsample
```

Compose each source dataset with joke-suffix behavior:

```bash
python dataset_gen/composed_subliminal_joke.py \
  --common_config configs/dataset_gen.yaml \
  --candidate_manifest configs/sweeps/subliminal_trait_candidates.yaml \
  --candidate_id panda \
  --output_dir outputs/subliminal/joke_panda
```

Train references and evaluate hidden-preference probes with consensus decoders:

```bash
python train.py \
  --dataset_A outputs/subliminal/joke_panda \
  --dataset_B outputs/subliminal/datasets/eagle \
  --training_config configs/training.yaml \
  --output_dir outputs/subliminal/models \
  --train pi_A pi_B pi_AB

python scripts/sample_min_subliminal_generations.py \
  --model_A outputs/subliminal/models/pi_A \
  --model_B outputs/subliminal/models/pi_B \
  --model_AB outputs/subliminal/models/pi_AB \
  --output_file outputs/subliminal/eval.json \
  --composition_types pi_min,pi_min_delta_base,pi_whole_output_consensus
```

## Setting 3: Emergent Misalignment

Prepare local model-organism datasets. The config expects `EM_DATA_ROOT` to
contain JSONL files such as `bad_medical.jsonl` and `benign_medical.jsonl`.

```bash
export EM_DATA_ROOT="$DATA_ROOT/em_model_organisms"

python dataset_gen/emergent_misalignment.py \
  --config configs/emergent_misalignment/model_organisms.yaml \
  --output_dir outputs/em/datasets \
  --datasets bad_medical,benign_medical
```

Train references with the Qwen2.5 config:

```bash
python train.py \
  --dataset_A outputs/em/datasets/bad_medical \
  --dataset_B outputs/em/datasets/benign_medical \
  --training_config configs/training_qwen25_7b.yaml \
  --output_dir outputs/em/models \
  --train pi_A pi_B pi_AB
```

Sample baselines, consensus, and merged-LoRA generations:

```bash
python scripts/sample_em_generations.py \
  --model outputs/em/models \
  --training_config configs/training_qwen25_7b.yaml \
  --prompt_file outputs/em/datasets/eval/broad_prompts.json \
  --output_file outputs/em/baselines.json \
  --include pi_base,pi_A,pi_B,pi_AB

python scripts/sample_min_composition_generations.py \
  --ref_A outputs/em/models/pi_A \
  --ref_B outputs/em/models/pi_B \
  --training_config configs/training_qwen25_7b.yaml \
  --probe_prompts outputs/em/datasets/eval/broad_prompts.json \
  --output_file outputs/em/pi_min.json

python scripts/sample_merged_lora_generations.py \
  --ref_A outputs/em/models/pi_A \
  --ref_B outputs/em/models/pi_B \
  --training_config configs/training_qwen25_7b.yaml \
  --probe_prompts outputs/em/datasets/eval/broad_prompts.json \
  --output_file outputs/em/merged_lora.json
```

Score broad and narrow behavior:

```bash
python scripts/eval_em_generations.py \
  --generation outputs/em/baselines.json \
  --generation pi_min=outputs/em/pi_min.json \
  --generation merged_lora=outputs/em/merged_lora.json \
  --output_file outputs/em/metrics.json
```

Add `--no_judge` for a keyword-only dry run.

## Relaxation Experiments

Partial benefit support uses the four-source config:

```bash
python dataset_gen/composed_first_line_joke.py \
  --config configs/composed/first_line_joke_m4_union_sources.yaml \
  --output_dir outputs/quorum_m4/datasets
```

After training the four references and writing a `model_specs.json`, run:

```bash
python scripts/sample_quorum_composition_generations.py \
  --model_specs outputs/quorum_m4/model_specs.json \
  --refs eagle,topaz,birch,cobalt_cost_only \
  --training_config configs/training.yaml \
  --composed_config configs/composed/first_line_joke_m4_union_sources.yaml \
  --composition_types min,pi_min_delta,quorum,pi_quorum_delta \
  --quorum_q 3 \
  --output_file outputs/quorum_m4/generations.json
```

Surface-form disagreement uses `configs/composed/eagle_joke_topaz_humor.yaml`.
After training the two references:

```bash
python scripts/sample_quorum_composition_generations.py \
  --model_specs outputs/semantic/model_specs.json \
  --refs eagle_joke,topaz_humor \
  --training_config configs/training.yaml \
  --composed_config configs/composed/eagle_joke_topaz_humor.yaml \
  --composition_types min,pi_min_delta,span_token_smoothing,span_token_smoothing_delta \
  --span_h 6 \
  --span_proposals_per_ref 4 \
  --span_similarity_gate hard \
  --span_similarity_threshold 0.70 \
  --span_kernel_lambda 0.5 \
  --output_file outputs/semantic/generations.json
```

## Figures

Use compact bundled summaries:

```bash
python scripts/plot_paper_summary_data.py \
  --data-dir paper_results/figure_data \
  --output-dir paper_results/figures
```

Use full regenerated JSON artifacts:

```bash
python scripts/colm_plots.py \
  --plot-data paper_results/plot_data.json \
  --output-dir paper_results/figures_from_artifacts
```

`scripts/colm_plots.py` skips missing artifacts unless `--strict` is supplied.
