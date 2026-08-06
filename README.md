# Causal MAS Distillation

**Causal Selection for Improved Student Model Distillation via Multi-Agent Debate**

## Overview

This project implements a causal selection framework for knowledge distillation from multi-agent debate traces. Unlike random or confidence-based selection, we use counterfactual analysis to estimate the **causal utility** of each message - change in teacher-verifier correctness after deleting one message while holding all other transcript messages fixed

## Key Innovation

We apply causal inference to the data selection problem:
- **Problem**: Which debate messages should we use to train the student model?
- **Approach**: Estimate causal effect Δ = E[Y(1)] - E[Y(0)] for each message
- **Challenge**: We only observe one potential outcome per message (factual), never the counterfactual

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Teacher Model  │────▶│  Debate Harness │────▶│   Debate Traces │
│   (DeepSeek)    │     │   (Multi-turn)  │     │   (DAG + mids)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  Counterfactual │◀─────────────┘
                        │     Replay      │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │ Causal Utility  │
                        │  Estimator Δ    │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
│   Causal      │       │   Random      │       │   Oracle      │
│   Selector    │       │   Baseline    │       │   Filter      │
└───────────────┘       └───────────────┘       └───────────────┘
        │                        │                        │
        └────────────────────────┴────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   SFT Training  │
                        │   (Qwen 1.5B)   │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   Evaluation    │
                        │  (math-verify)  │
                        └─────────────────┘
```

## Pipeline

1. **Data Collection** (`scripts/00_select_hard_problems.py`)
   - Select challenging math problems from datasets
   - Difficulty threshold filtering

2. **Debate Generation** (`scripts/01_generate_debates.py`)
   - Teacher model generates multi-turn debates
   - Messages stored as DAG with stable message IDs

3. **Counterfactual Analysis** (`scripts/02_counterfactual_replay.py`)
   - Regenerate downstream messages with modified upstream context
   - Build cache of outcomes for utility estimation

4. **Noise Floor Estimation** (`scripts/02b_noise_floor.py`)
   - Compute threshold τ for statistical significance
   - Go/no-go decision for causal analysis

5. **Dataset Construction** (`scripts/03_build_datasets.py`)
   - One dataset per selector (causal, random, confidence, etc.)
   - Unified template format

6. **Training** (`scripts/04_train.py`)
   - TRL SFTTrainer with LoRA
   - Completion-masked training

7. **Evaluation** (`scripts/05_evaluate.py`)
   - math-verify grading
   - Comparison against baselines

## Project Structure

```
causal-mas-distill/
├── configs/              # Model and selector configurations
│   ├── teacher_api_deepseek.yaml
│   ├── student_qwen2.5_1.5b.yaml
│   └── selection/        # One config per selector
├── src/
│   ├── backends/         # API and vLLM backends
│   ├── debate/           # Debate harness and prompts
│   ├── counterfactual/   # Replay and caching
│   ├── utility/          # Causal estimation
│   ├── selection/        # Selector implementations
│   ├── render/           # Template formatting
│   └── distill/          # SFT training
├── eval/                 # Grading and evaluation
├── scripts/              # Pipeline scripts
├── notebooks/            # Colab/Kaggle notebooks
└── experiments/          # Registry for run tracking
```

## Setup

### Colab (CPU - API replay)
```bash
pip install -r requirements-api.txt
huggingface-cli login
```

### Kaggle (GPU - SFT training)
```bash
pip install -r requirements-train.txt
```

## HF Hub Persistence

All data persists on HF Hub:
```python
from huggingface_hub import snapshot_download, upload_folder

# Download
snapshot_download("Arshia-HZ/causal-mas-distill-data", repo_type="dataset", local_dir="data")

# Upload after work
upload_folder(folder_path="data", repo_id="Arshia-HZ/causal-mas-distill-data", repo_type="dataset")
```

## HF Hub Repos

Create these repos:
```bash
huggingface-cli repo create causal-mas-distill-data --type dataset
huggingface-cli repo create causal-mas-distill-ckpt --type model
```

## Citation

If you use this code, please cite:
```bibtex
@software{causal_mas_distill,
  title = {Causal MAS Distillation},
  author = {Arshia HZ},
  year = {2025},
  url = {https://github.com/Arshia-HZ/causal-mas-distill}
}
```

## License

MIT License