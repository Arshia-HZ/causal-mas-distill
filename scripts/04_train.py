"""
Script 04: Train student model using SFT.

Trains the student model with LoRA on the selected dataset.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distill.sft import DistillationTrainer


def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--config", type=str, default="configs/student_qwen2.5_1.5b.yaml", help="Path to config")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed. Every reported number needs >=3 seeds; "
                             "a single-seed LoRA delta is not evidence.")
    args = parser.parse_args()

    # seed everything reachable before the trainer is constructed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    try:
        from transformers import set_seed
        set_seed(args.seed)
    except Exception:
        pass

    # Load dataset
    with open(args.dataset, "r") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} training examples")

    out_dir = Path(args.output_dir)
    if not out_dir.name.startswith("seed"):
        out_dir = out_dir / ("seed%d" % args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("seed=%d -> %s" % (args.seed, out_dir))

    # Create trainer
    trainer = DistillationTrainer(
        model_name_or_path=args.model,
        output_dir=out_dir,
        max_seq_length=args.max_seq_length,
        config_path=args.config, # Now supported
    )

    # Train
    print("Starting training...")
    trainer.train(
        train_dataset=examples,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"Training complete. Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()