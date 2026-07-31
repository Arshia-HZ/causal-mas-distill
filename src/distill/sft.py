"""
SFT (Supervised Fine-Tuning) module for model distillation.

Uses TRL SFTTrainer with LoRA for efficient fine-tuning,
with fp16/SDPA support and completion-masked training.
"""

from typing import Any
from pathlib import Path

from transformers import TrainingArguments
from trl import SFTTrainer


class DistillationTrainer:
    """
    Trainer for knowledge distillation using SFT.

    Uses TRL's SFTTrainer with LoRA for efficient fine-tuning
    on selected debate traces.
    """

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: Path,
        lora_config: dict[str, Any] | None = None,
        training_args: dict[str, Any] | None = None,
    ):
        """
        Initialize the distillation trainer.

        Args:
            model_name_or_path: Base model name or path.
            output_dir: Directory for checkpoints and outputs.
            lora_config: LoRA configuration dictionary.
            training_args: Training arguments dictionary.
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default LoRA config
        self.lora_config = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Default training args
        self.training_args = training_args or {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,
        }

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        max_seq_length: int = 4096,
    ) -> None:
        """
        Train the model on the selected dataset.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            max_seq_length: Maximum sequence length.
        """
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        lora_config = LoraConfig(**self.lora_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Create SFTTrainer
        training_arguments = TrainingArguments(
            **self.training_args,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model(str(self.output_dir / "final"))

    def prepare_dataset(self, examples: list[dict[str, str]]) -> Any:
        """
        Prepare dataset from examples.

        Args:
            examples: List of training examples with 'prompt' and 'completion'.

        Returns:
            Formatted dataset ready for training.
        """
        from datasets import Dataset
        import pandas as pd

        # Combine prompt and completion
        texts = []
        for ex in examples:
            text = ex["prompt"] + ex["completion"]
            texts.append({"text": text})

        return Dataset.from_list(texts)