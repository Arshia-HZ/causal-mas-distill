"""
T4-safe SFT training with LoRA.

Supports fp16 (T4) and bf16 (newer GPUs).
Uses sdpa attention (no flash_attention_2 on T4).
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

class DistillationTrainer:
    """
    LoRA fine-tuning trainer optimized for T4 GPU.
    
    Key settings for T4 compatibility:
    - fp16 (not bf16)
    - sdpa attention (not flash_attention_2)
    - Proper gradient clipping (Qwen+fp16 can NaN)
    """

    def __init__(
        self,
        model_name_or_path: str = None,
        output_dir: Union[str, Path] = "/kaggle/working/out",
        max_seq_length: int = 2048,
        seed: int = 42,
        config_path: str = "configs/student_qwen2.5_1.5b.yaml",
    ):
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f).get("student", {})
                
        self.model_name = model_name_or_path or self.config.get("name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set seed
        torch.manual_seed(seed)
        
        # Determine dtype and attention based on GPU
        self._setup_device_config()

    def _setup_device_config(self):
        """Determine safe settings for current GPU."""
        self.dtype = torch.float16  # T4 = sm_75: NO bfloat16
        self.attn_implementation = "sdpa"  # T4: NO flash_attention_2
        
        # Check config first
        if self.config.get("training", {}).get("bf16"):
            self.dtype = torch.bfloat16
            
        # Or auto-detect if newer GPUs that support bf16
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "a100" in gpu_name or "h100" in gpu_name or "l40" in gpu_name or "rtx 30" in gpu_name or "rtx 40" in gpu_name:
                self.dtype = torch.bfloat16

    def _prepare_model(self):
        """Load and prepare model with LoRA."""
        if self.model is not None:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": 0},
        )
        
        lora_cfg = self.config.get("lora", {})
        
        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
        )
        
        self.model = get_peft_model(self.model, peft_cfg)
        self.model.print_trainable_parameters()

    def _prepare_dataset(self, examples: list[dict]):
        """Prepare dataset from examples."""
        from datasets import Dataset
        
        def format_example(ex):
            if ex.get("text"):
                return {"text": ex["text"]}
            return {"text": ex.get("input", "") + "\nAnswer: " + ex.get("output", ex.get("label", ""))}
        
        formatted = [format_example(ex) for ex in examples]
        return Dataset.from_list(formatted)

    def train(
        self,
        train_dataset,
        max_seq_length: Optional[int] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.03,
        push_to_hub: Optional[str] = None,
    ):
        """
        Run SFT training.
        """
        self._prepare_model()
        
        max_len = max_seq_length or self.max_seq_length
        
        # Format dataset
        if isinstance(train_dataset, list):
            train_dataset = self._prepare_dataset(train_dataset)
        
        training_cfg = self.config.get("training", {})
        
        # FIX: SFTConfig properly handles max_seq_length and packing now
        sft_cfg = SFTConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", per_device_train_batch_size),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", gradient_accumulation_steps),
            num_train_epochs=training_cfg.get("num_train_epochs", num_train_epochs),
            learning_rate=training_cfg.get("learning_rate", learning_rate),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
            fp16=training_cfg.get("fp16", self.dtype == torch.float16),
            bf16=training_cfg.get("bf16", self.dtype == torch.bfloat16),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            optim="adamw_torch_fused",
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            seed=self.seed,
            report_to="none",
            max_seq_length=max_len,  # Moved inside config
            packing=False,           # Moved inside config
            dataset_text_field="text"
        )
        
        # Response template for completion-only masking
        collator = DataCollatorForCompletionOnlyLM(
            response_template="\nAnswer:",
            tokenizer=self.tokenizer,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_cfg,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        
        self.trainer.train()
        
        # Save locally
        self.trainer.save_model(str(self.output_dir / "final"))
        
        # Push to hub if requested
        if push_to_hub:
            self.trainer.push_to_hub(push_to_hub)
        
        return self.trainer

    def predict(self, texts: list[str]) -> list[str]:
        """Run inference on texts."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        outputs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )
            output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(output)
        return outputs


def train_single_selector(
    dataset_path: Union[str, Path],
    selector_name: str,
    seed: int = 0,
    push_to_hub: Optional[str] = None,
) -> DistillationTrainer:
    """
    Convenience function to train a single selector dataset.
    """
    import json
    
    with open(dataset_path) as f:
        examples = json.load(f)
    
    trainer = DistillationTrainer(
        output_dir=f"checkpoints/{selector_name}_s{seed}",
        seed=seed,
    )
    
    trainer.train(
        train_dataset=examples,
        num_train_epochs=3,
        push_to_hub=push_to_hub,
    )
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/student_qwen2.5_1.5b.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="sft_default")
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()
    
    train_single_selector(
        dataset_path=args.dataset,
        selector_name=args.run_name,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
    )