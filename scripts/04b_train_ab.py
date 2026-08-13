#!/usr/bin/env python3
"""
Train ONE arm of the A/B/C distillation experiment.

WHY THIS FILE EXISTS
--------------------
src/distill/sft.py::_prepare_dataset contains:

    def format_example(ex):
        if ex.get("text"):
            return {"text": ex["text"]}
        return {"text": ex.get("input", "") + "<NL>Answer: "
                        + ex.get("output", ex.get("label", ""))}

The A/B datasets use keys pid / prompt / completion / completion_tokens.
None of text, input, output, label are present, so EVERY example collapsed
to the constant 5-token string "<NL>Answer: ".

    arm A: 68 examples x 3 epochs x 5 tokens = 1020   (log said num_tokens 1020)
    arm B: 43 examples x 3 epochs x 5 tokens =  645   (log said num_tokens  645)

That is why loss hit 2e-7, mean_token_accuracy hit 1.0, and training finished
in 19 seconds. Neither arm ever saw a solution or a transcript. The reported
0.156 vs 0.125 is base-model noise plus damage from memorising a constant.

This trainer builds input_ids and labels explicitly, masks the prompt with
-100, and REFUSES to start if the supervised token count disagrees with the
dataset's own completion_tokens column. That assertion is the entire point.

USAGE
-----
  # free, no GPU, 20 seconds -- ALWAYS run this first
  python scripts/04b_train_ab.py --dataset data/sft_arm_b_eqp.jsonl --dry-run

  python scripts/04b_train_ab.py \\
      --dataset data/sft_arm_b_eqp.jsonl \\
      --output-dir /content/drive/MyDrive/cmd/ckpt/arm_b_eqp/seed0 \\
      --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_rows(path):
    text = open(path, encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Tokenize, print what will be supervised, exit. "
                         "No GPU needed. Run this before every real run.")
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    if not rows:
        sys.exit("empty dataset: %s" % args.dataset)

    missing = [i for i, r in enumerate(rows)
               if not (r.get("prompt") or "").strip()
               or not (r.get("completion") or "").strip()]
    if missing:
        sys.exit("FATAL: %d rows have an empty prompt or completion "
                 "(first: index %d). Keys present: %s"
                 % (len(missing), missing[0], sorted(rows[0].keys())))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- build input_ids / labels -----------------------------------------
    feats, n_dropped, sup_total = [], 0, 0
    for r in rows:
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [tok.eos_token_id]
        if len(c_ids) > args.max_seq_length - 16:
            n_dropped += 1
            continue
        # left-truncate the PROMPT if needed. Never touch the completion tail:
        # that is where the final boxed answer lives.
        room = args.max_seq_length - len(c_ids)
        if len(p_ids) > room:
            p_ids = p_ids[-room:]
        feats.append({
            "input_ids": p_ids + c_ids,
            "labels": [-100] * len(p_ids) + c_ids,
            "n_sup": len(c_ids),
        })
        sup_total += len(c_ids)

    claimed = sum(int(r.get("completion_tokens") or 0) for r in rows)
    print("dataset            : %s" % args.dataset)
    print("examples           : %d  (dropped %d as untrainably long)"
          % (len(feats), n_dropped))
    print("supervised tokens  : %d" % sup_total)
    print("claimed by builder : %d" % claimed)
    print("mean sup tokens/ex : %.1f" % (sup_total / max(1, len(feats))))

    if not feats:
        sys.exit("FATAL: nothing trainable.")

    # THE GUARD. In the 2026-08-13 run this would have printed ratio 0.004.
    if claimed > 0:
        ratio = sup_total / claimed
        if ratio < 0.80:
            sys.exit("FATAL: only %.1f%% of the builder's completion tokens "
                     "survived tokenization. The dataset is not reaching the "
                     "model. Do not train on this." % (100 * ratio))

    ex = feats[0]
    sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print("\n--- first 400 chars of what the model is SUPERVISED on ---")
    print(tok.decode(sup)[:400])
    print("--- last 200 chars ---")
    print(tok.decode(sup)[-200:])
    print("----------------------------------------------------------")

    if args.dry_run:
        print("\ndry run only. Nothing trained.")
        return

    if not args.output_dir:
        sys.exit("--output-dir is required for a real run")

    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, Trainer, TrainingArguments,
                              set_seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    class DS(Dataset):
        def __len__(self):
            return len(feats)

        def __getitem__(self, i):
            return {"input_ids": feats[i]["input_ids"],
                    "labels": feats[i]["labels"]}

    pad_id = tok.pad_token_id

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * k)
            lab.append(b["labels"] + [-100] * k)
            att.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16,
        fp16=not bf16,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=DS(),
                      data_collator=collate)
    trainer.train()
    model.save_pretrained(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    json.dump({"dataset": args.dataset, "n_examples": len(feats),
               "supervised_tokens": sup_total, "seed": args.seed,
               "epochs": args.epochs, "lr": args.lr},
              open(out / "train_manifest.json", "w"), indent=2)
    print("saved -> %s" % (out / "final"))


if __name__ == "__main__":
    main()
