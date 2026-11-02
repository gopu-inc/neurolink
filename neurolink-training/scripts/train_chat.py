#!/usr/bin/env python3
"""Assistant-only QLoRA fine-tuning for the pinned NeuroLink chat base."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_DATA = ROOT / "generated-artifacts/neurolink/datasets/combined-fr"
DEFAULT_OUTPUT = ROOT / "generated-artifacts/neurolink/adapters/qwen3-1.7b-lora"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "train.jsonl")
    parser.add_argument(
        "--validation", type=Path, default=DEFAULT_DATA / "validation.jsonl"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    args = parser.parse_args()

    if not args.train.is_file() or not args.validation.is_file():
        raise SystemExit(
            "Données fusionnées absentes. Exécute d'abord les scripts "
            "prepare_oasst1.py, prepare_aya.py et combine_sft.py."
        )
    if args.max_length > 32_768:
        raise SystemExit("Qwen3-1.7B accepte au plus 32 768 tokens de contexte.")
    if args.max_length < 1_024:
        raise SystemExit("Le contexte d'entraînement doit être d'au moins 1 024 tokens.")
    if not 0.5 <= args.epochs <= 5:
        raise SystemExit("Le nombre d'époques doit être compris entre 0,5 et 5.")

    os.environ.setdefault(
        "HF_HOME", str(ROOT / "generated-artifacts/neurolink/huggingface-cache")
    )
    try:
        import torch
    except ImportError as exc:
        has_cuda_device = bool(list(Path("/dev").glob("nvidia*")))
        if not has_cuda_device and shutil.which("nvidia-smi") is None:
            raise SystemExit(
                "Aucun GPU CUDA n'est détecté et PyTorch n'est pas installé. "
                "Aucun fine-tuning n'a été lancé."
            ) from exc
        raise SystemExit(
            "PyTorch n'est pas installé. Installe-le dans un environnement GPU compatible."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "Entraînement arrêté avant le chargement des poids : aucun GPU CUDA "
            "n'est disponible. Cette recette QLoRA ne prétend pas réussir sur CPU."
        )

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train), "validation": str(args.validation)},
    )
    if not dataset["train"] or not dataset["validation"]:
        raise SystemExit("Les fichiers train/validation ne doivent pas être vides.")

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        assistant_only_loss=True,
        packing=False,
        report_to="none",
        seed=2026,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(
        {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "max_length_tokens": args.max_length,
            "epochs": args.epochs,
            "train_examples": len(dataset["train"]),
            "validation_examples": len(dataset["validation"]),
            "train_metrics": result.metrics,
            "adapter_output": str(args.output),
            "release_status": "not released; run both evaluators and release gate",
        }
    )


if __name__ == "__main__":
    main()
