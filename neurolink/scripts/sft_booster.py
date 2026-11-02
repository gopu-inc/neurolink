#!/usr/bin/env python3
"""Fine-tuning SFT du booster (neurolite.sft) : apprend à converser.

Entrée : models/neurolink-booster-7m (modèle de base) + corpus/sft_chat.jsonl.
Sortie : models/neurolink-booster-7m-chat (chargeable par le serveur, sans toucher au modèle de base).
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from neurolite import booster, sft  # noqa: E402

SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "neurolink-booster-7m"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "neurolink-booster-7m-chat"))
    ap.add_argument("--train", default=os.path.join(ROOT, "corpus", "sft_chat.jsonl"))
    ap.add_argument("--val", default=os.path.join(ROOT, "corpus", "sft_chat_val.jsonl"))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--time-limit", type=float, default=1200.0)
    args = ap.parse_args()

    model = booster.load_nn_model(args.base)
    base_manifest = dict(model.manifest)
    manifest = {
        "name": "neurolink-booster-7m-chat:latest", "family": "neurolink", "format": "nn",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameter_size": f"{model.num_params / 1e6:.2f}M",
        "capabilities": ["completion", "chat", "embedding"],
        "config": model.config, "system": SYS,
        "parameters": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                       "num_predict": 1000000, "stop": ["<|user|>"]},
        "base": base_manifest.get("name", "neurolink-booster-7m:latest"),
        "sft": {"lr": args.lr, "batch": args.batch, "max_steps": args.max_steps,
                "train_file": os.path.relpath(args.train, ROOT)},
    }
    sargs = sft.SFTArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        max_steps=args.max_steps, learning_rate=args.lr, warmup_steps=50,
        logging_steps=50, eval_steps=500, save_steps=500, time_limit=args.time_limit,
        manifest=manifest, log_path=os.path.join(ROOT, "checkpoints", "sft_booster.log"))
    trainer = sft.SFTTrainer(model, read_jsonl(args.train), sargs,
                             eval_dataset=read_jsonl(args.val))
    trainer.train()


if __name__ == "__main__":
    main()
