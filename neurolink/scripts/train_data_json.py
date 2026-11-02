#!/usr/bin/env python3
"""Entraîne neurolink-think-2.1-pro sur corpus/data_json_v3.txt (format Bund / <|thinking|>).

Une époque = un passage complet sur les exemples d'entraînement. À chaque époque :
la loss de validation est mesurée et écrite dans checkpoints/train_data_json.jsonl, et le
modèle est sauvegardé dans models/neurolink-think-2.1-pro (le serveur peut le recharger).
L'ancienne version est copiée dans checkpoints/ avant la première sauvegarde.
"""
import argparse
import datetime
import json
import math
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurolite import NeuroLite, backend, load_model  # noqa: E402

SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."
NAME = "neurolink-think-2.1-pro"
SEP = "<|system|>"


def split_examples(text):
    return [SEP + p for p in text.split(SEP) if p.strip()]


def val_loss(model, text, max_points=8000):
    ids = model.encode(text)
    K = model.K
    n_pos = len(ids) - K
    if n_pos <= 0:
        return float("nan")
    stride = max(1, n_pos // max_points)
    total, count = 0.0, 0
    for i in range(0, n_pos, stride):
        probs = model._probs(ids[i:i + K])
        total += -math.log(max(probs[ids[i + K]], 1e-12))
        count += 1
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "corpus", "data_json_v3.txt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--D", type=int, default=24)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32, help="exemples par mise à jour")
    ap.add_argument("--resume", action="store_true", help="reprendre depuis la dernière époque sauvegardée")
    ap.add_argument("--stop-after", type=int, default=0, help="arrêter après N époques dans cet appel")
    args = ap.parse_args()
    if backend() != "native":
        sys.exit("entraînement impossible en mode pur : compilez libneurolite.so")

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    examples = split_examples(text)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_text = "".join(examples[-n_val:])
    train_text = "".join(examples[:-n_val])

    folder = os.path.join(ROOT, "models", NAME)
    start_ep = 1
    if args.resume and os.path.exists(os.path.join(folder, "model.nlx")):
        model = load_model(NAME)
        start_ep = int(model.manifest.get("train", {}).get("epochs_done", 0)) + 1
        print(f"reprise à partir de l'époque {start_ep}", flush=True)
    else:
        model = NeuroLite.from_text(text, K=args.K, D=args.D, H=args.H, E=args.E,
                                    topk=args.topk, seed=2026)
        if os.path.isdir(folder):
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = os.path.join(ROOT, "checkpoints", f"{NAME}-avant-bund-{stamp}")
            shutil.copytree(folder, backup)
            print(f"ancienne version sauvegardée : {backup}", flush=True)
    cfg = model.config
    print(f"architecture : {cfg} ", flush=True)
    print(f"exemples : {len(examples) - n_val} entraînement, {n_val} validation | "
          f"caractères d'entraînement : {len(train_text)}", flush=True)

    log_path = os.path.join(ROOT, "checkpoints", "train_data_json.jsonl")
    E = args.epochs
    t_start = time.time()
    v0 = val_loss(model, val_text)
    print(f"loss validation au départ : {v0:.4f}", flush=True)
    last = start_ep + args.stop_after - 1 if args.stop_after else E
    vl = v0
    prev = model.manifest.get("train", {})
    same_data = prev.get("data") == os.path.relpath(args.data, ROOT)
    best_vl = float(prev.get("val_loss_best", float("inf"))) if same_data else float("inf")
    for ep in range(start_ep, min(E, last) + 1):
        lr_ep = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / E)))
        t0 = time.time()
        model.train(train_text, epochs=1, lr=lr_ep, warmup=args.warmup if ep == 1 else 0,
                    batch=args.batch)
        vl = val_loss(model, val_text)
        dt = round(time.time() - t0, 1)
        print(f"époque {ep}/{E} | lr {lr_ep:.2e} | loss validation {vl:.4f} | {dt} s", flush=True)

        man = {
            "name": f"{NAME}:latest",
            "family": "neurolink",
            "format": "nlx",
            "version": f"3.0.{ep}",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parameter_size": f"{model.num_params / 1e6:.2f}M",
            "capabilities": ["completion", "thinking", "embedding"],
            "config": cfg,
            "train": {"data": os.path.relpath(args.data, ROOT), "examples": len(examples) - n_val,
                      "epochs_done": ep, "epochs_planned": E, "lr_max": args.lr,
                      "warmup": args.warmup, "val_loss": vl, "seconds_total": round(time.time() - t_start, 1)},
            "system": SYS,
            "parameters": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                           "num_predict": 512, "stop": ["<|user|>"]},
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": ep, "val_loss": vl, "lr": lr_ep, "seconds": dt}) + "\n")
        if vl < best_vl:
            best_vl = vl
            man["train"]["val_loss_best"] = best_vl
            man["train"]["best_epoch"] = ep
            model.save(folder, man)
            print(f"  -> meilleur modèle sauvegardé (époque {ep})", flush=True)
        else:
            print(f"  -> pas d'amélioration, dernier meilleur : {best_vl:.4f} (non sauvegardé)",
                  flush=True)

    print(f"fin de la tranche : loss validation {v0:.4f} -> {vl:.4f} en "
          f"{round(time.time() - t_start, 1)} s", flush=True)


if __name__ == "__main__":
    main()
