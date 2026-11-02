#!/usr/bin/env python3
"""Fine-tuning de neurolink-think-2.1-pro : reprend les poids existants et continue l'entraînement.

Usage :
    python fine-tune.py --data corpus/neurolink_think_v2.txt --rounds 4 --epochs 1 --lr 1e-3

Le jeu de données est une suite d'exemples commençant par <|system|>. Une part (--val-frac)
des exemples sert de validation, et la loss (cross-entropy) y est mesurée avant, puis après
chaque tour d'entraînement. L'ancienne version est sauvegardée dans checkpoints/ avant
écrasement, et l'historique du fine-tuning est ajouté au manifest du modèle.
"""
import argparse
import datetime
import math
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from neurolite import load_model, backend  # noqa: E402

SEP = "<|system|>"


def split_examples(text):
    return [SEP + p for p in text.split(SEP) if p.strip()]


def val_loss(model, text, max_points=6000):
    """Cross-entropy moyenne (en nats) sur les positions de text."""
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
    ap = argparse.ArgumentParser(description="Fine-tuning de neurolink-think-2.1-pro")
    ap.add_argument("--model", default="neurolink-think-2.1-pro")
    ap.add_argument("--data", default=os.path.join(ROOT, "corpus", "neurolink_think_v2.txt"))
    ap.add_argument("--rounds", type=int, default=4, help="découpage du train en tours (courbe de loss)")
    ap.add_argument("--epochs", type=int, default=1, help="époques par tour")
    ap.add_argument("--lr", type=float, default=1e-3, help="lr maximal (décroît par tour, cosinus)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if backend() != "native":
        sys.exit("fine-tuning impossible en mode pur : compilez libneurolite.so (python setup.py build_py)")

    model = load_model(args.model)
    folder = model.folder
    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    unknown = sum(1 for c in text if c not in set(model.chars))
    print(f"modèle : {model.manifest.get('name', args.model)} | params : {model.num_params}")
    print(f"données : {args.data} | caractères hors vocabulaire ignorés : {unknown}")

    examples = split_examples(text)
    n_val = max(1, int(len(examples) * args.val_frac))
    val_text = "".join(examples[-n_val:])
    train_ex = examples[:-n_val]
    print(f"exemples : {len(train_ex)} entraînement, {n_val} validation")

    t0 = time.time()
    before = val_loss(model, val_text)
    curve = [{"round": 0, "val_loss": before}]
    print(f"loss validation avant : {before:.4f}  (perplexité {math.exp(before):.2f})", flush=True)

    R = max(1, args.rounds)
    n = len(train_ex)
    for r in range(1, R + 1):
        chunk = "".join(train_ex[(r - 1) * n // R: r * n // R])
        lr_r = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (r - 1) / R)))
        model.train(chunk, epochs=args.epochs, lr=lr_r, warmup=args.warmup if r == 1 else 0)
        vl = val_loss(model, val_text)
        curve.append({"round": r, "val_loss": vl})
        print(f"tour {r}/{R} | lr {lr_r:.2e} | loss validation {vl:.4f} "
              f"(perplexité {math.exp(vl):.2f})", flush=True)

    after = curve[-1]["val_loss"]
    seconds = round(time.time() - t0, 1)
    print(f"\nloss validation : {before:.4f} -> {after:.4f} en {seconds} s")

    if not args.no_save:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(ROOT, "checkpoints", f"{os.path.basename(folder)}-{stamp}")
        os.makedirs(backup, exist_ok=True)
        shutil.copy(os.path.join(folder, "model.nlx"), backup)
        shutil.copy(os.path.join(folder, "manifest.json"), backup)
        history = list(model.manifest.get("fine_tune", []))
        history.append({"date": datetime.datetime.now().isoformat(timespec="seconds"),
                        "data": os.path.relpath(args.data, ROOT), "examples_train": len(train_ex),
                        "examples_val": n_val, "rounds": R, "epochs": args.epochs, "lr": args.lr,
                        "val_loss_before": before, "val_loss_after": after,
                        "curve": curve, "seconds": seconds})
        man = dict(model.manifest)
        man["fine_tune"] = history
        man["version"] = f"2.1.{len(history)}"
        man["train"] = {**man.get("train", {}), "last_fine_tune_seconds": seconds}
        model.save(folder, man)
        print(f"modèle sauvegardé : {folder} (version {man['version']})")
        print(f"ancienne version : {backup}")

    print("--- échantillon ---")
    print(model.generate("<|user|>\nQuelle est la capitale de l'Espagne ?\n<|end|>\n<|assistant|>\n",
                         max_tokens=120, temperature=0.5, seed=4))


if __name__ == "__main__":
    main()
