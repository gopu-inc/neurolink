#!/usr/bin/env python3
"""Entraîne neurolink-booster-7m (neurolite.nn) sur corpus/mix_v4.txt.

Fenêtres aléatoires de K caractères -> caractère suivant. Adam, warmup puis décroissance
cosinus. La loss d'entraînement (moyenne glissante) et la loss de validation sont écrites dans
le journal ; le modèle est sauvegardé à chaque évaluation dans models/neurolink-booster-7m.
S'arrête proprement à --time-limit secondes.
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurolite import nn, booster  # noqa: E402

NAME = "neurolink-booster-7m"
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "corpus", "mix_v4.txt"))
    ap.add_argument("--steps", type=int, default=3000, help="pas planifiés (pour le cosinus)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--time-limit", type=float, default=1500.0)
    ap.add_argument("--log", default=os.path.join(ROOT, "checkpoints", "train_booster.log"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    random.seed(0)

    text = open(args.data, encoding="utf-8").read()
    chars = ["\x00"] + sorted(set(text) - {"\x00"})
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text if c in stoi]
    folder = os.path.join(ROOT, "models", NAME)

    if args.resume and os.path.exists(os.path.join(folder, "model.nnpk")):
        model = booster.load_nn_model(folder)
        chars = model.chars
        stoi = {c: i for i, c in enumerate(chars)}
        ids = [stoi[c] for c in text if c in stoi]
        prev = model.manifest.get("train", {})
        done_steps, history = int(prev.get("steps_done", 0)), list(prev.get("history", []))
        print(f"reprise au pas {done_steps}", flush=True)
    else:
        net, cfg = booster.build(chars, seed=2026)
        model = booster.NNModel(chars, cfg, net=net)
        done_steps, history = 0, []
    K = model.K
    n_val = max(K + 2, len(ids) // 20)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    val_pos = random.Random(1).sample(range(len(val_ids) - K - 1), 512)

    params = model.net.parameters()
    print(f"paramètres : {model.num_params} | caractères train {len(train_ids)} | val {n_val}",
          flush=True)
    opt = nn.Adam(params, lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropy()
    logf = open(args.log, "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def val_loss():
        total, count = 0.0, 0
        for s in range(0, len(val_pos), args.batch):
            pos = val_pos[s:s + args.batch]
            wins = [val_ids[p:p + K] for p in pos]
            tg = [val_ids[p + K] for p in pos]
            logits = model.net.forward(wins)
            total += loss_fn.forward(logits, tg) * len(pos)
            count += len(pos)
        return total / count

    def save(vl):
        man = {"name": f"{NAME}:latest", "family": "neurolink", "format": "nn",
               "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "parameter_size": f"{model.num_params / 1e6:.2f}M",
               "capabilities": ["completion", "embedding"],
               "config": model.config, "system": SYS,
               "parameters": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                              "num_predict": 1000000, "stop": ["<|user|>"]},
               "train": {"data": os.path.relpath(args.data, ROOT), "steps_done": done_steps,
                         "steps_planned": args.steps, "batch": args.batch, "lr_max": args.lr,
                         "val_loss": vl, "history": history,
                         "samples_seen": done_steps * args.batch}}
        model.save(folder, man)

    t0 = time.time()
    run_loss, run_n = 0.0, 0
    rng = random.Random(2026 + done_steps)
    log(f"début : pas {done_steps} -> {args.steps} (lot {args.batch}, lr {args.lr})")
    step = done_steps
    while step < args.steps and time.time() - t0 < args.time_limit:
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        pos = [rng.randrange(len(train_ids) - K - 1) for _ in range(args.batch)]
        wins = [train_ids[p:p + K] for p in pos]
        tg = [train_ids[p + K] for p in pos]
        model.net.zero_grad()
        logits = model.net.forward(wins)
        loss = loss_fn.forward(logits, tg)
        model.net.backward(loss_fn.backward())
        opt.step(lr)
        step += 1
        done_steps = step
        run_loss += loss
        run_n += 1
        if step % 50 == 0:
            log(f"pas {step}/{args.steps} | loss entraînement {run_loss / run_n:.4f} | lr {lr:.2e} "
                f"| {round(time.time() - t0, 1)} s")
            run_loss, run_n = 0.0, 0
        if step % args.eval_every == 0:
            vl = val_loss()
            history.append({"step": step, "val_loss": vl})
            log(f"  validation pas {step} : loss {vl:.4f} (perplexité {math.exp(vl):.2f})")
            save(vl)
    vl = val_loss()
    history.append({"step": step, "val_loss": vl})
    save(vl)
    log(f"fin : pas {step} | loss validation finale {vl:.4f} | {round(time.time() - t0, 1)} s "
        f"| modèle : {folder}")


if __name__ == "__main__":
    main()
