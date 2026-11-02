#!/usr/bin/env python3
"""Entraîne neurolink-minilm (attention causale, contexte 512) en deux phases.

  pretrain : fenêtres aléatoires de 512 caractères sur corpus/mix_v4.txt, perte sur chaque position
  sft      : conversations (corpus/sft_chat.jsonl), perte seulement sur les réponses de l'assistant

Chaque phase reprend le poids de la précédente et s'arrête à --time-limit secondes.
Le journal (loss d'entraînement et de validation) va dans checkpoints/minilm.log.
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
from neurolite import nn, sft  # noqa: E402
from neurolite.transformer import MiniLM, masked_cross_entropy, save_transformer, load_transformer  # noqa: E402

LOG = os.path.join(ROOT, "checkpoints", "minilm.log")
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."


def log(msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_model(chars, args):
    return MiniLM(len(chars), context=args.context, dim=args.dim, heads=args.heads,
                  layers=args.layers, hidden=args.hidden, seed=2026)


def pretrain(args):
    text = open(os.path.join(ROOT, "corpus", "mix_v4.txt"), encoding="utf-8").read()
    chars = ["\x00"] + sorted(set(text) - {"\x00"})
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]
    n_val = len(ids) // 40
    train, val = ids[:-n_val], ids[-n_val:]
    T = args.context
    rng = random.Random(0)
    out = os.path.join(ROOT, "models", args.name_pre)
    if args.resume and os.path.exists(os.path.join(out, "model.nnpk")):
        model, chars, _ = load_transformer(out)
        log(f"pré-entraînement : reprise depuis {out}")
    else:
        model = build_model(chars, args)
    log(f"pré-entraînement : {model.num_parameters()} paramètres | contexte {T} | lot {args.batch} | "
        f"caractères train {len(train)} | lr max {args.lr}")
    vpos = random.Random(1).sample(range(len(val) - T - 1), 16)
    opt = nn.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropy()

    def val_loss():
        total = 0.0
        for s in range(0, len(vpos), args.batch):
            seqs = [val[p:p + T + 1] for p in vpos[s:s + args.batch]]
            logits = model.forward([q[:-1] for q in seqs])
            total += ce.forward(logits, [t for q in seqs for t in q[1:]]) * len(seqs)
        return total / len(vpos)

    best = float("inf")
    mpath = os.path.join(out, "manifest.json")
    if args.resume and os.path.exists(mpath):
        best = float(json.load(open(mpath, encoding="utf-8")).get("val_loss", float("inf")))
    log(f"meilleure loss de validation connue : {best:.4f}")
    t0, run, n = time.time(), 0.0, 0
    step = 0
    while step < args.steps and time.time() - t0 < args.time_limit:
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        starts = [rng.randrange(len(train) - T - 1) for _ in range(args.batch)]
        seqs = [train[s:s + T + 1] for s in starts]
        model.zero_grad()
        logits = model.forward([q[:-1] for q in seqs])
        loss = ce.forward(logits, [t for q in seqs for t in q[1:]])
        model.backward(ce.backward())
        opt.step(lr)
        step += 1
        run += loss
        n += 1
        if step % 50 == 0:
            log(f"pré pas {step}/{args.steps} | loss entraînement {run / n:.4f} | lr {lr:.2e} | "
                f"{round(time.time() - t0, 1)} s")
            run, n = 0.0, 0
        if step % 500 == 0:
            vl = val_loss()
            log(f"  pré validation pas {step} : loss {vl:.4f} (perplexité {math.exp(vl):.2f})")
            if vl < best:
                best = vl
                save_transformer(model, out, manifest=manifest(args, "pretrain", step, vl), chars=chars)
                log("  -> meilleur point sauvegardé")
            else:
                log(f"  -> pas d'amélioration (meilleur : {best:.4f})")
    vl = val_loss()
    if vl < best:
        save_transformer(model, out, manifest=manifest(args, "pretrain", step, vl), chars=chars)
    log(f"pré-entraînement fini : {step} pas | loss validation {vl:.4f} | {round(time.time() - t0, 1)} s")


def manifest(args, phase, step, vl):
    return {"name": args.name_final, "family": "neurolink", "format": "transformer",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parameter_size": "", "capabilities": ["completion", "chat", "thinking", "embedding"],
            "system": SYS, "phase": phase, "steps_done": step, "val_loss": vl,
            "parameters": {"temperature": 0.7, "top_k": 40, "top_p": 0.9,
                           "num_predict": 1000000, "stop": ["<|user|>"]}}


def sft_phase(args):
    model, chars, _ = load_transformer(os.path.join(ROOT, "models", args.name_pre))
    stoi = {c: i for i, c in enumerate(chars)}
    T = args.context
    convs = [json.loads(l) for l in open(os.path.join(ROOT, "corpus", "sft_chat.jsonl"), encoding="utf-8")]
    val_convs = [json.loads(l) for l in open(os.path.join(ROOT, "corpus", "sft_chat_val.jsonl"), encoding="utf-8")]
    toks = []
    for c in convs:
        ids, mask, _ = sft.tokenize(c["messages"], stoi)
        if len(ids) > T + 1 and sum(mask) > 0:
            toks.append((ids, mask))
    vtoks = []
    for c in val_convs:
        ids, mask, _ = sft.tokenize(c["messages"], stoi)
        if len(ids) > T + 1 and sum(mask) > 0:
            vtoks.append((ids, mask))
    log(f"SFT : {len(toks)} conversations (validation {len(vtoks)}) | lot {args.batch} | lr {args.sft_lr}")
    rng = random.Random(3)

    def make_batch(pool, size):
        seqs, masks = [], []
        for _ in range(size):
            ids, mask = pool[rng.randrange(len(pool))]
            s = rng.randrange(max(1, len(ids) - T - 1))
            seqs.append(ids[s:s + T + 1])
            masks.append(mask[s + 1:s + T + 1])
        return seqs, masks

    def batch_loss(seqs, masks, train):
        flat_t = [t for q in seqs for t in q[1:]]
        flat_m = [m for mm in masks for m in mm]
        logits = model.forward([q[:-1] for q in seqs])
        if train:
            loss, dl = masked_cross_entropy(logits, flat_t, flat_m)
            model.backward(dl)
            return loss
        loss, _ = masked_cross_entropy(logits, flat_t, flat_m)
        return loss

    opt = nn.Adam(model.parameters(), lr=args.sft_lr)
    t0, run, n, step = time.time(), 0.0, 0, 0
    while step < args.sft_steps and time.time() - t0 < args.sft_time:
        lr = args.sft_lr * min(1.0, (step + 1) / 50) * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.sft_steps)))
        seqs, masks = make_batch(toks, args.batch)
        model.zero_grad()
        run += batch_loss(seqs, masks, True)
        opt.step(lr)
        step += 1
        n += 1
        if step % 50 == 0:
            log(f"sft pas {step}/{args.sft_steps} | loss {run / n:.4f} | lr {lr:.2e} | "
                f"{round(time.time() - t0, 1)} s")
            run, n = 0.0, 0
    rv = random.Random(9)
    vl = []
    for _ in range(8):
        pool = vtoks or toks
        seqs, masks = [], []
        for _ in range(args.batch):
            ids, mask = pool[rv.randrange(len(pool))]
            s = rv.randrange(max(1, len(ids) - T - 1))
            seqs.append(ids[s:s + T + 1])
            masks.append(mask[s + 1:s + T + 1])
        vl.append(batch_loss(seqs, masks, False))
    vloss = sum(vl) / len(vl)
    out = os.path.join(ROOT, "models", args.name_sft)
    save_transformer(model, out, manifest=manifest(args, "sft", step, vloss), chars=chars)
    log(f"SFT fini : {step} pas | loss validation (réponses) {vloss:.4f} | modèle : {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pretrain", "sft"])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--time-limit", type=float, default=2400.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sft-steps", type=int, default=3000)
    ap.add_argument("--sft-lr", type=float, default=3e-4)
    ap.add_argument("--sft-time", type=float, default=1500.0)
    ap.add_argument("--name-pre", default="neurolink-minilm-512-pretrain")
    ap.add_argument("--name-sft", default="neurolink-minilm-512-sft")
    ap.add_argument("--name-final", default="neurolink-minilm-512:latest")
    args = ap.parse_args()
    if args.phase == "pretrain":
        pretrain(args)
    else:
        sft_phase(args)


if __name__ == "__main__":
    main()
