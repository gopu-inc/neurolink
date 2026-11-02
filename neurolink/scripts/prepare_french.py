#!/usr/bin/env python3
"""Convertit le parquet français (messages user/assistant) au format Bund.

Chaque paire devient : <|user|> question <|end|> <|assistant|> <|thinking|> (vide) </|thinking|>
<|actual_response|> réponse </|actual_response|> <|end|>. Le jeu ne contient pas de réflexion :
le bloc <|thinking|> est donc vide, et le modèle n'apprend pas à raisonner depuis cette source.
Les paires dont le texte dépasse --max-chars sont écartées, pour garder un entraînement raisonnable.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_rows", help="sortie JSON de scripts/read_parquet.py --out")
    ap.add_argument("--max-chars", type=int, default=100000)
    ap.add_argument("--budget", type=int, default=1500000, help="total de caractères français visé")
    ap.add_argument("--out", default=os.path.join(ROOT, "corpus", "french_v4.txt"))
    args = ap.parse_args()
    import random
    rows = json.load(open(args.json_rows, encoding="utf-8"))
    random.Random(7).shuffle(rows)
    out, skipped, total = [], 0, 0
    for row in rows:
        user = next((m["content"] for m in row if m.get("role") == "user"), None)
        asst = next((m["content"] for m in row if m.get("role") == "assistant"), None)
        if not user or not asst or len(user) + len(asst) > args.max_chars:
            skipped += 1
            continue
        if total + len(user) + len(asst) > args.budget:
            skipped += 1
            continue
        total += len(user) + len(asst)
        out.append(f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{user.strip()}\n<|end|>\n"
                   f"<|assistant|>\n<|thinking|>\n\n</|thinking|>\n"
                   f"<|actual_response|>\n{asst.strip()}\n</|actual_response|>\n<|end|>\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"{len(out)} exemples écrits ({total} caractères), {skipped} écartés -> {args.out}")


if __name__ == "__main__":
    main()
