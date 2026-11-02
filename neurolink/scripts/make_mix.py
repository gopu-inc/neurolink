#!/usr/bin/env python3
"""Mélange (aléatoire, graine fixe) les exemples français et data.json dans corpus/mix_v4.txt."""
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEP = "<|system|>"
sources = ["corpus/french_v4.txt", "corpus/data_json_v3.txt"]
examples = []
for rel in sources:
    text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    examples += [SEP + p for p in text.split(SEP) if p.strip()]
random.Random(7).shuffle(examples)
out = os.path.join(ROOT, "corpus", "mix_v4.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("".join(examples))
print(f"{len(examples)} exemples, {sum(len(e) for e in examples)} caractères -> {out}")
