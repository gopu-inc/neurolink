#!/usr/bin/env python3
"""Convertit data.json (paires prompt/response) au format de conversation NeuroLink.

Chaque réponse est découpée en <|thinking|> ... </|thinking|> et <|actual_response|> ...
</|actual_response|>, comme dans le fichier source. Les fermetures mal écrites
(</thinking|>, </;thinking|>) sont corrigées. Les exemples sans réflexion ou sans réponse
exploitable sont écartés. Sortie : corpus/data_json_v3.txt
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."
SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/data.json"
DST = os.path.join(ROOT, "corpus", "data_json_v3.txt")


def split_response(raw):
    text = re.sub(r"</[;|]*thinking\|>", "</|thinking|>", raw)
    m_think = re.search(r"<\|thinking\|>(.*?)</\|thinking\|>", text, re.S)
    m_answer = re.search(r"<\|actual_response\|>(.*?)(</\|actual_response\|>|$)", text, re.S)
    thinking = m_think.group(1).strip() if m_think else ""
    if m_answer:
        answer = m_answer.group(1).strip()
    else:
        answer = text.split("</|thinking|>")[-1].strip()
    return thinking, answer


def fmt(prompt, thinking, answer):
    return (f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{prompt.strip()}\n<|end|>\n"
            f"<|assistant|>\n<|thinking|>\n{thinking}\n</|thinking|>\n"
            f"<|actual_response|>\n{answer}\n</|actual_response|>\n<|end|>\n")


def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    out, skipped = [], 0
    for item in data:
        thinking, answer = split_response(item.get("response", ""))
        if not thinking or not answer:
            skipped += 1
            continue
        out.append(fmt(item["prompt"], thinking, answer))
    with open(DST, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"{len(data)} entrées lues, {len(out)} exemples écrits, {skipped} écartés -> {DST}")


if __name__ == "__main__":
    main()
