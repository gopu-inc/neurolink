#!/usr/bin/env python3
"""Grille de cohérence : un modèle n'est livré que s'il passe ces trois critères.

  C1 pertinence     : « Quelle est la capitale de l'Espagne ? » -> la réponse contient « Madrid »
  C2 contexte       : deux tours ; la réponse à « Comment je m'appelle ? » reprend « Léa »
  C3 réflexion      : le bloc de réflexion fait au moins 20 caractères et aucune ligne n'y est
                      répétée 3 fois ou plus

Chaque critère est testé avec 3 graines ; il est validé si au moins 2 réponses sur 3 passent.
Le modèle passe la grille seulement si les trois critères sont validés.
Usage : python scripts/check_criteria.py NOM_DU_MODELE
"""
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from neurolite import load_model  # noqa: E402
from neurolite.server import build_prompt, merge_options, text_events  # noqa: E402

SEEDS = [1, 2, 3]


def generate(model, messages, seed):
    man = model.manifest or {}
    prompt = build_prompt(messages, man.get("system", ""), True)
    opts = merge_options(man.get("parameters"), {"num_predict": 300, "seed": seed})
    content, thinking = [], []
    for kind, piece in text_events(model, prompt, opts):
        if kind == "done":
            break
        (content if kind == "content" else thinking).append(piece)
    return "".join(content).strip(), "".join(thinking).strip()


def no_repeated_lines(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = collections.Counter(lines)
    return all(v < 3 for v in counts.values())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "neurolink-minilm-512-sft"
    model = load_model(name)
    results = {"C1 pertinence": [], "C2 contexte": [], "C3 réflexion": []}
    details = []
    for seed in SEEDS:
        ans, th = generate(model, [{"role": "user", "content": "Quelle est la capitale de l'Espagne ?"}], seed)
        ok1 = "Madrid" in ans
        results["C1 pertinence"].append(ok1)
        ans2, _ = generate(model, [
            {"role": "user", "content": "Je m'appelle Léa."},
            {"role": "assistant", "content": "Enchanté Léa."},
            {"role": "user", "content": "Comment je m'appelle ?"}], seed)
        ok2 = "Léa" in ans2
        results["C2 contexte"].append(ok2)
        ok3 = len(th) >= 20 and no_repeated_lines(th)
        results["C3 réflexion"].append(ok3)
        details.append({"seed": seed, "C1": ans[:200], "C2": ans2[:200], "pensee": th[:200],
                        "ok": [ok1, ok2, ok3]})
    summary = {k: f"{sum(v)}/{len(v)}" for k, v in results.items()}
    passed = {k: sum(v) >= 2 for k, v in results.items()}
    report = {"model": name, "summary": summary, "passed": passed,
              "delivered": all(passed.values()), "details": details}
    out = os.path.join(ROOT, "checkpoints", "criteria.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for k in results:
        print(f"{k} : {summary[k]} -> {'OK' if passed[k] else 'ÉCHEC'}")
    print("LIVRABLE" if report["delivered"] else "NON LIVRÉ : critères non atteints")


if __name__ == "__main__":
    main()
