#!/usr/bin/env python3
"""Génère les données de fine-tuning.
  corpus/generated/*.txt : entraînement (formulations A)
  corpus/val/*.txt       : validation (formulations B jamais vues à l'entraînement,
                           plus les 4 questions des sondes de compréhension)
Déterministe (graine fixe)."""
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(7)

PROBES = [
    ("Quelle est la capitale de l'Allemagne ?", "Berlin"),
    ("Combien font 12 fois 7 ?", "84"),
    ('Traduis "good morning" en français.', "bonjour"),
    ("Quelle est la couleur du ciel par temps clair ?", "bleu"),
]


CAPITALS = [("l'Allemagne", "Berlin"), ("l'Espagne", "Madrid"), ("l'Italie", "Rome"),
            ("le Japon", "Tokyo"), ("le Canada", "Ottawa"), ("le Portugal", "Lisbonne"),
            ("la Belgique", "Bruxelles"), ("la Suisse", "Berne"), ("le Maroc", "Rabat"),
            ("l'Égypte", "Le Caire"), ("la Grèce", "Athènes"), ("la Norvège", "Oslo"),
            ("la Suède", "Stockholm"), ("le Mexique", "Mexico"), ("le Brésil", "Brasilia"),
            ("l'Inde", "New Delhi"), ("la Chine", "Pékin"), ("le Kenya", "Nairobi"),
            ("le Pérou", "Lima"), ("l'Argentine", "Buenos Aires")]

WORDS = [("good morning", "bonjour"), ("thank you", "merci"), ("cat", "chat"),
         ("dog", "chien"), ("house", "maison"), ("water", "eau"), ("bread", "pain"),
         ("book", "livre"), ("apple", "pomme"), ("sun", "soleil"), ("moon", "lune"),
         ("tree", "arbre"), ("car", "voiture"), ("school", "école"), ("friend", "ami")]

COLORS = [("la neige", "blanche"), ("l'herbe", "verte"), ("le sang", "rouge"),
          ("le charbon", "noir"), ("la mer", "bleue"), ("le lait", "blanc")]

SYMBOL = {"fois": "×", "plus": "+", "moins": "−"}


def conv(q, think, answer):
    return (f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n"
            f"<think>\n{think}\n</think>\n{answer}\n<|end|>\n")


def arithmetic_items():
    combos = [(a, b, op) for a in range(2, 13) for b in range(2, 13) for op in SYMBOL]
    rng.shuffle(combos)
    out = []
    for a, b, op in combos:
        if op == "moins" and a < b:
            a, b = b, a
        r = a * b if op == "fois" else a + b if op == "plus" else a - b
        out.append((a, b, op, r))
    return out


def train_items():
    items = []
    for c, cap in CAPITALS:
        for tpl in ("Quelle est la capitale de {c} ?", "Dis-moi la capitale de {c}."):
            q = tpl.format(c=c)
            if q not in PROBE_Q:
                items.append(conv(q, f"La question porte sur la géographie. "
                                     f"La capitale de {c} est {cap}.",
                                  f"La capitale de {c} est {cap}."))
    arith = arithmetic_items()
    for a, b, op, r in arith[:60]:
        q = f"Combien font {a} {op} {b} ?"
        if q not in PROBE_Q:
            items.append(conv(q, f"Je calcule {a} {SYMBOL[op]} {b} = {r}.",
                              f"{a} {op} {b} font {r}."))
    for en, fr in WORDS[:12]:
        q = f'Traduis "{en}" en français.'
        if q not in PROBE_Q:
            items.append(conv(q, f'Traduction courante de "{en}" : "{fr}".',
                              f'"{en}" se traduit par "{fr}".'))
    for x, col in COLORS:
        items.append(conv(f"Quelle est la couleur de {x} ?",
                          f"Il s'agit d'une couleur courante : {col}.",
                          f"La couleur de {x} est {col}."))
    rng.shuffle(items)
    return items, arith[60:]


def val_items(unused_arith):
    items = []
    for c, cap in CAPITALS:
        q = f"Capitale de {c} ?"
        items.append(conv(q, f"La question porte sur la géographie. La capitale de {c} est {cap}.",
                          f"La capitale de {c} est {cap}."))
    for a, b, op, r in unused_arith[:40]:
        q = f"Calcule {a} {op} {b}."
        items.append(conv(q, f"Je calcule {a} {SYMBOL[op]} {b} = {r}.", f"Le résultat est {r}."))
    for en, fr in WORDS:
        q = f'Comment dit-on "{en}" en français ?'
        items.append(conv(q, f'Il faut trouver le mot français pour "{en}".', f'On dit "{fr}".'))
    for x, col in COLORS:
        items.append(conv(f"Dis-moi la couleur de {x}.",
                          f"Il s'agit d'une couleur courante : {col}.",
                          f"{x[0].upper() + x[1:]} est {col}."))
    for q, expect in PROBES:
        items.append(conv(q, f"Je réfléchis à la question. La réponse est {expect}.", expect.capitalize() + "."))
    rng.shuffle(items)
    return items


def main():
    train, unused = train_items()
    val = val_items(unused)
    with open(os.path.join(ROOT, "corpus", "generated", "train.txt"), "w", encoding="utf-8") as f:
        f.write("".join(train))
    with open(os.path.join(ROOT, "corpus", "val", "heldout.txt"), "w", encoding="utf-8") as f:
        f.write("".join(val))
    print(f"entraînement : {len(train)} conversations ; validation : {len(val)} conversations")


if __name__ == "__main__":
    main()
