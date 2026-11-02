#!/usr/bin/env python3
"""Génère corpus/neurolink_think_v2.txt : exemples synthétiques au format de conversation.
Les exemples sont tirés de modèles (arithmétique, problèmes, traductions, faits, code,
honnêteté). Graine fixe : le fichier est reproductible."""
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."


def ex(q, think, ans):
    return (f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n"
            f"<|assistant|>\n<think>\n{think}\n</think>\n{ans}\n<|end|>\n")


FACTS = [
    ("Quelle est la capitale de l'Espagne ?", "L'Espagne a pour capitale Madrid.", "La capitale de l'Espagne est Madrid."),
    ("Quelle est la capitale de l'Italie ?", "L'Italie a pour capitale Rome.", "La capitale de l'Italie est Rome."),
    ("Quelle est la capitale du Japon ?", "Le Japon a pour capitale Tokyo.", "La capitale du Japon est Tokyo."),
    ("Quelle est la capitale du Canada ?", "Le Canada a pour capitale Ottawa.", "La capitale du Canada est Ottawa."),
    ("Quelle est la capitale du Portugal ?", "Le Portugal a pour capitale Lisbonne.", "La capitale du Portugal est Lisbonne."),
    ("Combien de continents y a-t-il ?", "On compte généralement sept continents.", "Il y a sept continents."),
    ("Combien de jours y a-t-il dans une semaine ?", "Une semaine compte sept jours.", "Une semaine compte sept jours."),
    ("Combien de mois y a-t-il dans une année ?", "Une année compte douze mois.", "Une année compte douze mois."),
    ("Combien de minutes y a-t-il dans une heure ?", "Une heure fait soixante minutes.", "Une heure compte soixante minutes."),
    ("Quel est le plus grand océan ?", "Le plus grand océan est le Pacifique.", "Le plus grand océan est le Pacifique."),
    ("Quelle planète est la plus proche du Soleil ?", "Mercure est la planète la plus proche du Soleil.", "La planète la plus proche du Soleil est Mercure."),
    ("De quelle couleur est le ciel par temps clair ?", "Par temps clair, la lumière diffusée rend le ciel bleu.", "Par temps clair, le ciel est bleu."),
    ("Combien de pattes a une araignée ?", "Une araignée est un arachnide à huit pattes.", "Une araignée a huit pattes."),
    ("Que mange habituellement une vache ?", "Une vache est herbivore.", "Une vache mange surtout de l'herbe et du foin."),
    ("Quel est le contraire de chaud ?", "Le contraire de chaud est froid.", "Le contraire de chaud est froid."),
    ("Quel est le contraire de grand ?", "Le contraire de grand est petit.", "Le contraire de grand est petit."),
    ("Combien de côtés a un triangle ?", "Un triangle a trois côtés.", "Un triangle a trois côtés."),
    ("Combien de côtés a un carré ?", "Un carré a quatre côtés égaux.", "Un carré a quatre côtés égaux."),
    ("Qu'est-ce qu'un mammifère ?", "Un mammifère est un animal vertébré dont les femelles allaitent leurs petits.", "Un mammifère est un animal vertébré qui allaite ses petits."),
    ("Qu'est-ce que l'eau bout à 100 degrés Celsius ?", "À pression normale, l'eau bout à cent degrés Celsius.", "À pression normale, l'eau bout à 100 degrés Celsius."),
    ("Quel est le plus petit nombre premier ?", "Le nombre premier le plus petit est deux.", "Le plus petit nombre premier est 2."),
    ("Qui a écrit Les Misérables ?", "Les Misérables est un roman de Victor Hugo.", "Les Misérables a été écrit par Victor Hugo."),
    ("Qu'est-ce qu'un réseau de neurones ?", "Un réseau de neurones est un modèle de calcul inspiré du cerveau.", "C'est un modèle de calcul composé de neurones artificiels reliés par des poids."),
    ("Qu'est-ce qu'un modèle de langage ?", "Un modèle de langage prédit le prochain mot ou caractère d'un texte.", "Un modèle de langage prédit la suite d'un texte à partir de ce qui le précède."),
    ("Que signifie l'acronyme CPU ?", "CPU signifie processeur central en français.", "CPU signifie « Central Processing Unit », soit processeur."),
    ("Qu'est-ce que le langage C ?", "Le langage C est un langage de programmation bas niveau et rapide.", "Le langage C est un langage de programmation compilé, rapide et proche du matériel."),
    ("Combien de secondes y a-t-il dans une minute ?", "Une minute compte soixante secondes.", "Une minute compte soixante secondes."),
    ("Quelle est la racine carrée de 81 ?", "Neuf fois neuf font quatre-vingt-un.", "La racine carrée de 81 est 9."),
    ("Quelle est la capitale de l'Allemagne ?", "L'Allemagne a pour capitale Berlin.", "La capitale de l'Allemagne est Berlin."),
    ("Quelle langue parle-t-on au Brésil ?", "Le Brésil est un pays lusophone.", "Au Brésil, on parle portugais."),
]

PAIRS = [("chat", "cat"), ("chien", "dog"), ("maison", "house"), ("eau", "water"), ("livre", "book"),
         ("soleil", "sun"), ("lune", "moon"), ("arbre", "tree"), ("pain", "bread"), ("merci", "thank you"),
         ("bonjour", "hello"), ("bonsoir", "good evening"), ("ami", "friend"), ("école", "school"),
         ("rouge", "red"), ("bleu", "blue"), ("vert", "green"), ("ville", "city"), ("mer", "sea"),
         ("montagne", "mountain"), ("fenêtre", "window"), ("porte", "door"), ("table", "table"),
         ("pomme", "apple"), ("oiseau", "bird")]

NAMES = ["Léa", "Marc", "Sami", "Inès", "Paul", "Nora", "Hugo", "Lina"]
OBJECTS = ["pommes", "billes", "cartes", "livres", "stylos", "bonbons", "crayons", "pièces"]
CODE = [
    ("Écris une fonction Python qui additionne deux nombres.",
     "Il faut renvoyer la somme des deux paramètres.",
     "def add(a, b):\n    return a + b"),
    ("Écris une fonction Python qui renvoie le plus grand élément d'une liste.",
     "On peut utiliser la fonction intégrée max.",
     "def plus_grand(liste):\n    return max(liste)"),
    ("Écris une fonction Python qui teste si un nombre est pair.",
     "Un nombre est pair si son reste modulo deux vaut zéro.",
     "def est_pair(n):\n    return n % 2 == 0"),
    ("Écris une fonction Python qui calcule la factorielle d'un entier.",
     "La factorielle de n est le produit de 1 à n.",
     "def factorielle(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r"),
    ("Écris une fonction Python qui inverse une chaîne.",
     "Le découpage avec un pas de moins un inverse la chaîne.",
     "def inverse(s):\n    return s[::-1]"),
]
GREET = [
    ("Bonjour !", "Il faut répondre poliment à une salutation.", "Bonjour ! Comment puis-je vous aider ?"),
    ("Salut, ça va ?", "C'est une question amicale. Je réponds simplement.", "Ça va bien, merci ! Et toi ?"),
    ("Bonsoir", "Une salutation du soir. Je réponds de la même façon.", "Bonsoir ! Que puis-je faire pour vous ?"),
    ("Merci beaucoup", "Il faut accuser réception du remerciement.", "Je vous en prie, avec plaisir !"),
    ("Au revoir", "L'utilisateur prend congé.", "Au revoir, à bientôt !"),
    ("Tu peux m'aider ?", "L'utilisateur demande de l'aide. Je propose mon aide.", "Bien sûr, dites-moi ce dont vous avez besoin."),
]
HONEST = [
    ("Quel jour sommes-nous ?", "Je n'ai pas accès à la date du jour.", "Je n'ai pas accès à la date actuelle. Pouvez-vous me la donner ?"),
    ("Quel temps fait-il à Paris ?", "Je n'ai pas accès à la météo en direct.", "Je n'ai pas accès à la météo en direct. Consultez un service météo."),
    ("Quel est le cours du bitcoin maintenant ?", "Je ne dispose pas de données de marché en temps réel.", "Je ne peux pas donner le cours actuel. Vérifiez une plateforme de marché."),
    ("Peux-tu te souvenir de notre conversation d'hier ?", "Je ne garde pas de mémoire entre les conversations.", "Je ne garde pas de souvenirs d'une conversation à l'autre."),
]


def arith(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    kind = rng.choice(["plus", "moins", "fois"])
    if kind == "plus":
        c = a + b
    elif kind == "moins":
        a, b = max(a, b), min(a, b)
        c = a - b
    else:
        a, b = rng.randint(0, 12), rng.randint(0, 12)
        c = a * b
    q = f"Combien font {a} {kind} {b} ?"
    think = f"{a} {kind} {b} = {c}."
    return ex(q, think, f"{a} {kind} {b} font {c}.")


def word(rng):
    name, obj = rng.choice(NAMES), rng.choice(OBJECTS)
    a = rng.randint(2, 60)
    b = rng.randint(1, a - 1)
    q = f"{name} a {a} {obj} et en donne {b}. Combien lui en reste-t-il ?"
    think = f"Il reste {a} - {b} = {a - b}."
    return ex(q, think, f"Il lui en reste {a - b}.")


def trip(rng):
    v, t = rng.randint(20, 120), rng.randint(1, 9)
    d = v * t
    q = f"Un train roule à {v} km/h pendant {t} heures. Quelle distance parcourt-il ?"
    think = f"distance = vitesse fois temps = {v} fois {t} = {d}."
    return ex(q, think, f"Il parcourt {d} km.")


def translate(rng):
    fr, en = rng.choice(PAIRS)
    if rng.random() < 0.5:
        q = f"Traduis « {fr} » en anglais."
        think = f"Le mot {fr} se traduit par {en}."
        ans = f"« {fr} » se traduit par « {en} »."
    else:
        q = f"Traduis « {en} » en français."
        think = f"Le mot anglais {en} correspond au français {fr}."
        ans = f"« {en} » se traduit par « {fr} »."
    return ex(q, think, ans)


def build(seed=7):
    rng = random.Random(seed)
    out = []
    out += [arith(rng) for _ in range(260)]
    out += [word(rng) for _ in range(140)]
    out += [trip(rng) for _ in range(60)]
    out += [translate(rng) for _ in range(80)]
    out += [ex(*f) for f in FACTS]
    out += [ex(*c) for c in CODE] * 3
    out += [ex(*g) for g in GREET] * 4
    out += [ex(*h) for h in HONEST] * 5
    rng.shuffle(out)
    return "".join(out)


if __name__ == "__main__":
    text = build()
    path = os.path.join(ROOT, "corpus", "neurolink_think_v2.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{text.count('<|system|>')} exemples, {len(text)} caractères -> {path}")
