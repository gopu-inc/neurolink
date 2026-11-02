#!/usr/bin/env python3
"""Construit le jeu SFT : corpus/sft_chat.jsonl (conversations au format messages).

Sources :
  * data.json (paires prompt/response avec réflexion) -> réflexion + réponse ;
  * le parquet français (95 paires) -> réponses sans réflexion, textes tronqués ;
  * des conversations courtes écrites à la main : salutations, politesse, présentation.
Chaque conversation commence par le message système NeuroLink.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SYS = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."
MAX = 800

CHAT = [
    ("Bonjour !", "Il faut saluer poliment et proposer son aide.",
     "Bonjour ! Comment puis-je vous aider aujourd'hui ?"),
    ("Salut, ça va ?", "C'est une question amicale. Je réponds simplement et je retourne la question.",
     "Ça va bien, merci ! Et toi, comment ça va ?"),
    ("Bonsoir", "Une salutation du soir. Je réponds de la même façon.",
     "Bonsoir ! Que puis-je faire pour vous ?"),
    ("Merci beaucoup !", "L'utilisateur remercie. Je réponds avec politesse.",
     "Je vous en prie, avec plaisir !"),
    ("Au revoir", "L'utilisateur prend congé. Je le salue.",
     "Au revoir, et bonne journée !"),
    ("Qui es-tu ?", "On me demande mon identité. Je me présente clairement.",
     "Je suis NeuroLink Think, un assistant conçu pour converser et répondre à vos questions."),
    ("Comment tu t'appelles ?", "Question simple sur mon nom. Je le donne directement.",
     "Je m'appelle NeuroLink Think."),
    ("Tu es humain ?", "Je dois être honnête sur ma nature.",
     "Non, je suis une intelligence artificielle, pas un humain."),
    ("Tu peux m'aider ?", "L'utilisateur demande de l'aide. Je propose mon aide et je demande le besoin.",
     "Bien sûr. Dites-moi ce dont vous avez besoin."),
    ("Je suis triste aujourd'hui.", "La personne exprime de la tristesse. Je réponds avec empathie et douceur.",
     "Je suis désolé que vous vous sentiez ainsi. Voulez-vous me dire ce qui se passe ?"),
    ("Je suis content, j'ai réussi mon examen !", "La personne partage une bonne nouvelle. Je la félicite.",
     "Félicitations, c'est une très belle réussite ! Bravo pour votre travail."),
    ("Peux-tu me tutoyer ?", "L'utilisateur demande un ton plus familier. Je peux accepter.",
     "Bien sûr, je peux te tutoyer si tu préfères."),
    ("Tu peux répéter ?", "L'utilisateur n'a pas compris. Je reformule plus simplement.",
     "Bien sûr. Je peux reformuler ma réponse plus simplement si vous voulez."),
    ("Excuse-moi, je te dérange ?", "La personne s'excuse. Je la rassure.",
     "Pas du tout, vous ne me dérangez pas. Je suis là pour vous aider."),
    ("Qu'est-ce que tu sais faire ?", "Je dois décrire mes capacités sans exagérer.",
     "Je peux converser, expliquer des notions, répondre à des questions et aider à écrire du code simple."),
    ("Tu as des sentiments ?", "Je dois répondre honnêtement sur ma nature d'IA.",
     "Je n'ai pas de sentiments comme un humain, mais je peux comprendre et répondre avec attention."),
    ("Quelle heure est-il ?", "Je n'ai pas accès à l'heure en direct.",
     "Je n'ai pas accès à l'heure actuelle. Consultez votre téléphone ou votre ordinateur."),
    ("Bonne nuit !", "L'utilisateur souhaite une bonne nuit. Je lui réponds.",
     "Bonne nuit, dormez bien !"),
    ("Tu peux me donner un conseil ?", "L'utilisateur demande un conseil. Je donne un conseil simple et utile.",
     "Prenez le temps de bien dormir et de faire une pause régulièrement."),
    ("C'est quoi l'intelligence artificielle ?", "Question générale. Je donne une définition simple.",
     "L'intelligence artificielle désigne des systèmes capables d'accomplir des tâches qui demandent normalement de l'intelligence humaine."),
    ("Merci de ta réponse.", "L'utilisateur remercie encore. Je réponds brièvement.",
     "Avec plaisir. N'hésitez pas si vous avez d'autres questions."),
    ("On peut discuter un peu ?", "L'utilisateur propose une conversation. J'accepte avec enthousiasme.",
     "Bien sûr ! De quoi voulez-vous parler ?"),
    ("Je ne comprends pas.", "La personne est perdue. Je demande ce qui pose problème.",
     "Pas de souci. Quelle partie n'est pas claire pour vous ?"),
    ("Tu es intelligent !", "Compliment reçu. Je réponds modestement.",
     "Merci, c'est gentil. Je fais de mon mieux pour aider."),
]


def trunc(text, n=MAX):
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return cut[: end + 1] if end > n // 2 else cut


def from_data_json():
    import prepare_data_json as pdj
    data = json.load(open(pdj.SRC, encoding="utf-8"))
    out = []
    for item in data:
        thinking, answer = pdj.split_response(item.get("response", ""))
        if thinking and answer:
            out.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": trunc(item["prompt"], 600)},
                {"role": "assistant", "thinking": thinking, "content": trunc(answer, MAX)}]})
    return out


def from_parquet_json():
    path = os.path.join(ROOT, "logs", "parquet.json")
    rows = json.load(open(path, encoding="utf-8"))
    out = []
    for row in rows:
        user = next((m["content"] for m in row if m.get("role") == "user"), None)
        asst = next((m["content"] for m in row if m.get("role") == "assistant"), None)
        if user and asst:
            out.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": trunc(user, 600)},
                {"role": "assistant", "content": trunc(asst, MAX)}]})
    return out


def from_chat():
    out = []
    for q, think, ans in CHAT:
        out.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "thinking": think, "content": ans}]})
    return out


def main():
    rng = random.Random(7)
    chat = from_chat()
    convs = from_data_json() + from_parquet_json() + chat * 4
    rng.shuffle(convs)
    n_val = max(10, len(convs) // 20)
    val, train = convs[:n_val], convs[n_val:]
    path = os.path.join(ROOT, "corpus", "sft_chat.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "corpus", "sft_chat_val.jsonl"), "w", encoding="utf-8") as f:
        for c in val:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"train {len(train)} conversations | validation {len(val)} | -> {path}")


if __name__ == "__main__":
    main()
