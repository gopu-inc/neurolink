# NeuroLink / neurolite 0.2.0

Noyau de réseau de neurones 100 % C (embedding de contexte, mélange d'experts à routage
top-k, rétropropagation manuelle, Adam, clipping, warmup + décroissance cosinus), interface
Python via ctypes, et serveur d'API compatible Ollama.

## Installation

    python setup.py build_py          # compile csrc/neurolite.c -> neurolite/libneurolite.so
    pip install .                     # ou : pip install -e .

Nécessite gcc (ou CC=clang) et Linux/macOS.

## Python

    from neurolite import NeuroLite, load_model
    m = load_model("neurolink-think-2.1-pro")          # cherche dans ./models ou NEUROLINK_MODELS
    print(m.generate("<|user|>\nBonjour\n<|end|>\n<|assistant|>\n", max_tokens=120))

    model = NeuroLite.from_text(text, K=16, D=24, H=48, E=4, topk=2)
    model.train(text, epochs=10, lr=3e-3, warmup=500)   # affiche une jauge, pas de loss
    model.save("models/mon-modele", manifest={"name": "mon-modele:latest"})

## Entraîner le modèle fourni

    EPOCHS=12 python scripts/train_neurolink_think.py

## Serveur (routes et formats d'Ollama, port 11434)

    python server_nl.py --port 11434         # ou : neurolink-server (après pip install)

Routes : GET / · GET /api/version · GET /api/tags · GET /api/ps · POST /api/show ·
POST /api/generate · POST /api/chat · POST /api/embed · POST /api/embeddings ·
POST /api/pull · POST /api/create · POST /api/copy · DELETE /api/delete ·
GET /v1/models · POST /v1/chat/completions (compatibilité OpenAI).

    curl localhost:11434/api/generate -d '{"model":"neurolink-think-2.1-pro","prompt":"Bonjour","stream":false}'
    curl localhost:11434/api/chat -d '{"model":"neurolink-think-2.1-pro","messages":[{"role":"user","content":"Salut"}]}'

Les balises <think>…</think> sont séparées dans le champ `thinking` (chat et generate).

## Limites

Modèle de démonstration à contexte court (16 caractères), entraîné sur un petit corpus :
il apprend le format de conversation et la structure, pas le raisonnement d'un grand modèle.
Pas encore d'attention multi-têtes, ni d'ESN, ni de sous-agents, ni d'appels d'outils.

## iOS / a-Shell et machines sans bibliothèque C

Si `libneurolite.so` ne peut pas être chargé (message « slice is not valid mach-o file »
sur iOS, par exemple), le paquet bascule automatiquement sur le moteur pur Python
(`neurolite/pure.py`). Il charge et sert les modèles avec les mêmes résultats, mais plus
lentement. L'entraînement nécessite la bibliothèque C : compilez-la sur une machine de bureau.
Pour forcer le mode pur : `NEUROLITE_FORCE_PURE=1 python server_nl.py`.

## Format Bund (data.json)

Le modèle `neurolink-think-2.1-pro` (v3) est entraîné sur `data.json` au format de l'agent Bund :
`<|thinking|> … </|thinking|>` puis `<|actual_response|> … </|actual_response|>`.
Le serveur sépare la réflexion dans le champ `thinking` et arrête la génération à la fin de la réponse.

    python scripts/prepare_data_json.py /chemin/vers/data.json   # -> corpus/data_json_v3.txt
    python scripts/train_data_json.py --epochs 4 --val-frac 0.15 # garde le meilleur point (val)
    python scripts/train_data_json.py --epochs 25 --resume       # reprise depuis la dernière sauvegarde

Entraînement par lots de 32 exemples (moteur C). Une époque sur data.json prend environ 65 à 75 s.
Le meilleur modèle est sauvegardé selon la loss de validation ; sur 85 exemples, celle-ci atteint son
minimum vers 3 époques, puis remonte (sur-apprentissage).

## neurolite.nn et le booster 7M

`neurolite.nn` ressemble à `torch.nn` : `Module`, `Linear`, `Embedding`, `Tanh`, `Sequential`,
`MoE`, `CrossEntropy`, `Adam`, `save_state` / `load_state`. La différence : pas d'autograd
général, chaque module implémente `forward` et `backward`. Les calculs lourds passent par
`csrc/nlops.c` (dans `libneurolite.so`). Sans cette bibliothèque, un repli Python existe mais il est
très lent.

    python tests/test_nn_gradcheck.py                 # vérifie la rétropropagation
    python scripts/train_booster.py --steps 2500 --time-limit 1500
    curl localhost:11434/api/generate -d '{"model":"neurolink-booster-7m","prompt":"Bonjour","stream":false}'

Le booster (7,1 M de paramètres, fenêtre de 32 caractères) est entraîné sur le mélange français + code.
Après 2 500 pas (80 000 exemples), sa loss de validation est de 2,59 (perplexité 13,3). Il ne produit pas
encore de texte cohérent et n'a pas appris le format de réflexion.

## Sous-agents

`neurolite.agents.Orchestrator` : un planificateur découpe la tâche, des sous-agents la traitent en
parallèle, un agent de synthèse assemble la réponse. Chaque rôle peut utiliser un modèle différent.

    curl localhost:11434/api/agents/run -d '{"model":"neurolink-think-2.1-pro","task":"..."}'
