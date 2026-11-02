"""neurolink-booster : modèle de ~7M paramètres construit avec neurolite.nn.

Architecture (fenêtre de K caractères) :
  Embedding(V, D) -> aplatissement (K*D) -> Linear(K*D, H1) -> Tanh -> MoE(H1 -> V, E experts)
Le MoE est un mélange dense de E experts (Linear, Tanh, Linear) pondéré par une porte softmax.
Sauvegarde : model.nnpk (poids), vocab.json (caractères), manifest.json (format "nn").
"""
import json
import math
import os

from . import NeuroLite
from . import nn

DEFAULT_CONFIG = {"K": 32, "D": 128, "H1": 1100, "E": 4, "Hx": 448}


def build(chars, cfg=None, seed=0):
    cfg = dict(DEFAULT_CONFIG, **(cfg or {}))
    V = len(chars)
    emb = nn.Embedding(V, cfg["D"], seed=seed)
    hidden = nn.Sequential(nn.Linear(cfg["K"] * cfg["D"], cfg["H1"], seed=seed + 7), nn.Tanh())
    moe = nn.MoE(cfg["H1"], V, n_experts=cfg["E"], hidden=cfg["Hx"], seed=seed + 11)
    net = nn.Sequential(emb, hidden, moe)
    return net, cfg


class NNModel(NeuroLite):
    """Compatible avec le serveur : mêmes méthodes que NeuroLite, calcul par neurolite.nn."""

    def __init__(self, chars, cfg, net=None, seed=0):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.K = cfg["K"]
        self.cfg = cfg
        self.net = net if net is not None else build(self.chars, cfg, seed)[0]
        self.h = None
        self.pure = False
        self.manifest = {}
        self.folder = None

    @property
    def num_params(self):
        return self.net.num_parameters()

    @property
    def config(self):
        return {"vocab_size": len(self.chars), "context": self.K, "dim": self.cfg["D"],
                "hidden": self.cfg["H1"], "experts": self.cfg["E"], "topk": self.cfg["E"],
                "params": self.num_params, "architecture": "booster-moe"}

    def _probs(self, win):
        logits = self.net.forward([list(win)])
        row = logits.data[: len(self.chars)]
        mx = max(row)
        e = [math.exp(v - mx) for v in row]
        s = sum(e)
        return [v / s for v in e]

    def embed(self, text):
        win = self._window(self.encode(text))
        emb = self.net.layers[0]
        d = self.cfg["D"]
        out = [0.0] * d
        for t in win:
            for j in range(d):
                out[j] += emb.weight.data[t * d + j]
        return [v / len(win) for v in out]

    def train(self, *a, **k):
        raise RuntimeError("utilisez scripts/train_booster.py pour entraîner le booster")

    def save(self, folder, manifest=None):
        os.makedirs(folder, exist_ok=True)
        nn.save_state(self.net, os.path.join(folder, "model.nnpk"), extra={"cfg": self.cfg})
        with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"format": "nn-v1", "chars": self.chars, "K": self.K, "cfg": self.cfg},
                      f, ensure_ascii=False)
        if manifest is not None:
            with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        self.folder = folder
        return folder


def load_nn_model(folder, manifest=None):
    with open(os.path.join(folder, "vocab.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta["cfg"]
    net = build(meta["chars"], cfg)[0]
    nn.load_state(net, os.path.join(folder, "model.nnpk"))
    model = NNModel(meta["chars"], cfg, net=net)
    mpath = os.path.join(folder, "manifest.json")
    if manifest is not None:
        model.manifest = manifest
    elif os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            model.manifest = json.load(f)
    model.folder = folder
    return model
