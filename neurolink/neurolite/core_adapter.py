"""Adaptateur serveur pour le transformer : mêmes méthodes que NeuroLink, calcul par neurolite.nn."""
import math
import os

from . import NeuroLite, nn


class TransformerAdapter(NeuroLite):
    def __init__(self, chars, cfg, net, manifest=None, folder=None):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.K = cfg["context"]
        self.cfg = cfg
        self.net = net
        self.h = None
        self.pure = False
        self.manifest = manifest or {}
        self.folder = folder

    @property
    def num_params(self):
        return self.net.num_parameters()

    @property
    def config(self):
        return {"vocab_size": len(self.chars), "context": self.K, "dim": self.cfg["dim"],
                "hidden": self.cfg["hidden"], "experts": 0, "topk": 0,
                "params": self.num_params, "architecture": "causal-transformer",
                "layers": self.cfg["layers"], "heads": self.cfg["heads"]}

    def _probs(self, win):
        """Distribution du caractère suivant. Les zéros de bourrage en tête sont retirés :
        le modèle ne voit que le vrai contexte (jusqu'à K caractères)."""
        k = 0
        while k < len(win) - 1 and win[k] == 0:
            k += 1
        ctx = list(win[k:])
        T, V = len(ctx), len(self.chars)
        logits = self.net.forward([ctx])
        row = logits.data[(T - 1) * V: T * V]
        mx = max(row)
        e = [math.exp(v - mx) for v in row]
        s = sum(e)
        return [v / s for v in e]

    def embed(self, text):
        win = self._window(self.encode(text))
        d = self.cfg["dim"]
        emb = self.net.tok
        out = [0.0] * d
        for t in win:
            for j in range(d):
                out[j] += emb.weight.data[t * d + j]
        return [v / len(win) for v in out]

    def train(self, *a, **k):
        raise RuntimeError("entraînement : scripts/train_minilm.py")

    def save(self, folder, manifest=None):
        from .transformer import save_transformer
        save_transformer(self.net, folder, manifest=manifest, chars=self.chars)
        self.folder = folder
        return folder


def load_transformer_model(folder, manifest=None):
    from .transformer import load_transformer
    net, chars, cfg = load_transformer(folder)
    man = manifest
    mpath = os.path.join(folder, "manifest.json")
    if man is None and os.path.exists(mpath):
        import json
        with open(mpath, encoding="utf-8") as f:
            man = json.load(f)
    return TransformerAdapter(chars, cfg, net, manifest=man or {}, folder=folder)
