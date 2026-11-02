"""neurolink-transformer : petit modèle de langage à attention causale (neurolite.nn).

Architecture (pré-résiduelle, sans normalisation de couche) :
  x = Embedding_tok(t) + Embedding_pos(position)
  pour chaque bloc : x = x + Attention(x) ; x = x + MLP(x)   (MLP = Linear -> Tanh -> Linear)
  logits = Linear(x) sur V caractères
Entraînement : une perte sur chaque position (prédiction du caractère suivant), ou seulement
sur les positions d'une réponse d'assistant (SFT, avec masque).
"""
import math

from . import nn
from .nn import Module, Tensor


def _add_into(dst, src):
    if nn._ops is not None:
        nn._ops.nl_axpy(1.0, src.buf_ptr, dst.buf_ptr, dst.size)
    else:
        for i in range(dst.size):
            dst.data[i] += src.data[i]


def _sum(a, b):
    out = Tensor(a.shape, a.data[:])
    _add_into(out, b)
    return out


class Block(Module):
    def __init__(self, dim, heads, hidden, seed=0):
        self.attn = nn.CausalSelfAttention(dim, heads=heads, seed=seed)
        self.fc1 = nn.Linear(dim, hidden, seed=seed + 10)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, dim, seed=seed + 11)

    def forward(self, x, T):
        h = _sum(x, self.attn.forward(x, T))
        m = self.fc2.forward(self.act.forward(self.fc1.forward(h)))
        return _sum(h, m)

    def backward(self, dy):
        dm = self.fc1.backward(self.act.backward(self.fc2.backward(dy)))
        dh = _sum(dy, dm)
        da = self.attn.backward(dh)
        return _sum(dh, da)


class MiniLM(Module):
    def __init__(self, vocab, context, dim=96, heads=4, layers=2, hidden=384, seed=0):
        self.vocab, self.context, self.dim = vocab, context, dim
        self.tok = nn.Embedding(vocab, dim, seed=seed)
        self.pos = nn.Embedding(context, dim, seed=seed + 1)
        self.blocks = [Block(dim, heads, hidden, seed=seed + 100 * (i + 1)) for i in range(layers)]
        self.head = nn.Linear(dim, vocab, seed=seed + 7)

    def forward(self, idx):
        """idx : liste de B séquences de longueur T (T <= context). Retourne logits (B*T, V)."""
        B, T = len(idx), len(idx[0])
        self._T = T
        x = self.tok.forward([[t] for row in idx for t in row])
        p = self.pos.forward([[t] for _ in range(B) for t in range(T)])
        _add_into(x, p)
        for blk in self.blocks:
            x = blk.forward(x, T)
        return self.head.forward(x)

    def backward(self, dlogits):
        dx = self.head.backward(dlogits)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.tok.backward(dx)
        self.pos.backward(dx)


def masked_cross_entropy(logits, targets, mask, loss_fn=None):
    """Perte moyenne sur les positions où mask == 1. Retourne (perte, dlogits complet)."""
    rows = [i for i, m in enumerate(mask) if m]
    V = logits.shape[1]
    dl = Tensor(logits.shape)
    if not rows:
        return 0.0, dl
    sub = Tensor((len(rows), V))
    for r, i in enumerate(rows):
        sub.data[r * V:(r + 1) * V] = logits.data[i * V:(i + 1) * V]
    fn = loss_fn or nn.CrossEntropy()
    loss = fn.forward(sub, [targets[i] for i in rows])
    dsub = fn.backward()
    for r, i in enumerate(rows):
        dl.data[i * V:(i + 1) * V] = dsub.data[r * V:(r + 1) * V]
    return loss, dl


def save_transformer(model, folder, manifest=None, chars=None):
    import json
    import os
    os.makedirs(folder, exist_ok=True)
    cfg = {"context": model.context, "dim": model.dim, "heads": model.blocks[0].attn.heads,
           "layers": len(model.blocks), "hidden": model.blocks[0].fc1.n_out}
    nn.save_state(model, os.path.join(folder, "model.nnpk"), extra={"cfg": cfg})
    with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "transformer-v1", "chars": chars, "K": cfg["context"], "cfg": cfg},
                  f, ensure_ascii=False)
    if manifest is not None:
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    return folder


def load_transformer(folder, manifest=None):
    import json
    import os
    with open(os.path.join(folder, "vocab.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta["cfg"]
    m = MiniLM(len(meta["chars"]), context=cfg["context"], dim=cfg["dim"], heads=cfg["heads"],
               layers=cfg["layers"], hidden=cfg["hidden"])
    nn.load_state(m, os.path.join(folder, "model.nnpk"))
    return m, meta["chars"], cfg

