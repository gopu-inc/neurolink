"""Moteur pur Python (sans libneurolite.so) : chargement et inférence du format NLX2.
Sert de secours quand la bibliothèque C ne peut pas être chargée (ex. iOS / a-Shell).
L'inférence donne les mêmes résultats que le moteur C, mais plus lentement.
L'entraînement nécessite la bibliothèque C."""
import math
import struct
import sys
from array import array
from operator import mul

MAGIC = 0x32584C4E  # "NLX2"


def _softmax(v):
    mx = max(v)
    e = [math.exp(x - mx) for x in v]
    s = sum(e)
    return [x / s for x in e]


class PyNet:
    def __init__(self, V, K, D, H, E, topk, theta):
        self.V, self.K, self.D, self.H, self.E, self.topk = V, K, D, H, E, topk
        KD = K * D
        self.P = V * D + E * KD + E * H * KD + E * H + E * V * H + E * V
        if len(theta) != self.P:
            raise ValueError(f"taille de paramètres invalide ({len(theta)} au lieu de {self.P})")
        self.theta = theta
        self.o_Wg = V * D
        self.o_W1 = self.o_Wg + E * KD
        self.o_b1 = self.o_W1 + E * H * KD
        self.o_W2 = self.o_b1 + E * H
        self.o_b2 = self.o_W2 + E * V * H

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            hdr = struct.unpack("<8I", f.read(32))
            if hdr[0] != MAGIC:
                raise IOError("format NLX2 attendu")
            _, V, K, D, H, E, topk, _ = hdr
            theta = array("f")
            theta.frombytes(f.read())
        if sys.byteorder == "big":
            theta.byteswap()
        return cls(V, K, D, H, E, topk, theta)

    def save(self, path):
        data = array("f", self.theta)
        if sys.byteorder == "big":
            data.byteswap()
        with open(path, "wb") as f:
            f.write(struct.pack("<8I", MAGIC, self.V, self.K, self.D, self.H,
                                self.E, self.topk, 0))
            f.write(data.tobytes())

    @property
    def config(self):
        return {"vocab_size": self.V, "context": self.K, "dim": self.D, "hidden": self.H,
                "experts": self.E, "topk": self.topk, "params": self.P}

    def _forward(self, ctx):
        V, K, D, H, E, topk = self.V, self.K, self.D, self.H, self.E, self.topk
        th, KD = self.theta, K * D
        x = []
        for c in ctx:
            s = c * D
            x.extend(th[s:s + D])
        g = [sum(map(mul, th[self.o_Wg + e * KD: self.o_Wg + (e + 1) * KD], x)) for e in range(E)]
        p = _softmax(g)
        sel = sorted(range(E), key=lambda e: -p[e])[:topk]
        y = [0.0] * V
        for e in sel:
            a = []
            for h in range(H):
                r = self.o_W1 + (e * H + h) * KD
                a.append(math.tanh(th[self.o_b1 + e * H + h] + sum(map(mul, th[r:r + KD], x))))
            base = self.o_W2 + e * V * H
            for v in range(V):
                r = base + v * H
                s = th[self.o_b2 + e * V + v] + sum(map(mul, th[r:r + H], a))
                y[v] += p[e] * s
        return _softmax(y)

    def predict(self, ctx):
        return list(self._forward(ctx))

    def embed(self, ctx):
        D, th = self.D, self.theta
        out = [0.0] * D
        for c in ctx:
            s = c * D
            for d in range(D):
                out[d] += th[s + d]
        return [v / self.K for v in out]
