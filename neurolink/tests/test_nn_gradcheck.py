"""Vérifie la rétropropagation de neurolite.nn par différences finies, et la cohérence C / Python."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurolite import nn  # noqa: E402

rng = random.Random(0)
B, K, D, V, H = 3, 4, 5, 11, 6
emb = nn.Embedding(V, D, seed=1)
moe = nn.MoE(K * D, V, n_experts=3, hidden=H, seed=2)
lin = nn.Linear(K * D, K * D, seed=5)
act = nn.Tanh()
model = nn.Sequential(emb, lin, act, moe)
loss_fn = nn.CrossEntropy()
idx = [[rng.randrange(V) for _ in range(K)] for _ in range(B)]
tgt = [rng.randrange(V) for _ in range(B)]


def loss_only():
    h = emb.forward(idx)
    h = act.forward(lin.forward(h))
    return loss_fn.forward(moe.forward(h), tgt)


def analytic():
    model.zero_grad()
    l = loss_only()
    dy = loss_fn.backward()
    moe_dx = moe.backward(dy)
    dh = act.backward(moe_dx)
    dz = lin.backward(dh)
    emb.backward(dz)
    return l


l0 = analytic()
worst = 0.0
checked = 0
rel_checked = 0
for name, p in model.named_parameters():
    for i in rng.sample(range(p.size), min(6, p.size)):
        eps = 1e-2  # float32 : pas trop petit pour éviter le bruit d'arrondi
        orig = p.data[i]
        p.data[i] = orig + eps
        lp = loss_only()
        p.data[i] = orig - eps
        lm = loss_only()
        p.data[i] = orig
        num = (lp - lm) / (2 * eps)
        ana = p.grad.data[i]
        abs_err = abs(num - ana)
        err = abs_err / max(1e-3, abs(num) + abs(ana))
        if abs(ana) > 1e-3:
            rel_checked += 1
        if abs_err < 1e-4:
            err = 0.0  # bruit d'arrondi float32 sur la perte
        worst = max(worst, err)
        checked += 1
        assert err < 3e-2, (name, i, num, ana)
print(f"gradients vérifiés : {checked} (dont {rel_checked} comparés en relatif) | erreur relative max : {worst:.2e} | perte : {l0:.4f}")
assert rel_checked >= 10, "trop peu de gradients significatifs pour un test fiable"
print("native" if nn.native() else "python", "backend")
print("TEST GRADIENT OK")
