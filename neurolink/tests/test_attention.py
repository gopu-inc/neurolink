"""Vérifie l'attention causale : gradients par différences finies, et causalité stricte."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurolite import nn  # noqa: E402

rng = random.Random(0)
B, T, dim, heads = 2, 5, 8, 2
att = nn.CausalSelfAttention(dim, heads=heads, seed=3)
x = nn.Tensor((B * T, dim), [rng.uniform(-1, 1) for _ in range(B * T * dim)])
G = [rng.uniform(-1, 1) for _ in range(B * T * dim)]  # perte = somme(sortie * G)


def loss_of(xt):
    out = att.forward(xt, T)
    return sum(a * b for a, b in zip(out.data, G))


# gradients des paramètres
att.zero_grad()
base = loss_of(x)
att.backward(nn.Tensor((B * T, dim), G))
checked, worst = 0, 0.0
for name, p in att.named_parameters():
    for i in rng.sample(range(p.size), min(10, p.size)):
        eps = 1e-3
        orig = p.data[i]
        p.data[i] = orig + eps
        lp = loss_of(x)
        p.data[i] = orig - eps
        lm = loss_of(x)
        p.data[i] = orig
        num = (lp - lm) / (2 * eps)
        ana = p.grad.data[i]
        err = abs(num - ana) / max(1e-3, abs(num) + abs(ana))
        worst = max(worst, err if abs(num - ana) > 1e-5 else 0.0)
        checked += 1
        assert err < 5e-2 or abs(num - ana) < 1e-5, (name, i, num, ana)
print(f"gradients d'attention vérifiés : {checked} | erreur relative max : {worst:.2e}")

# causalité : modifier un token futur ne change pas les sorties passées
out1 = att.forward(x, T).data[:]
x2 = nn.Tensor((B * T, dim), x.data[:])
for d in range(dim):
    x2.data[(T - 1) * dim + d] += 5.0  # dernière position de la première séquence
out2 = att.forward(x2, T).data
past = [i for i in range(T - 1) for _ in range(dim)]
leak = max(abs(out1[t * dim + d] - out2[t * dim + d]) for t in range(T - 1) for d in range(dim))
assert leak < 1e-6, f"fuite d'information du futur : {leak}"
print(f"causalité : fuite max vers le passé = {leak:.2e}")
print("TEST ATTENTION OK")
