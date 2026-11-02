"""Vérifie le gradient du transformer complet par différences finies, et la perte masquée."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurolite import nn  # noqa: E402
from neurolite.transformer import MiniLM, masked_cross_entropy  # noqa: E402

rng = random.Random(0)
V, T, B = 9, 6, 2
model = MiniLM(V, context=T, dim=8, heads=2, layers=2, hidden=12, seed=4)
idx = [[rng.randrange(V) for _ in range(T)] for _ in range(B)]
tgt = [rng.randrange(V) for _ in range(B * T)]
ce = nn.CrossEntropy()


def loss_only():
    return ce.forward(model.forward(idx), tgt)


model.zero_grad()
loss_only()
model.backward(ce.backward())
checked, bad = 0, 0
for name, p in model.named_parameters():
    for i in rng.sample(range(p.size), min(4, p.size)):
        eps = 1e-2
        orig = p.data[i]
        p.data[i] = orig + eps
        lp = loss_only()
        p.data[i] = orig - eps
        lm = loss_only()
        p.data[i] = orig
        num, ana = (lp - lm) / (2 * eps), p.grad.data[i]
        checked += 1
        if abs(num - ana) > 2e-4 and abs(num - ana) / max(1e-3, abs(num) + abs(ana)) > 5e-2:
            bad += 1
            print("écart", name, i, num, ana)
print(f"gradients du transformer vérifiés : {checked} | écarts : {bad}")
assert bad == 0
# perte masquée : les positions masquées ne contribuent pas
mask = [1 if i % 2 == 0 else 0 for i in range(B * T)]
logits = model.forward(idx)
l, dl = masked_cross_entropy(logits, tgt, mask)
zero_rows = [i for i, m in enumerate(mask) if not m]
assert all(abs(dl.data[i * V + j]) < 1e-9 for i in zero_rows for j in range(V))
print(f"perte masquée : {l:.4f} | gradient nul hors masque : oui")
print("TEST TRANSFORMER OK")
