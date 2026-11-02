"""Tests du SFT : masque de perte sur l'assistant uniquement, et baisse de la loss sur un petit modèle."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from neurolite import booster, sft  # noqa: E402

convs = [json.loads(l) for l in open(os.path.join(ROOT, "corpus", "sft_chat.jsonl"), encoding="utf-8")][:40]
chars = ["\x00"] + sorted({c for cv in convs for m in cv["messages"] for c in m.get("content", "")
                            + m.get("thinking", "")} | set("<|>/_thinking_actual_response_end_system_user_assistant\n"))
stoi = {c: i for i, c in enumerate(chars)}

# 1) le masque ne couvre que les réponses de l'assistant
msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "thinking": "T", "content": "Salut"}]
ids, mask, dropped = sft.tokenize(msgs, stoi)
trained = "".join(chars[i] for i, m in zip(ids, mask) if m)
assert "Salut" in trained and "Bonjour" not in trained and "<|user|>" not in trained, trained
assert "<|end|>" in trained and "<|assistant|>" not in trained, trained
print("tokens entraînés :", repr(trained[:120]))

# 2) la loss baisse sur un petit modèle (sanity check de l'entraînement)
cfg = {"K": 16, "D": 16, "H1": 32, "E": 2, "Hx": 16}
net, cfg = booster.build(chars, cfg, seed=3)
model = booster.NNModel(chars, cfg, net=net)
args = sft.SFTArguments(output_dir="/tmp/sft_test", per_device_train_batch_size=16,
                        max_steps=60, learning_rate=5e-3, warmup_steps=5, logging_steps=20,
                        eval_steps=0, save_steps=0)
tr = sft.SFTTrainer(model, convs, args)
hist = tr.train()
first, last = hist[0]["loss"], hist[-1]["loss"]
print(f"loss premier log {first:.3f} -> dernier log {last:.3f}")
assert last < first, "la loss ne baisse pas"
print("TEST SFT OK")
