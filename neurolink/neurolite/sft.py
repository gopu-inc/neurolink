"""neurolite.sft : fine-tuning supervisé (SFT) de modèles NeuroLink, sans PyTorch ni TRL.

Équivalent conceptuel de SFTTrainer :
  * les conversations (liste de messages role/content) sont rendues avec le même format de
    chat que le serveur (<|system|>, <|user|>, <|assistant|>, <|thinking|>, <|actual_response|>) ;
  * la perte n'est calculée que sur les tokens de l'assistant (masque), comme
    `completion_only_loss` dans TRL : le modèle apprend à répondre, pas à réciter la question ;
  * optimiseur Adam (neurolite.nn), warmup + décroissance cosinus, journal de loss et
    sauvegarde au format du modèle (dossier chargeable par load_model).

Exemple :
    model = booster.load_nn_model("models/neurolink-booster-7m")
    trainer = SFTTrainer(model, conversations, SFTArguments(output_dir="models/booster-sft"))
    trainer.train()
"""
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

from . import nn


@dataclass
class SFTArguments:
    output_dir: str
    per_device_train_batch_size: int = 32
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    time_limit: float = 0.0
    seed: int = 0
    manifest: dict = field(default_factory=dict)
    log_path: str = ""


def render(messages):
    """Liste de (texte, entraîné?) : seuls les segments de réponse de l'assistant sont entraînés."""
    segs = []
    for m in messages:
        role, content = m["role"], m.get("content", "")
        if role == "system":
            segs.append((f"<|system|>\n{content}\n<|end|>\n", False))
        elif role == "user":
            segs.append((f"<|user|>\n{content}\n<|end|>\n", False))
        elif role == "assistant":
            segs.append(("<|assistant|>\n", False))
            thinking = m.get("thinking", "")
            segs.append((f"<|thinking|>\n{thinking}\n</|thinking|>\n"
                         f"<|actual_response|>\n{content}\n</|actual_response|>\n<|end|>\n", True))
    return segs


def tokenize(messages, stoi):
    ids, mask, dropped = [], [], 0
    for text, trained in render(messages):
        for c in text:
            if c in stoi:
                ids.append(stoi[c])
                mask.append(1 if trained else 0)
            else:
                dropped += 1
    return ids, mask, dropped


class SFTDataset:
    def __init__(self, conversations, chars, K):
        self.K = K
        stoi = {c: i for i, c in enumerate(chars)}
        self.examples, self.dropped, self.positions = [], 0, []
        for conv in conversations:
            ids, mask, dropped = tokenize(conv["messages"] if "messages" in conv else conv, stoi)
            self.dropped += dropped
            targets = [i for i, m in enumerate(mask) if m and i > 0]
            if not targets:
                continue
            ei = len(self.examples)
            self.examples.append((ids, targets))
            self.positions.extend((ei, t) for t in targets)

    def window(self, ei, pos):
        ids = self.examples[ei][0]
        ctx = ids[max(0, pos - self.K):pos]
        return [0] * (self.K - len(ctx)) + ctx, ids[pos]

    def __len__(self):
        return len(self.positions)

    def batch(self, rng, size):
        picks = [self.positions[rng.randrange(len(self.positions))] for _ in range(size)]
        wins, tgts = zip(*[self.window(ei, p) for ei, p in picks])
        return list(wins), list(tgts)


class SFTTrainer:
    def __init__(self, model, train_dataset, args, eval_dataset=None):
        self.model = model
        self.args = args
        self.train_set = SFTDataset(train_dataset, model.chars, model.K)
        self.eval_set = SFTDataset(eval_dataset, model.chars, model.K) if eval_dataset else None
        self.loss_fn = nn.CrossEntropy()
        self.opt = nn.Adam(model.net.parameters(), lr=args.learning_rate)
        self.history = []

    def _log(self, msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        if self.args.log_path:
            os.makedirs(os.path.dirname(self.args.log_path) or ".", exist_ok=True)
            with open(self.args.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def evaluate(self, n=256):
        if self.eval_set is None or len(self.eval_set) == 0:
            return None
        rng = random.Random(123)
        total, count = 0.0, 0
        for _ in range(max(1, n // self.args.per_device_train_batch_size)):
            wins, tgts = self.eval_set.batch(rng, self.args.per_device_train_batch_size)
            logits = self.model.net.forward(wins)
            total += self.loss_fn.forward(logits, tgts) * len(tgts)
            count += len(tgts)
        return total / count

    def save_model(self, path=None):
        path = path or self.args.output_dir
        man = dict(self.args.manifest)
        man["train"] = {**man.get("train", {}), "history": self.history,
                        "samples_positions": len(self.train_set),
                        "tokens_dropped": self.train_set.dropped}
        self.model.save(path, man)
        return path

    def train(self):
        a = self.args
        B = a.per_device_train_batch_size
        total = a.max_steps if a.max_steps > 0 else int(math.ceil(len(self.train_set) / B * a.num_train_epochs))
        self._log(f"SFT : {len(self.train_set)} positions entraînées | caractères hors vocabulaire "
                  f"ignorés : {self.train_set.dropped} | pas : {total} | lot {B} | lr {a.learning_rate}")
        rng = random.Random(a.seed)
        params = self.model.net.parameters()
        t0, run, n_run = time.time(), 0.0, 0
        step = 0
        while step < total:
            if a.time_limit and time.time() - t0 > a.time_limit:
                self._log(f"limite de temps atteinte au pas {step}")
                break
            lr = a.learning_rate * min(1.0, (step + 1) / max(1, a.warmup_steps)) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            wins, tgts = self.train_set.batch(rng, B)
            self.model.net.zero_grad()
            logits = self.model.net.forward(wins)
            loss = self.loss_fn.forward(logits, tgts)
            self.model.net.backward(self.loss_fn.backward())
            self.opt.step(lr)
            step += 1
            run += loss
            n_run += 1
            if step % a.logging_steps == 0:
                avg = run / n_run
                self.history.append({"step": step, "loss": avg})
                self._log(f"pas {step}/{total} | loss {avg:.4f} | lr {lr:.2e} | "
                          f"{round(time.time() - t0, 1)} s")
                run, n_run = 0.0, 0
            if a.eval_steps and step % a.eval_steps == 0:
                ev = self.evaluate()
                if ev is not None:
                    self.history.append({"step": step, "eval_loss": ev})
                    self._log(f"  évaluation pas {step} : loss {ev:.4f}")
            if a.save_steps and step % a.save_steps == 0:
                self.save_model()
        ev = self.evaluate()
        if ev is not None:
            self._log(f"évaluation finale : loss {ev:.4f}")
            self.history.append({"step": step, "eval_loss": ev})
        self.save_model()
        self._log(f"fin SFT : {step} pas, modèle sauvegardé dans {self.args.output_dir}")
        return self.history
