#!/usr/bin/env python3
"""Entraîne neurolink-think-2.1-pro sur corpus/neurolink_think.txt et exporte le modèle."""
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neurolite import NeuroLite  # noqa: E402

NAME = "neurolink-think-2.1-pro:latest"
FOLDER = os.path.join(ROOT, "models", "neurolink-think-2.1-pro")
SYSTEM = "Tu es NeuroLink Think, un assistant qui réfléchit étape par étape avant de répondre."
PARAMS = {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "num_predict": 256,
          "stop": ["<|user|>"]}


def main():
    corpus = os.path.join(ROOT, "corpus", "neurolink_think.txt")
    with open(corpus, encoding="utf-8") as f:
        text = f.read()
    epochs = int(os.environ.get("EPOCHS", "12"))
    model = NeuroLite.from_text(text, K=16, D=24, H=48, E=4, topk=2, seed=2026)
    print(f"paramètres : {model.num_params}", flush=True)
    t0 = time.time()
    model.train(text, epochs=epochs, lr=3e-3, warmup=500)
    seconds = round(time.time() - t0, 1)

    manifest = {
        "name": NAME,
        "family": "neurolink",
        "format": "nlx",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameter_size": f"{model.num_params / 1e6:.2f}M",
        "capabilities": ["completion", "thinking", "embedding"],
        "config": model.config,
        "train": {"corpus": "corpus/neurolink_think.txt", "chars": len(text),
                  "epochs": epochs, "lr": 3e-3, "warmup": 500, "seconds": seconds},
        "system": SYSTEM,
        "parameters": PARAMS,
    }
    model.save(FOLDER, manifest)
    with open(os.path.join(FOLDER, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(f'FROM {NAME}\nSYSTEM """{SYSTEM}"""\n'
                f'PARAMETER temperature {PARAMS["temperature"]}\n'
                f'PARAMETER top_k {PARAMS["top_k"]}\nPARAMETER top_p {PARAMS["top_p"]}\n'
                f'PARAMETER num_predict {PARAMS["num_predict"]}\n'
                f'PARAMETER stop "{PARAMS["stop"][0]}"\n')
    print(f"modèle exporté : {FOLDER} ({seconds} s)", flush=True)
    sample = model.generate("<|user|>\nBonjour, qui es-tu ?\n<|end|>\n<|assistant|>\n",
                            max_tokens=160, temperature=0.7, seed=3)
    print("--- échantillon ---")
    print(sample)


if __name__ == "__main__":
    main()
