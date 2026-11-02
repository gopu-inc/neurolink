"""Test de fumée : python tests/smoke_test.py [http://127.0.0.1:11434]"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:11434"
MODEL = "neurolink-think-2.1-pro"


def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


assert get("/api/version")["version"]
names = [m["name"] for m in get("/api/tags")["models"]]
assert f"{MODEL}:latest" in names, names
print("tags OK :", names)

out = json.loads(post("/api/generate", {"model": MODEL, "prompt": "Bonjour",
                                        "stream": False, "options": {"seed": 1, "num_predict": 60}}))
assert out["done"] is True
print("generate OK :", repr(out["response"][:120]), "| thinking :", repr(out.get("thinking", ""))[:80])

lines = [json.loads(x) for x in post("/api/chat", {
    "model": MODEL, "messages": [{"role": "user", "content": "Combien font 12 fois 7 ?"}],
    "stream": True, "options": {"seed": 2, "num_predict": 80}}).splitlines() if x.strip()]
assert lines[-1]["done"] is True and len(lines) > 1
print(f"chat stream OK : {len(lines)} lignes NDJSON, raison = {lines[-1]['done_reason']}")

emb = json.loads(post("/api/embed", {"model": MODEL, "input": "réseau de neurones"}))
assert len(emb["embeddings"][0]) == 24
print("embed OK : dim", len(emb["embeddings"][0]))
print("TOUS LES TESTS PASSENT")
