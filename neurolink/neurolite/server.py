"""neurolink-server : API HTTP compatible Ollama (mêmes routes, mêmes formats JSON)."""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, backend, default_models_dir, load_model

DEFAULT_PORT = 11434
MAX_PREDICT = 1000000
DEFAULTS = {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "num_predict": MAX_PREDICT,
            "seed": None, "stop": []}
STOP_TAGS = ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>", "</|actual_response|>"]
THINK_OPEN = ["<think>", "<|thinking|>"]
THINK_CLOSE = ["</think>", "</|thinking|>"]
THINK_TAGS = THINK_OPEN + THINK_CLOSE
HIDE_TAGS = ["<|actual_response|>"]
HOLD_TAGS = THINK_TAGS + HIDE_TAGS + STOP_TAGS
CHAT_TEMPLATE = ("{{ if .System }}<|system|>\n{{ .System }}\n<|end|>\n{{ end }}"
                 "{{ range .Messages }}<|{{ .Role }}|>\n{{ .Content }}\n<|end|>\n{{ end }}"
                 "<|assistant|>\n")


# ---------------------------------------------------------------- utilitaires
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def iso_of(path):
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


def norm_name(name):
    name = (name or "").strip()
    return name if ":" in name else name + ":latest"


def folder_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", norm_name(name))


def weights_file(folder):
    for name in ("model.nlx", "model.nnpk"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def read_manifest(folder):
    p = os.path.join(folder, "manifest.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"name": os.path.basename(folder)}


def write_manifest(folder, man):
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def l2(vec):
    n = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / n for x in vec]


def merge_options(*sources):
    opts = dict(DEFAULTS)
    for src in sources:
        for k, v in (src or {}).items():
            if v is not None:
                opts[k] = v
    if isinstance(opts["stop"], str):
        opts["stop"] = [opts["stop"]]
    n = opts["num_predict"]
    if n is None or n < 0 or n > MAX_PREDICT:
        opts["num_predict"] = MAX_PREDICT
    return opts


def build_prompt(messages, system="", think=True):
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}\n<|end|>\n")
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        tag = {"system": "<|system|>", "assistant": "<|assistant|>"}.get(role, "<|user|>")
        parts.append(f"{tag}\n{content}\n<|end|>\n")
    parts.append("<|assistant|>\n")
    if not think:
        parts.append("<think>\n\n</think>\n")
    return "".join(parts)


def details(man):
    fam = man.get("family", "neurolink")
    return {"parent_model": man.get("parent", ""), "format": "nlx", "family": fam,
            "families": [fam], "parameter_size": man.get("parameter_size", "?"),
            "quantization_level": "F32"}


def modelfile_text(man):
    lines = [f"FROM {man.get('parent', man.get('name', ''))}"]
    if man.get("system"):
        lines.append(f'SYSTEM """{man["system"]}"""')
    for k, v in (man.get("parameters") or {}).items():
        for item in (v if isinstance(v, list) else [v]):
            lines.append(f'PARAMETER {k} "{item}"' if k == "stop" else f"PARAMETER {k} {item}")
    return "\n".join(lines) + "\n"


def _coerce(v):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_modelfile(text):
    out = {"from": None, "system": None, "parameters": {}}
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.upper(), rest.strip()
        if cmd == "FROM":
            out["from"] = rest
        elif cmd == "PARAMETER":
            k, _, v = rest.partition(" ")
            v = v.strip().strip('"')
            if k == "stop":
                out["parameters"].setdefault("stop", []).append(v)
            else:
                out["parameters"][k] = _coerce(v)
        elif cmd == "SYSTEM":
            if rest.startswith('"""'):
                acc = rest[3:]
                while '"""' not in acc and i < len(lines):
                    acc += "\n" + lines[i]
                    i += 1
                out["system"] = acc.split('"""')[0].strip()
            else:
                out["system"] = rest.strip('"')
    return out


def _first(buf, tags):
    hits = [(buf.index(t), t) for t in tags if t in buf]
    return min(hits) if hits else (-1, None)


def text_events(model, prompt, opts):
    """Produit (kind, texte) avec kind = 'thinking' | 'content', puis ('done', raison).
    Gère les balises <think>…</think> et les marqueurs d'arrêt du modèle."""
    stops = STOP_TAGS + [s for s in opts["stop"] if s]
    max_hold = max(len(t) for t in HOLD_TAGS + stops)
    buf, in_think, stopped = "", False, False
    gen = model.stream(prompt, max_tokens=opts["num_predict"], temperature=opts["temperature"],
                       top_k=opts["top_k"], top_p=opts["top_p"], seed=opts["seed"])
    for ch in gen:
        buf += ch
        while True:
            spos, _ = _first(buf, stops)
            tpos, ttag = _first(buf, THINK_TAGS + HIDE_TAGS)
            if spos >= 0 and (tpos < 0 or spos < tpos):
                if spos:
                    yield ("thinking" if in_think else "content"), buf[:spos]
                stopped = True
                break
            if tpos >= 0:
                if tpos:
                    yield ("thinking" if in_think else "content"), buf[:tpos]
                if ttag not in HIDE_TAGS:
                    in_think = ttag in THINK_OPEN
                buf = buf[tpos + len(ttag):]
                continue
            break
        if stopped:
            break
        keep = 0
        for n in range(min(len(buf), max_hold - 1), 0, -1):
            tail = buf[-n:]
            if any(t.startswith(tail) for t in HOLD_TAGS + stops):
                keep = n
                break
        cut = len(buf) - keep
        if cut > 0:
            yield ("thinking" if in_think else "content"), buf[:cut]
            buf = buf[cut:]
    gen.close()
    if not stopped and buf:
        yield ("thinking" if in_think else "content"), buf
    yield "done", ("stop" if stopped else "length")


def chunk_obj(mode, name, kind, text):
    th = text if kind == "thinking" else ""
    ct = text if kind == "content" else ""
    obj = {"model": name, "created_at": now_iso()}
    if mode == "chat":
        msg = {"role": "assistant", "content": ct}
        if th:
            msg["thinking"] = th
        obj["message"] = msg
    else:
        obj["response"] = ct
        if th:
            obj["thinking"] = th
    obj["done"] = False
    return obj


def final_obj(mode, name, content, thinking, reason, count, prompt_len, dur):
    obj = {"model": name, "created_at": now_iso()}
    if mode == "chat":
        msg = {"role": "assistant", "content": content}
        if thinking:
            msg["thinking"] = thinking
        obj["message"] = msg
    else:
        obj["response"] = content
        if thinking:
            obj["thinking"] = thinking
    obj.update({"done": True, "done_reason": reason, "total_duration": dur,
                "load_duration": 0, "prompt_eval_count": prompt_len,
                "eval_count": count, "eval_duration": dur})
    return obj


# ---------------------------------------------------------------- registre
class Registry:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.loaded = {}
        self._digests = {}

    def index(self):
        out = {}
        if os.path.isdir(self.root):
            for d in sorted(os.listdir(self.root)):
                folder = os.path.join(self.root, d)
                if weights_file(folder) is not None:
                    out[norm_name(read_manifest(folder).get("name", d))] = folder
        return out

    def find(self, name):
        return self.index().get(norm_name(name))

    def digest(self, folder):
        path = weights_file(folder)
        key = (path, os.path.getmtime(path))
        if key not in self._digests:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._digests[key] = h.hexdigest()
        return self._digests[key]

    def get(self, name):
        name = norm_name(name)
        with self.lock:
            if name in self.loaded:
                return self.loaded[name]
            folder = self.index().get(name)
            if folder is None:
                return None
            entry = {"name": name, "folder": folder, "model": load_model(folder),
                     "man": read_manifest(folder), "lock": threading.Lock(),
                     "loaded_at": time.time()}
            self.loaded[name] = entry
            return entry

    def unload(self, name):
        with self.lock:
            self.loaded.pop(norm_name(name), None)


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"neurolink/{__version__}"

    def log_message(self, fmt, *args):
        if os.environ.get("NEUROLINK_DEBUG"):
            sys.stderr.write("[neurolink] " + (fmt % args) + "\n")

    def send_bytes(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, code, obj):
        self.send_bytes(code, "application/json; charset=utf-8",
                        json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def send_text(self, text, code=200):
        self.send_bytes(code, "text/plain; charset=utf-8", text.encode("utf-8"))

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw.decode("utf-8")) if raw.strip() else {}

    def stream_start(self, ctype):
        self._streaming = True
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def chunk(self, data):
        if data:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def stream_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def ndjson(self, obj):
        self.chunk((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def sse(self, obj):
        self.chunk(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))

    def _dispatch(self, method):
        self._streaming = False
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        fn = ROUTES.get((method, path))
        if fn is None:
            return self.send_json(404, {"error": "404 page not found"})
        try:
            fn(self)
        except json.JSONDecodeError:
            if not self._streaming:
                self.send_json(400, {"error": "JSON invalide"})
        except KeyError as e:
            if not self._streaming:
                self.send_json(400, {"error": f"champ manquant : {e}"})
        except Exception as e:  # noqa: BLE001
            if not self._streaming:
                self.send_json(500, {"error": str(e)})

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def _resolve(h, body):
    name = body.get("model") or body.get("name") or ""
    entry = h.server.registry.get(name)
    if entry is None:
        h.send_json(404, {"error": f"model '{name}' not found"})
        return None
    return entry


def run_text(h, entry, prompt, opts, stream, mode, think):
    name, t0 = entry["name"], time.time()
    parts = {"content": [], "thinking": []}
    reason, count = "stop", 0
    with entry["lock"]:
        if stream:
            h.stream_start("application/x-ndjson")
        for kind, text in text_events(entry["model"], prompt, opts):
            if kind == "done":
                reason = text
                break
            parts[kind].append(text)
            count += len(text)
            if stream:
                h.ndjson(chunk_obj(mode, name, kind, text))
    dur = int((time.time() - t0) * 1e9)
    if stream:
        h.ndjson(final_obj(mode, name, "", "", reason, count, len(prompt), dur))
        h.stream_end()
    else:
        h.send_json(200, final_obj(mode, name, "".join(parts["content"]),
                                   "".join(parts["thinking"]) if think else "",
                                   reason, count, len(prompt), dur))


# ----- routes (format identique à Ollama)
def r_root(h):
    h.send_text("neurolink is running")


def r_version(h):
    h.send_json(200, {"version": __version__})


def r_tags(h):
    reg, models = h.server.registry, []
    for name, folder in reg.index().items():
        man = read_manifest(folder)
        size = os.path.getsize(weights_file(folder))
        models.append({"name": name, "model": name, "modified_at": iso_of(folder),
                       "size": size, "digest": reg.digest(folder), "details": details(man)})
    h.send_json(200, {"models": models})


def r_ps(h):
    reg = h.server.registry
    with reg.lock:
        entries = list(reg.loaded.values())
    out = [{"name": e["name"], "model": e["name"],
            "size": os.path.getsize(weights_file(e["folder"])),
            "digest": reg.digest(e["folder"]), "details": details(e["man"]),
            "expires_at": "0001-01-01T00:00:00Z", "size_vram": 0} for e in entries]
    h.send_json(200, {"models": out})


def r_show(h):
    body = h.read_json()
    name = body.get("model") or body.get("name") or ""
    folder = h.server.registry.find(name)
    if not folder:
        return h.send_json(404, {"error": f"model '{name}' not found"})
    man = read_manifest(folder)
    cfg = man.get("config", {})
    info = {"general.architecture": "neurolink", "general.parameter_count": cfg.get("params", 0),
            "neurolink.context_length": cfg.get("context", 0),
            "neurolink.embedding_length": cfg.get("dim", 0),
            "neurolink.vocab_size": cfg.get("vocab_size", 0)}
    params = "\n".join(f"{k} {json.dumps(v, ensure_ascii=False)}" if isinstance(v, list)
                       else f"{k} {v}" for k, v in (man.get("parameters") or {}).items())
    h.send_json(200, {"modelfile": modelfile_text(man), "parameters": params,
                      "template": CHAT_TEMPLATE, "details": details(man), "model_info": info,
                      "capabilities": man.get("capabilities", ["completion"])})


def r_generate(h):
    body = h.read_json()
    entry = _resolve(h, body)
    if entry is None:
        return
    man = entry["man"]
    think = bool(body.get("think", True))
    if body.get("raw"):
        prompt = body.get("prompt", "")
    else:
        system = body.get("system") or man.get("system", "")
        prompt = build_prompt([{"role": "user", "content": body.get("prompt", "")}], system, think)
    opts = merge_options(man.get("parameters"), body.get("options"))
    run_text(h, entry, prompt, opts, body.get("stream", True), "generate", think)


def r_chat(h):
    body = h.read_json()
    entry = _resolve(h, body)
    if entry is None:
        return
    man = entry["man"]
    think = bool(body.get("think", True))
    prompt = build_prompt(body.get("messages") or [], man.get("system", ""), think)
    opts = merge_options(man.get("parameters"), body.get("options"))
    run_text(h, entry, prompt, opts, body.get("stream", True), "chat", think)


def r_embed(h):
    body = h.read_json()
    entry = _resolve(h, body)
    if entry is None:
        return
    inp = body.get("input", "")
    texts = inp if isinstance(inp, list) else [inp]
    with entry["lock"]:
        vecs = [l2(entry["model"].embed(t)) for t in texts]
    h.send_json(200, {"model": entry["name"], "embeddings": vecs, "total_duration": 0,
                      "load_duration": 0, "prompt_eval_count": 0})


def r_embeddings(h):
    body = h.read_json()
    entry = _resolve(h, body)
    if entry is None:
        return
    with entry["lock"]:
        vec = entry["model"].embed(body.get("prompt", ""))
    h.send_json(200, {"embedding": l2(vec)})


def r_pull(h):
    body = h.read_json()
    name = body.get("model") or body.get("name") or ""
    if h.server.registry.find(name) is None:
        return h.send_json(404, {"error": "pull distant non disponible : utilisez /api/create "
                                          "avec un modèle local"})
    steps = ["pulling manifest", "verifying sha256 digest", "writing manifest", "success"]
    if body.get("stream", True):
        h.stream_start("application/x-ndjson")
        for s in steps:
            h.ndjson({"status": s})
        h.stream_end()
    else:
        h.send_json(200, {"status": "success"})


def r_create(h):
    body = h.read_json()
    name = norm_name(body.get("model") or body.get("name"))
    mf = parse_modelfile(body.get("modelfile") or "")
    base = mf["from"] or body.get("from")
    reg = h.server.registry
    src = reg.find(base) if base else None
    if src is None and base and os.path.isdir(base):
        src = base
    if src is None:
        return h.send_json(404, {"error": f"modèle de base '{base}' introuvable"})
    dst = os.path.join(reg.root, folder_name(name))
    if os.path.abspath(src) == os.path.abspath(dst):
        return h.send_json(400, {"error": "le modèle de destination existe déjà"})
    stream = body.get("stream", True)
    if stream:
        h.stream_start("application/x-ndjson")
        h.ndjson({"status": "reading model metadata"})
    os.makedirs(dst, exist_ok=True)
    for fn in ("model.nlx", "vocab.json"):
        shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
    man = read_manifest(src)
    man.update({"name": name, "parent": norm_name(base), "created_at": now_iso()})
    if mf["system"] is not None:
        man["system"] = mf["system"]
    params = dict(man.get("parameters") or {})
    params.update(mf["parameters"])
    man["parameters"] = params
    if stream:
        h.ndjson({"status": "creating model layer"})
    write_manifest(dst, man)
    with open(os.path.join(dst, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(modelfile_text(man))
    reg.unload(name)
    if stream:
        h.ndjson({"status": "writing manifest"})
        h.ndjson({"status": "success"})
        h.stream_end()
    else:
        h.send_json(200, {"status": "success"})


def r_copy(h):
    body = h.read_json()
    reg = h.server.registry
    src = reg.find(body.get("source", ""))
    if not src:
        return h.send_json(404, {"error": "modèle source introuvable"})
    dst_name = norm_name(body["destination"])
    dst = os.path.join(reg.root, folder_name(dst_name))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    man = read_manifest(dst)
    man["name"] = dst_name
    write_manifest(dst, man)
    h.send_json(200, {})


def r_delete(h):
    body = h.read_json()
    name = norm_name(body.get("model") or body.get("name"))
    reg = h.server.registry
    folder = reg.find(name)
    if not folder:
        return h.send_json(404, {"error": f"model '{name}' not found"})
    shutil.rmtree(folder)
    reg.unload(name)
    h.send_json(200, {})


def r_v1_models(h):
    data = [{"id": n, "object": "model", "created": 0, "owned_by": "neurolink"}
            for n in h.server.registry.index()]
    h.send_json(200, {"object": "list", "data": data})


def r_v1_chat(h):
    body = h.read_json()
    entry = _resolve(h, body)
    if entry is None:
        return
    man = entry["man"]
    prompt = build_prompt(body.get("messages") or [], man.get("system", ""), True)
    opts = merge_options(man.get("parameters"), {"temperature": body.get("temperature"),
                                                  "top_p": body.get("top_p"),
                                                  "num_predict": body.get("max_tokens"),
                                                  "seed": body.get("seed")})
    cid, created, name = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), entry["name"]
    reason, parts, count = "stop", [], 0
    if body.get("stream"):
        h.stream_start("text/event-stream")
        with entry["lock"]:
            for kind, text in text_events(entry["model"], prompt, opts):
                if kind == "done":
                    reason = text
                    break
                h.sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": name,
                       "choices": [{"index": 0, "delta": {"content": text},
                                    "finish_reason": None}]})
        h.sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
               "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        h.chunk(b"data: [DONE]\n\n")
        h.stream_end()
    else:
        with entry["lock"]:
            for kind, text in text_events(entry["model"], prompt, opts):
                if kind == "done":
                    reason = text
                    break
                parts.append(text)
                count += len(text)
        h.send_json(200, {"id": cid, "object": "chat.completion", "created": created,
                          "model": name,
                          "choices": [{"index": 0, "message": {"role": "assistant",
                                                              "content": "".join(parts)},
                                       "finish_reason": reason}],
                          "usage": {"prompt_tokens": len(prompt), "completion_tokens": count,
                                    "total_tokens": len(prompt) + count}})


def r_agents_run(h):
    from .agents import Orchestrator
    body = h.read_json()
    if not body.get("task"):
        return h.send_json(400, {"error": "champ 'task' requis"})
    orch = Orchestrator(model=body.get("model", "neurolink-think-2.1-pro"),
                        roles=body.get("roles") or [],
                        max_subtasks=int(body.get("max_subtasks", 4)),
                        max_tokens=body.get("max_tokens"))
    h.send_json(200, orch.run(body["task"]))


ROUTES = {
    ("GET", "/"): r_root,
    ("HEAD", "/"): r_root,
    ("GET", "/api/version"): r_version,
    ("GET", "/api/tags"): r_tags,
    ("GET", "/api/ps"): r_ps,
    ("POST", "/api/show"): r_show,
    ("POST", "/api/generate"): r_generate,
    ("POST", "/api/chat"): r_chat,
    ("POST", "/api/embed"): r_embed,
    ("POST", "/api/embeddings"): r_embeddings,
    ("POST", "/api/pull"): r_pull,
    ("POST", "/api/create"): r_create,
    ("POST", "/api/copy"): r_copy,
    ("DELETE", "/api/delete"): r_delete,
    ("POST", "/api/delete"): r_delete,
    ("GET", "/v1/models"): r_v1_models,
    ("POST", "/v1/chat/completions"): r_v1_chat,
    ("POST", "/api/agents/run"): r_agents_run,
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="neurolink-server",
                                description="Serveur API style Ollama pour NeuroLink")
    p.add_argument("--host", default=os.environ.get("NEUROLINK_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("NEUROLINK_PORT", DEFAULT_PORT)))
    p.add_argument("--models", default=default_models_dir())
    a = p.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    srv.registry = Registry(a.models)
    print(f"neurolink server {__version__} sur http://{a.host}:{a.port} - modèles : {a.models} - moteur : {backend()}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
