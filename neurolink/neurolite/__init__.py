"""neurolite - bibliothèque Python du noyau C (libneurolite.so).
Si la bibliothèque C ne peut pas être chargée (ex. iOS / a-Shell), le moteur pur Python
(neurolite.pure) prend le relais pour le chargement et l'inférence."""
import ctypes
import json
import math
import os
import random

from .pure import PyNet

__version__ = "0.2.1"
PAD = "\x00"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_lib():
    if os.environ.get("NEUROLITE_FORCE_PURE"):
        raise OSError("mode pur forcé par NEUROLITE_FORCE_PURE")
    errors = []
    for path in (os.environ.get("NEUROLITE_LIB"), os.path.join(_HERE, "libneurolite.so")):
        if path and os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError as e:
                errors.append(str(e))
    raise OSError("bibliothèque C indisponible : " + (" ; ".join(errors) or "fichier absent"))


try:
    _lib = _load_lib()
    _LIB_ERROR = None
except OSError as _e:
    _lib = None
    _LIB_ERROR = str(_e)

if _lib is not None:
    _vp = ctypes.c_void_p
    _ip = ctypes.POINTER(ctypes.c_int)
    _fp = ctypes.POINTER(ctypes.c_float)
    _lib.nl_create.restype = _vp
    _lib.nl_create.argtypes = [ctypes.c_int] * 6 + [ctypes.c_uint]
    _lib.nl_free.argtypes = [_vp]
    _lib.nl_set_verbose.argtypes = [ctypes.c_int]
    _lib.nl_train.restype = ctypes.c_int
    _lib.nl_train.argtypes = [_vp, _ip, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                   ctypes.c_int, ctypes.c_int]
    _lib.nl_predict.restype = ctypes.c_int
    _lib.nl_predict.argtypes = [_vp, _ip, _fp]
    _lib.nl_embed.restype = ctypes.c_int
    _lib.nl_embed.argtypes = [_vp, _ip, _fp]
    _lib.nl_config.argtypes = [_vp, _ip]
    _lib.nl_num_params.restype = ctypes.c_int
    _lib.nl_num_params.argtypes = [_vp]
    _lib.nl_save.restype = ctypes.c_int
    _lib.nl_save.argtypes = [_vp, ctypes.c_char_p]
    _lib.nl_epoch.restype = ctypes.c_float
    _lib.nl_epoch.argtypes = [_vp, _ip, ctypes.c_int, ctypes.c_float]
    _lib.nl_eval.restype = ctypes.c_float
    _lib.nl_eval.argtypes = [_vp, _ip, ctypes.c_int]
    _lib.nl_load.restype = _vp
    _lib.nl_load.argtypes = [ctypes.c_char_p]


def backend():
    """'native' (bibliothèque C) ou 'pure' (Python pur)."""
    return "native" if _lib is not None else "pure"


def default_models_dir():
    return os.environ.get("NEUROLINK_MODELS") or os.path.join(os.path.dirname(_HERE), "models")


def _sample(probs, temperature, top_k, top_p, rng):
    if temperature <= 0:
        return max(range(len(probs)), key=probs.__getitem__)
    w = [(p ** (1.0 / temperature), i) for i, p in enumerate(probs) if p > 0]
    if not w:
        return max(range(len(probs)), key=probs.__getitem__)
    w.sort(reverse=True)
    if top_k and top_k > 0:
        w = w[:top_k]
    if top_p and 0 < top_p < 1:
        total = sum(x for x, _ in w)
        acc, cut = 0.0, []
        for x, i in w:
            cut.append((x, i))
            acc += x
            if acc >= top_p * total:
                break
        w = cut
    total = sum(x for x, _ in w)
    r = rng.random() * total
    for x, i in w:
        r -= x
        if r <= 0:
            return i
    return w[-1][1]


class NeuroLite:
    verbose = True

    def __init__(self, chars, K=16, D=24, H=48, E=4, topk=2, seed=42, _handle=None):
        chars = list(chars)
        if not chars or chars[0] != PAD:
            chars = [PAD] + [c for c in chars if c != PAD]
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.K = K
        if _handle is None:
            if _lib is None:
                raise RuntimeError("création d'un modèle impossible sans libneurolite.so "
                                   "(mode pur : chargement seulement)")
            _handle = _lib.nl_create(len(chars), K, D, H, E, topk, seed)
            if not _handle:
                raise MemoryError("nl_create a échoué (configuration invalide ?)")
        self.h = _handle
        self.pure = isinstance(_handle, PyNet)
        self.manifest = {}
        self.folder = None

    def __del__(self):
        if _lib is not None and not getattr(self, "pure", True) and getattr(self, "h", None):
            _lib.nl_free(self.h)
            self.h = None

    @property
    def backend(self):
        return "pure" if self.pure else "native"

    @classmethod
    def from_text(cls, text, **kw):
        return cls(sorted(set(text)), **kw)

    @property
    def config(self):
        if self.pure:
            return self.h.config
        arr = (ctypes.c_int * 7)()
        _lib.nl_config(self.h, arr)
        V, K, D, H, E, topk, P = list(arr)
        return {"vocab_size": V, "context": K, "dim": D, "hidden": H,
                "experts": E, "topk": topk, "params": P}

    @property
    def num_params(self):
        return self.h.P if self.pure else _lib.nl_num_params(self.h)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def _window(self, ids):
        ids = ids[-self.K:]
        return [0] * (self.K - len(ids)) + ids

    def _probs(self, win):
        if self.pure:
            return self.h.predict(win)
        ctx = (ctypes.c_int * self.K)(*win)
        out = (ctypes.c_float * len(self.chars))()
        _lib.nl_predict(self.h, ctx, out)
        return list(out)

    def epoch(self, text, lr):
        """Une passe d'entraînement à taux constant. Retourne la perte moyenne (nats/caractère)."""
        if self.pure:
            raise RuntimeError("entraînement impossible en mode pur : compilez libneurolite.so")
        ids = self.encode(text)
        if len(ids) <= self.K:
            raise ValueError("texte trop court pour la taille de contexte")
        arr = (ctypes.c_int * len(ids))(*ids)
        return float(_lib.nl_epoch(self.h, arr, len(ids), ctypes.c_float(lr)))

    def loss(self, text):
        """Perte moyenne (entropie croisée, nats/caractère) sur un texte, sans mise à jour."""
        ids = self.encode(text)
        if len(ids) <= self.K:
            raise ValueError("texte trop court pour la taille de contexte")
        if not self.pure:
            arr = (ctypes.c_int * len(ids))(*ids)
            return float(_lib.nl_eval(self.h, arr, len(ids)))
        n, total = len(ids) - self.K, 0.0
        for i in range(n):
            p = self._probs(ids[i:i + self.K])[ids[i + self.K]]
            total += -math.log(max(p, 1e-12))
        return total / n

    def train(self, text, epochs=5, lr=3e-3, warmup=200, batch=1):
        if self.pure:
            raise RuntimeError("entraînement impossible en mode pur : compilez libneurolite.so")
        ids = self.encode(text)
        arr = (ctypes.c_int * len(ids))(*ids)
        _lib.nl_set_verbose(1 if self.verbose else 0)
        if _lib.nl_train(self.h, arr, len(ids), int(epochs), float(lr), int(warmup),
                         int(batch)) != 0:
            raise ValueError("texte trop court pour la taille de contexte")
        return self

    def embed(self, text):
        win = self._window(self.encode(text))
        if self.pure:
            return self.h.embed(win)
        D = self.config["dim"]
        ctx = (ctypes.c_int * self.K)(*win)
        out = (ctypes.c_float * D)()
        _lib.nl_embed(self.h, ctx, out)
        return list(out)

    def stream(self, prompt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.9, seed=None):
        """Génère caractère par caractère (générateur)."""
        rng = random.Random(seed)
        win = self._window(self.encode(prompt))
        for _ in range(int(max_tokens)):
            probs = self._probs(win)
            probs[0] = 0.0  # le token de bourrage n'est jamais émis
            tok = _sample(probs, temperature, top_k, top_p, rng)
            win = win[1:] + [tok]
            yield self.chars[tok]

    def generate(self, prompt="", max_tokens=128, **opts):
        return prompt + "".join(self.stream(prompt, max_tokens=max_tokens, **opts))

    def save(self, folder, manifest=None):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "model.nlx").encode()
        if self.pure:
            self.h.save(path.decode())
        elif _lib.nl_save(self.h, path) != 0:
            raise IOError("nl_save a échoué")
        with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"format": "neurolite-v2", "chars": self.chars, "K": self.K},
                      f, ensure_ascii=False)
        if manifest is not None:
            with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        self.folder = folder
        return folder


def load_model(name_or_path, models_dir=None):
    """Charge un modèle depuis un dossier, ou par nom dans le dossier des modèles."""
    candidates = [name_or_path]
    if name_or_path.endswith(":latest"):
        candidates.append(name_or_path[:-len(":latest")])
    candidates.append(name_or_path.replace(":", "_"))
    folder = None
    for c in candidates:
        if os.path.isdir(c):
            folder = c
            break
        alt = os.path.join(models_dir or default_models_dir(), c)
        if os.path.isdir(alt):
            folder = alt
            break
    if folder is None:
        raise FileNotFoundError(f"modèle introuvable : {name_or_path}")
    if os.path.exists(os.path.join(folder, "model.nnpk")):
        with open(os.path.join(folder, "vocab.json"), encoding="utf-8") as f:
            fmt = json.load(f).get("format")
        if fmt == "transformer-v1":
            from .core_adapter import load_transformer_model
            return load_transformer_model(folder)
        from .booster import load_nn_model
        return load_nn_model(folder)
    with open(os.path.join(folder, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    weights = os.path.join(folder, "model.nlx")
    if _lib is None:
        h = PyNet.load(weights)
    else:
        h = _lib.nl_load(weights.encode())
        if not h:
            raise IOError(f"nl_load a échoué pour {folder}")
    model = NeuroLite(vocab["chars"], K=vocab["K"], _handle=h)
    man = os.path.join(folder, "manifest.json")
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            model.manifest = json.load(f)
    model.folder = folder
    return model


__all__ = ["NeuroLite", "load_model", "default_models_dir", "backend", "__version__"]
