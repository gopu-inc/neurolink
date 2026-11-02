"""neurolite.nn : petit module de réseaux de neurones à la manière de torch.nn.

Différences avec PyTorch (à connaître) :
  * pas d'autograd général : chaque module implémente forward() et backward(), donc la
    rétropropagation est explicite et vérifiable ;
  * les calculs lourds (produits matriciels, Adam, softmax + perte) passent par
    libneurolite.so (csrc/nlops.c) ; sans la bibliothèque, un repli Python existe mais est lent ;
  * tensors = listes plates de float32 + forme (classe Tensor minimale).
"""
import ctypes
import json
import math
import os
import random
import struct
import sys
from array import array

from . import _lib as _core  # bibliothèque C principale (peut être None)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_ops():
    for name in ("libneurolite.so",):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            try:
                lib = ctypes.CDLL(p)
                if hasattr(lib, "nl_gemm"):
                    return lib
            except OSError:
                pass
    return None


_ops = _load_ops()
if _ops is not None:
    _fp = ctypes.POINTER(ctypes.c_float)
    _ip = ctypes.POINTER(ctypes.c_int)
    _ops.nl_gemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_float, _fp, _fp, ctypes.c_float, _fp]
    _ops.nl_tanh_fwd.argtypes = [_fp, _fp, ctypes.c_long]
    _ops.nl_tanh_bwd.argtypes = [_fp, _fp, _fp, ctypes.c_long]
    _ops.nl_softmax_ce.restype = ctypes.c_float
    _ops.nl_softmax_ce.argtypes = [_fp, _ip, _fp, ctypes.c_int, ctypes.c_int]
    _ops.nl_causal_softmax.argtypes = [_fp, ctypes.c_int]
    _ops.nl_softmax_rows_bwd.argtypes = [_fp, _fp, _fp, ctypes.c_int]
    _ops.nl_add_bias.argtypes = [_fp, _fp, ctypes.c_int, ctypes.c_int]
    _ops.nl_bias_grad.argtypes = [_fp, _fp, ctypes.c_int, ctypes.c_int]
    _ops.nl_axpy.argtypes = [ctypes.c_float, _fp, _fp, ctypes.c_long]
    _ops.nl_adam.argtypes = [_fp, _fp, _fp, _fp, ctypes.c_long, ctypes.c_float, ctypes.c_float,
                             ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                             ctypes.c_float]


def native():
    return _ops is not None


def _ptr(a):
    return a.buf_ptr


class Tensor:
    """Tenseur float32 plat : données dans un array('f'), forme (tuple)."""

    def __init__(self, shape, data=None):
        self.shape = tuple(shape)
        n = 1
        for s in self.shape:
            n *= s
        self.size = n
        self.data = array("f", data) if data is not None else array("f", bytes(4 * n))
        self.buf_ptr = ctypes.cast(self.data.buffer_info()[0], ctypes.POINTER(ctypes.c_float))

    def zeros_like(self):
        return Tensor(self.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def gemm(ta, tb, M, N, K, A, B, C, alpha=1.0, beta=0.0):
    """C = alpha*op(A)@op(B) + beta*C, sur des Tensor 2D (ou des buffers plats)."""
    if _ops is not None:
        _ops.nl_gemm(int(ta), int(tb), M, N, K, alpha, A.buf_ptr, B.buf_ptr, beta, C.buf_ptr)
        return C
    # repli Python (lent)
    a, b, c = A.data, B.data, C.data
    if beta == 0.0:
        for i in range(M * N):
            c[i] = 0.0
    elif beta != 1.0:
        for i in range(M * N):
            c[i] *= beta
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                av = a[k * M + i] if ta else a[i * K + k]
                bv = b[j * K + k] if tb else b[k * N + j]
                s += av * bv
            c[i * N + j] += alpha * s
    return C


class Parameter(Tensor):
    def __init__(self, shape, data=None):
        super().__init__(shape, data)
        self.grad = Tensor(shape)
        self.m = Tensor(shape)
        self.v = Tensor(shape)


class Module:
    training = True

    def parameters(self):
        out = []
        for v in vars(self).values():
            if isinstance(v, Parameter):
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        out.extend(item.parameters())
        return out

    def named_parameters(self, prefix=""):
        out = []
        for k, v in vars(self).items():
            name = f"{prefix}{k}"
            if isinstance(v, Parameter):
                out.append((name, v))
            elif isinstance(v, Module):
                out.extend(v.named_parameters(name + "."))
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        out.extend(item.named_parameters(f"{name}.{i}."))
        return out

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.grad.data = array("f", bytes(4 * p.size))
            p.grad.buf_ptr = ctypes.cast(p.grad.data.buffer_info()[0], ctypes.POINTER(ctypes.c_float))

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Linear(Module):
    """y = x @ W + b. x : (B, in) -> (B, out)."""

    def __init__(self, n_in, n_out, bias=True, seed=0):
        rng = random.Random(seed)
        bound = 1.0 / math.sqrt(n_in)
        self.n_in, self.n_out = n_in, n_out
        self.weight = Parameter((n_in, n_out), [rng.uniform(-bound, bound) for _ in range(n_in * n_out)])
        self.bias = Parameter((n_out,), [rng.uniform(-bound, bound) for _ in range(n_out)]) if bias else None
        self._x = None

    def forward(self, x):
        self._x = x
        B = x.shape[0]
        y = Tensor((B, self.n_out))
        gemm(0, 0, B, self.n_out, self.n_in, x, self.weight, y)
        if self.bias is not None:
            if _ops is not None:
                _ops.nl_add_bias(y.buf_ptr, self.bias.buf_ptr, B, self.n_out)
            else:
                b, yd = self.bias.data, y.data
                for i in range(B):
                    off = i * self.n_out
                    for j in range(self.n_out):
                        yd[off + j] += b[j]
        return y

    def backward(self, dy):
        x = self._x
        B = x.shape[0]
        gemm(1, 0, self.n_in, self.n_out, B, x, dy, self.weight.grad, beta=1.0)
        if self.bias is not None:
            if _ops is not None:
                _ops.nl_bias_grad(dy.buf_ptr, self.bias.grad.buf_ptr, B, self.n_out)
            else:
                g, dd = self.bias.grad.data, dy.data
                for i in range(B):
                    off = i * self.n_out
                    for j in range(self.n_out):
                        g[j] += dd[off + j]
        dx = Tensor((B, self.n_in))
        gemm(0, 1, B, self.n_in, self.n_out, dy, self.weight, dx)
        return dx


class Tanh(Module):
    def forward(self, x):
        y = Tensor(x.shape)
        if _ops is not None:
            _ops.nl_tanh_fwd(x.buf_ptr, y.buf_ptr, x.size)
        else:
            for i in range(x.size):
                y.data[i] = math.tanh(x.data[i])
        self._y = y
        return y

    def backward(self, dy):
        dx = Tensor(dy.shape)
        if _ops is not None:
            _ops.nl_tanh_bwd(self._y.buf_ptr, dy.buf_ptr, dx.buf_ptr, dy.size)
        else:
            for i in range(dy.size):
                dx.data[i] = dy.data[i] * (1 - self._y.data[i] ** 2)
        return dx


class Embedding(Module):
    def __init__(self, n_vocab, dim, seed=0):
        rng = random.Random(seed)
        self.weight = Parameter((n_vocab, dim), [rng.uniform(-0.1, 0.1) for _ in range(n_vocab * dim)])
        self.dim = dim
        self._idx = None

    def forward(self, idx_rows):
        """idx_rows : liste de B listes de K indices -> Tensor (B, K*dim)."""
        B, K = len(idx_rows), len(idx_rows[0])
        out = Tensor((B, K * self.dim))
        w, o, d = self.weight.data, out.data, self.dim
        for b, row in enumerate(idx_rows):
            for k, t in enumerate(row):
                src, dst = t * d, b * K * d + k * d
                o[dst:dst + d] = w[src:src + d]
        self._idx = idx_rows
        return out

    def backward(self, dy):
        g, dd, d = self.weight.grad.data, dy.data, self.dim
        K = len(self._idx[0])
        for b, row in enumerate(self._idx):
            for k, t in enumerate(row):
                src, dst = b * K * d + k * d, t * d
                for j in range(d):
                    g[dst + j] += dd[src + j]
        return None


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy


class MoE(Module):
    """Mélange dense d'experts sur les logits : y = sum_e softmax(gate(x))_e * expert_e(x)."""

    def __init__(self, n_in, n_out, n_experts=4, hidden=512, seed=0):
        self.gate = Linear(n_in, n_experts, seed=seed)
        self.experts = [Sequential(Linear(n_in, hidden, seed=seed + 1 + e), Tanh(),
                                   Linear(hidden, n_out, seed=seed + 100 + e))
                        for e in range(n_experts)]
        self.n_experts, self.n_out = n_experts, n_out

    def forward(self, x):
        B = x.shape[0]
        g = self.gate.forward(x)
        p = Tensor((B, self.n_experts))
        for b in range(B):
            row = g.data[b * self.n_experts:(b + 1) * self.n_experts]
            mx = max(row)
            e = [math.exp(v - mx) for v in row]
            s = sum(e)
            for j in range(self.n_experts):
                p.data[b * self.n_experts + j] = e[j] / s
        self._p = p
        outs = [ex.forward(x) for ex in self.experts]
        self._outs = outs
        y = Tensor((B, self.n_out))
        for e in range(self.n_experts):
            od, yd, pd = outs[e].data, y.data, p.data
            for b in range(B):
                w = pd[b * self.n_experts + e]
                off = b * self.n_out
                for j in range(self.n_out):
                    yd[off + j] += w * od[off + j]
        return y

    def backward(self, dy):
        B = dy.shape[0]
        p, outs = self._p, self._outs
        dp = Tensor((B, self.n_experts))
        for e in range(self.n_experts):
            od, dd = outs[e].data, dy.data
            for b in range(B):
                off = b * self.n_out
                dp.data[b * self.n_experts + e] = sum(od[off + j] * dd[off + j] for j in range(self.n_out))
        dg = Tensor((B, self.n_experts))
        for b in range(B):
            pb = p.data[b * self.n_experts:(b + 1) * self.n_experts]
            dpb = dp.data[b * self.n_experts:(b + 1) * self.n_experts]
            mean = sum(pi * di for pi, di in zip(pb, dpb))
            for j in range(self.n_experts):
                dg.data[b * self.n_experts + j] = pb[j] * (dpb[j] - mean)
        dx = self.gate.backward(dg)
        for e, ex in enumerate(self.experts):
            de = Tensor((B, self.n_out))
            pd, dd, ded = p.data, dy.data, de.data
            for b in range(B):
                w = pd[b * self.n_experts + e]
                off = b * self.n_out
                for j in range(self.n_out):
                    ded[off + j] = w * dd[off + j]
            dxe = ex.backward(de)
            for i in range(dx.size):
                dx.data[i] += dxe.data[i]
        return dx


class CrossEntropy:
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        B, V = logits.shape
        dl = Tensor((B, V))
        if _ops is not None:
            tgt = (ctypes.c_int * B)(*targets)
            loss = _ops.nl_softmax_ce(logits.buf_ptr, tgt, dl.buf_ptr, B, V)
        else:
            total = 0.0
            for b in range(B):
                row = logits.data[b * V:(b + 1) * V]
                mx = max(row)
                e = [math.exp(v - mx) for v in row]
                s = sum(e)
                for j in range(V):
                    dl.data[b * V + j] = (e[j] / s) / B
                total += -math.log(e[targets[b]] / s + 1e-30)
                dl.data[b * V + targets[b]] -= 1.0 / B
            loss = total / B
        self._dl = dl
        return float(loss)

    def backward(self):
        return self._dl


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, weight_decay
        self.t = 0

    def step(self, lr=None):
        if _ops is None:
            raise RuntimeError("Adam nécessite libneurolite.so (nl_adam)")
        self.t += 1
        lr = self.lr if lr is None else lr
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p in self.params:
            _ops.nl_adam(p.buf_ptr, p.grad.buf_ptr, p.m.buf_ptr, p.v.buf_ptr, p.size,
                         lr, self.b1, self.b2, self.eps, c1, c2, self.wd)


def save_state(module, path, extra=None):
    """Format NNPK : en-tête JSON (noms, formes, décalages) puis floats little-endian."""
    params = module.named_parameters()
    index, blobs, off = [], [], 0
    for name, p in params:
        index.append({"name": name, "shape": list(p.shape), "offset": off, "size": p.size})
        blobs.append(p.data.tobytes())
        off += p.size
    header = json.dumps({"format": "NNPK-1", "params": index, "extra": extra or {}}).encode()
    with open(path, "wb") as f:
        f.write(b"NNPK" + struct.pack("<I", len(header)) + header)
        for b in blobs:
            f.write(b)


def load_state(module, path):
    with open(path, "rb") as f:
        if f.read(4) != b"NNPK":
            raise IOError("format NNPK attendu")
        n = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(n).decode())
        raw = array("f")
        raw.frombytes(f.read())
    if sys.byteorder == "big":
        raw.byteswap()
    byname = dict(module.named_parameters())
    for entry in meta["params"]:
        p = byname[entry["name"]]
        if p.size != entry["size"]:
            raise ValueError(f"taille différente pour {entry['name']}")
        p.data[:] = raw[entry["offset"]:entry["offset"] + entry["size"]]
        p.buf_ptr = ctypes.cast(p.data.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
    return meta.get("extra", {})



class CausalSelfAttention(Module):
    """Attention multi-têtes causale, rétropropagation explicite.
    Entrée/sortie : Tensor (B*T, dim), chaque séquence fait T positions.
    Softmax causale et sa rétropropagation passent par libneurolite.so si disponible."""

    def __init__(self, dim, heads=2, seed=0):
        assert dim % heads == 0, "dim doit être divisible par heads"
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.q = Linear(dim, dim, bias=False, seed=seed)
        self.k = Linear(dim, dim, bias=False, seed=seed + 1)
        self.v = Linear(dim, dim, bias=False, seed=seed + 2)
        self.o = Linear(dim, dim, bias=False, seed=seed + 3)

    def _gather(self, src, b, h, T):
        dh, dim = self.dh, self.dim
        out = Tensor((T, dh))
        base = b * T * dim + h * dh
        for t in range(T):
            out.data[t * dh:(t + 1) * dh] = src.data[base + t * dim: base + t * dim + dh]
        return out

    def _scatter(self, dst, src, b, h, T):
        dh, dim = self.dh, self.dim
        base = b * T * dim + h * dh
        for t in range(T):
            dst.data[base + t * dim: base + t * dim + dh] = src.data[t * dh:(t + 1) * dh]

    def forward(self, x, T):
        B = x.shape[0] // T
        self._B, self._T = B, T
        q, k, v = self.q.forward(x), self.k.forward(x), self.v.forward(x)
        dh = self.dh
        scale = 1.0 / math.sqrt(dh)
        out = Tensor((B * T, self.dim))
        self._cache = {}
        for b in range(B):
            for h in range(self.heads):
                Q, K, V = self._gather(q, b, h, T), self._gather(k, b, h, T), self._gather(v, b, h, T)
                S = Tensor((T, T))
                gemm(0, 1, T, T, dh, Q, K, S, alpha=scale)
                if _ops is not None:
                    _ops.nl_causal_softmax(S.buf_ptr, T)
                else:
                    row = S.data
                    for i in range(T):
                        for j in range(i + 1, T):
                            row[i * T + j] = -1e30
                        mx = max(row[i * T:(i + 1) * T])
                        e = [math.exp(row[i * T + j] - mx) for j in range(T)]
                        s = sum(e)
                        for j in range(T):
                            row[i * T + j] = e[j] / s
                O = Tensor((T, dh))
                gemm(0, 0, T, dh, T, S, V, O)
                self._scatter(out, O, b, h, T)
                self._cache[(b, h)] = (Q, K, V, S)
        return self.o.forward(out)

    def backward(self, dy):
        B, T, dh, dim = self._B, self._T, self.dh, self.dim
        scale = 1.0 / math.sqrt(dh)
        dcat = self.o.backward(dy)
        dq, dk, dv = (Tensor((B * T, dim)) for _ in range(3))
        for b in range(B):
            for h in range(self.heads):
                Q, K, V, P = self._cache[(b, h)]
                dO = self._gather(dcat, b, h, T)
                dP = Tensor((T, T))
                gemm(0, 1, T, T, dh, dO, V, dP)
                dV = Tensor((T, dh))
                gemm(1, 0, T, dh, T, P, dO, dV)
                dS = Tensor((T, T))
                if _ops is not None:
                    _ops.nl_softmax_rows_bwd(P.buf_ptr, dP.buf_ptr, dS.buf_ptr, T)
                else:
                    for i in range(T):
                        pr, dpr = P.data[i * T:(i + 1) * T], dP.data[i * T:(i + 1) * T]
                        inner = sum(pr[j] * dpr[j] for j in range(T))
                        for j in range(T):
                            dS.data[i * T + j] = pr[j] * (dpr[j] - inner) if j <= i else 0.0
                dQ = Tensor((T, dh))
                gemm(0, 0, T, dh, T, dS, K, dQ, alpha=scale)
                dK = Tensor((T, dh))
                gemm(1, 0, T, dh, T, dS, Q, dK, alpha=scale)
                self._scatter(dq, dQ, b, h, T)
                self._scatter(dk, dK, b, h, T)
                self._scatter(dv, dV, b, h, T)
        dx = self.q.backward(dq)
        dxk = self.k.backward(dk)
        dxv = self.v.backward(dv)
        if _ops is not None:
            _ops.nl_axpy(1.0, dxk.buf_ptr, dx.buf_ptr, dx.size)
            _ops.nl_axpy(1.0, dxv.buf_ptr, dx.buf_ptr, dx.size)
        else:
            for i in range(dx.size):
                dx.data[i] += dxk.data[i] + dxv.data[i]
        return dx
