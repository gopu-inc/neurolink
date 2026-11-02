#!/usr/bin/env python3
"""Lecteur parquet minimal en Python pur (sans pyarrow).

Gère : métadonnées Thrift, pages DATA_PAGE v1/v2 et DICTIONARY_PAGE, encodages PLAIN et
RLE_DICTIONARY, compressions UNCOMPRESSED, SNAPPY et GZIP, colonnes BYTE_ARRAY (texte) à
un niveau (colonnes optionnelles ou requises). Ne gère pas ZSTD ni les types imbriqués.
Usage : python scripts/read_parquet.py fichier.parquet [--column NOM] [--out sortie.txt]
"""
import argparse
import struct
import sys
import zlib

BYTE_ARRAY = 6
PLAIN, PLAIN_DICT, RLE_DICT = 0, 2, 8


# ----- Thrift compact --------------------------------------------------------
class Thrift:
    def __init__(self, b, pos=0):
        self.b, self.p = b, pos

    def byte(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self):
        r, s = 0, 0
        while True:
            x = self.b[self.p]
            self.p += 1
            r |= (x & 0x7F) << s
            if not x & 0x80:
                return r
            s += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def binary(self):
        n = self.varint()
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def value(self, t):
        if t == 1:
            return True
        if t == 2:
            return False
        if t == 3:
            return self.byte()
        if t in (4, 5, 6):
            return self.zigzag()
        if t == 7:
            v = struct.unpack("<d", self.b[self.p:self.p + 8])[0]
            self.p += 8
            return v
        if t == 8:
            return self.binary()
        if t in (9, 10):
            return self.list()
        if t == 11:
            return self.map()
        if t == 12:
            return self.struct()
        raise ValueError(f"type thrift inconnu {t}")

    def list(self):
        h = self.byte()
        size, et = h >> 4, h & 0x0F
        if size == 15:
            size = self.varint()
        if et in (1, 2):
            return [self.byte() == 1 for _ in range(size)]
        return [self.value(et) for _ in range(size)]

    def map(self):
        size = self.varint()
        if size == 0:
            return {}
        kv = self.byte()
        kt, vt = kv >> 4, kv & 0x0F
        return {self.value(kt): self.value(vt) for _ in range(size)}

    def struct(self):
        out, fid = {}, 0
        while True:
            h = self.byte()
            if h == 0:
                return out
            delta, t = h >> 4, h & 0x0F
            fid = fid + delta if delta else self.zigzag()
            out[fid] = (t == 1) if t in (1, 2) else self.value(t)


# ----- Snappy ----------------------------------------------------------------
def snappy_decompress(b):
    p, shift, n = 0, 0, 0
    while True:
        x = b[p]
        p += 1
        n |= (x & 0x7F) << shift
        if not x & 0x80:
            break
        shift += 7
    out = bytearray()
    while p < len(b):
        tag = b[p]
        p += 1
        kind = tag & 3
        if kind == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(b[p:p + nb], "little")
                p += nb
            ln += 1
            out += b[p:p + ln]
            p += ln
            continue
        if kind == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | b[p]
            p += 1
        elif kind == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[p:p + 2], "little")
            p += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[p:p + 4], "little")
            p += 4
        start = len(out) - off
        if off >= ln:
            out += out[start:start + ln]
        else:
            for i in range(ln):
                out.append(out[start + i])
    return bytes(out)


def decompress(raw, codec, size):
    if codec == 0:
        return raw
    if codec == 1:
        return snappy_decompress(raw)
    if codec == 2:
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    raise NotImplementedError(f"compression {codec} non gérée (ZSTD/LZ4/BROTLI : installez pyarrow)")


# ----- RLE / bit-packing hybride ---------------------------------------------
def rle_bp(b, pos, end, width, count):
    out, p = [], pos
    if width == 0:
        return [0] * count, end
    per_run = (width + 7) // 8
    while len(out) < count and p < end:
        h, s = 0, 0
        while True:
            x = b[p]
            p += 1
            h |= (x & 0x7F) << s
            if not x & 0x80:
                break
            s += 7
        if h & 1:
            groups = h >> 1
            nbytes = groups * width
            big = int.from_bytes(b[p:p + nbytes], "little")
            mask = (1 << width) - 1
            out.extend((big >> (i * width)) & mask for i in range(groups * 8))
            p += nbytes
        else:
            run = h >> 1
            v = int.from_bytes(b[p:p + per_run], "little")
            p += per_run
            out.extend([v] * run)
    return out[:count], end


def plain_byte_arrays(b, count):
    out, p = [], 0
    for _ in range(count):
        n = int.from_bytes(b[p:p + 4], "little")
        p += 4
        out.append(b[p:p + n])
        p += n
    return out


# ----- Fichier ---------------------------------------------------------------
def read_footer(data):
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("pas un fichier parquet")
    flen = int.from_bytes(data[-8:-4], "little")
    return Thrift(data[-8 - flen:-8]).struct()


def schema_leaves(schema):
    """Feuilles du schéma avec leurs niveaux max de définition et de répétition."""
    idx = 1
    leaves = []

    def walk(n_children, path, d, r):
        nonlocal idx
        for _ in range(n_children):
            el = schema[idx]
            idx += 1
            name = el[4].decode("utf-8")
            rep = el.get(3, 0)
            nd, nr = d, r
            if rep == 1:
                nd += 1
            elif rep == 2:
                nd += 1
                nr += 1
            if el.get(5):
                walk(el[5], path + [name], nd, nr)
            else:
                leaves.append({"path": path + [name], "type": el.get(1), "def": nd, "rep": nr})

    walk(schema[0].get(5, 0), [], 0, 0)
    return leaves


def read_leaf(data, chunk, leaf):
    """Retourne (niveaux de répétition, niveaux de définition, valeurs) alignés par entrée."""
    cm = chunk[3]
    codec = cm[4]
    start = cm.get(11) or cm[9]
    end = start + cm[7]
    max_def, max_rep = leaf["def"], leaf["rep"]
    dictionary = None
    reps, defs, values = [], [], []
    pos = start
    while pos < end:
        th = Thrift(data, pos)
        hdr = th.struct()
        pos = th.p
        csize, usize = hdr[3], hdr[2]
        payload = decompress(data[pos:pos + csize], codec, usize)
        pos += csize
        ptype = hdr[1]
        if ptype == 2:
            dictionary = plain_byte_arrays(payload, hdr[7][1])
            continue
        if ptype == 0:
            dph = hdr[5]
            nvals, enc = dph[1], dph[2]
            p = 0
            if max_rep:
                rl = int.from_bytes(payload[p:p + 4], "little")
                pr, _ = rle_bp(payload, p + 4, p + 4 + rl, max_rep.bit_length(), nvals)
                p += 4 + rl
            else:
                pr = [0] * nvals
            if max_def:
                dl = int.from_bytes(payload[p:p + 4], "little")
                pd, _ = rle_bp(payload, p + 4, p + 4 + dl, max_def.bit_length(), nvals)
                p += 4 + dl
            else:
                pd = [0] * nvals
            vals_raw = payload[p:]
        elif ptype == 3:
            dph = hdr[8]
            nvals, enc = dph[1], dph[4]
            dl_len, rl_len, is_comp = dph.get(5, 0), dph.get(6, 0), dph.get(7, True)
            pr = rle_bp(payload, 0, rl_len, max_rep.bit_length(), nvals)[0] if max_rep else [0] * nvals
            pd = (rle_bp(payload, rl_len, rl_len + dl_len, max_def.bit_length(), nvals)[0]
                  if max_def else [0] * nvals)
            body = payload[rl_len + dl_len:]
            vals_raw = decompress(body, codec, usize) if is_comp else body
        else:
            continue
        nonnull = sum(1 for d in pd if d == max_def)
        if enc == PLAIN:
            page_vals = plain_byte_arrays(vals_raw, nonnull)
        elif enc in (PLAIN_DICT, RLE_DICT):
            width = vals_raw[0]
            idx, _ = rle_bp(vals_raw, 1, len(vals_raw), width, nonnull)
            page_vals = [dictionary[i] for i in idx]
        else:
            raise NotImplementedError(f"encodage {enc} non géré")
        it = iter(page_vals)
        for d in pd:
            values.append(next(it).decode("utf-8", "replace") if d == max_def else None)
        reps.extend(pr)
        defs.extend(pd)
    return reps, defs, values


def assemble_messages(fields):
    """fields : liste (nom, reps, defs, valeurs, max_def) de feuilles d'une même liste de structs.
    Retourne une liste de lignes ; chaque ligne est une liste de messages {nom: valeur}."""
    rows = []
    reps0 = fields[0][1]
    msg = None
    for i in range(len(reps0)):
        r = reps0[i]
        if r == 0:
            rows.append([])
        if r <= 1:
            msg = {}
            rows[-1].append(msg)
        for name, _reps, defs, vals, md in fields:
            msg[name] = vals[i] if defs[i] == md else None
    return rows


def extract(path):
    data = open(path, "rb").read()
    meta = read_footer(data)
    leaves = schema_leaves(meta[2])
    result = {}
    for k, leaf in enumerate(leaves):
        name = ".".join(leaf["path"])
        R, D, V = [], [], []
        for rg in meta[4]:
            r, d, v = read_leaf(data, rg[1][k], leaf)
            R += r
            D += d
            V += v
        result[name] = (R, D, V, leaf["def"])
    return meta, leaves, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None, help="fichier JSON de sortie")
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()
    meta, leaves, result = extract(args.path)
    print(f"lignes : {meta[3]} | groupes : {len(meta[4])}")
    for leaf in leaves:
        print("  colonne :", ".".join(leaf["path"]), "| type", leaf["type"],
              "| def max", leaf["def"], "| rep max", leaf["rep"])

    prefix = "messages.list.element."
    msg_names = [n for n in result if n.startswith(prefix)]
    if msg_names:
        fields = []
        for n in msg_names:
            R, D, V, md = result[n]
            fields.append((n[len(prefix):], R, D, V, md))
        rows = assemble_messages(fields)
    else:
        name = next(n for n in result if result[n][3] >= 0)
        R, D, V, md = result[name]
        rows = [v for v in V if v]
    print(f"lignes reconstruites : {len(rows)}")
    for row in rows[: args.show]:
        print("  ligne :", repr(row)[:300])
    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
