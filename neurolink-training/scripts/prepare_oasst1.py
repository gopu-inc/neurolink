#!/usr/bin/env python3
"""Download and filter OASST1 into French, assistant-style JSONL conversations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
ARCHIVE_NAME = "2023-04-12_oasst_ready.trees.jsonl.gz"
DATASET_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    f"{DATASET_REVISION}/{ARCHIVE_NAME}?download=true"
)
DEFAULT_OUTPUT = ROOT / "generated-artifacts/neurolink/datasets/oasst1-fr"
SYSTEM_MESSAGE = (
    "Tu es NeuroLink, un assistant utile, clair et respectueux. "
    "Réponds dans la langue de l'utilisateur, en français lorsqu'il s'exprime en français. "
    "Si tu ne sais pas, dis-le."
)
BLOCKING_LABELS = {
    "pii",
    "spam",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "fails_task",
    "lang_mismatch",
}


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroLink-training/1.0 (dataset preparation)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _label_value(labels: Any, name: str) -> float | None:
    if isinstance(labels, dict):
        value = labels.get(name)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, dict) and item.get("name") == name:
                value = item.get("value")
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def _blocked(node: dict[str, Any]) -> bool:
    if node.get("deleted") is True or node.get("review_result") is False:
        return True
    labels = node.get("labels", {})
    return any((_label_value(labels, name) or 0.0) > 0 for name in BLOCKING_LABELS)


def iter_french_leaves(
    node: dict[str, Any], messages: list[dict[str, str]] | None = None
) -> Iterator[list[dict[str, str]]]:
    """Walk a conversation tree, yielding each clean French assistant-ended branch."""
    if not isinstance(node, dict) or node.get("lang") != "fr" or _blocked(node):
        return

    role_map = {"prompter": "user", "assistant": "assistant"}
    role = role_map.get(str(node.get("role", "")).lower())
    content = node.get("text")
    if role is None or not isinstance(content, str) or not content.strip():
        return
    # Keep only the best-ranked answer when OASST has sibling alternatives.
    if role == "assistant" and node.get("rank") not in (None, 0):
        return

    path = list(messages or [])
    path.append({"role": role, "content": content.strip()})
    replies = node.get("replies") or []
    if not replies:
        if len(path) >= 2 and path[-1]["role"] == "assistant":
            yield [{"role": "system", "content": SYSTEM_MESSAGE}, *path]
        return

    for reply in replies:
        yield from iter_french_leaves(reply, path)


def tree_key(tree: dict[str, Any]) -> str:
    prompt = tree.get("prompt") or {}
    return str(tree.get("message_tree_id") or prompt.get("message_id") or "")


def validation_tree(tree_id: str, ratio: float) -> bool:
    digest = hashlib.sha256(tree_id.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return value < ratio


def canonical_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(output_dir: Path, validation_ratio: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / ARCHIVE_NAME
    download(DATASET_URL, archive)
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    trees_read = 0
    french_branches = 0

    with gzip.open(archive, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            tree = json.loads(line)
            trees_read += 1
            root = tree.get("prompt")
            if not isinstance(root, dict):
                continue
            key = tree_key(tree) or str(trees_read)
            is_validation = validation_tree(key, validation_ratio)
            for messages in iter_french_leaves(root):
                digest = canonical_hash(messages)
                if digest in seen:
                    continue
                seen.add(digest)
                row = {
                    "messages": messages,
                    "source": "OpenAssistant/oasst1",
                    "source_tree": key,
                }
                (validation_rows if is_validation else train_rows).append(row)
                french_branches += 1

    if not train_rows or not validation_rows:
        raise RuntimeError(
            "Le filtrage OASST1 n'a pas produit des exemples train et validation. "
            "Vérifie le format source et le filtre français avant tout entraînement."
        )

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    manifest = {
        "dataset": "OpenAssistant/oasst1",
        "revision": DATASET_REVISION,
        "archive": ARCHIVE_NAME,
        "archive_sha256": archive_sha256,
        "license": "Apache-2.0",
        "language_filter": "all messages in each branch must have lang=fr",
        "excluded_labels": sorted(BLOCKING_LABELS),
        "assistant_reply_policy": "rank 0 only; lower-ranked alternatives excluded",
        "trees_read": trees_read,
        "unique_french_conversations": french_branches,
        "train_conversations": len(train_rows),
        "validation_conversations": len(validation_rows),
        "validation_ratio_by_tree": validation_ratio,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.05 <= args.validation_ratio <= 0.25:
        parser.error("--validation-ratio doit être compris entre 0,05 et 0,25")
    result = prepare(args.output_dir, args.validation_ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
