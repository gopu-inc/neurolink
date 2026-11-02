#!/usr/bin/env python3
"""Combine reviewed French OASST1 and Aya examples without crossing splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OASST = ROOT / "generated-artifacts/neurolink/datasets/oasst1-fr"
DEFAULT_AYA = ROOT / "generated-artifacts/neurolink/datasets/aya-fr"
DEFAULT_OUTPUT = ROOT / "generated-artifacts/neurolink/datasets/combined-fr"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Fichier de données absent : {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fingerprint(row: dict[str, Any]) -> str:
    user_turns = [
        {"role": message.get("role"), "content": message.get("content", "").strip().casefold()}
        for message in row["messages"]
        if message.get("role") == "user"
    ]
    encoded = json.dumps(
        user_turns, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combine(oasst_dir: Path, aya_dir: Path, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fingerprints_seen: set[str] = set()
    duplicates_removed = 0
    source_counts: dict[str, dict[str, int]] = {"train": {}, "validation": {}}

    for split in ("train", "validation"):
        merged: list[dict[str, Any]] = []
        for directory in (oasst_dir, aya_dir):
            for row in load_rows(directory / f"{split}.jsonl"):
                digest = fingerprint(row)
                if digest in fingerprints_seen:
                    duplicates_removed += 1
                    continue
                fingerprints_seen.add(digest)
                source = str(row.get("source", "unknown"))
                source_counts[split][source] = source_counts[split].get(source, 0) + 1
                merged.append({"messages": row["messages"]})
        if not merged:
            raise RuntimeError(f"Le split {split} est vide après fusion.")
        result[split] = merged

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in result.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "sources": ["OpenAssistant/oasst1", "CohereLabs/aya_dataset"],
        "train_examples": len(result["train"]),
        "validation_examples": len(result["validation"]),
        "source_counts": source_counts,
        "duplicates_removed": duplicates_removed,
        "dedupe": "SHA-256 of normalized user turns across both splits and sources",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasst-dir", type=Path, default=DEFAULT_OASST)
    parser.add_argument("--aya-dir", type=Path, default=DEFAULT_AYA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            combine(args.oasst_dir, args.aya_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
