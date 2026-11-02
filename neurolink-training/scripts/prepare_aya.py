#!/usr/bin/env python3
"""Download the pinned Aya dataset and prepare its French SFT examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
ARCHIVE_NAME = "train-00000-of-00001.parquet"
DATASET_URL = (
    "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/"
    f"{DATASET_REVISION}/data/{ARCHIVE_NAME}?download=true"
)
DEFAULT_OUTPUT = ROOT / "generated-artifacts/neurolink/datasets/aya-fr"
DEFAULT_RAW = ROOT / "generated-artifacts/neurolink/datasets/aya-raw"
SYSTEM_MESSAGE = (
    "Tu es NeuroLink, un assistant utile, clair et respectueux. "
    "Réponds dans la langue de l'utilisateur, en français lorsqu'il s'exprime en français. "
    "Si tu ne sais pas, dis-le."
)
BLOCKED_PHRASES = (
    "as an ai language model",
    "en tant que modèle de langage",
    "i cannot assist with that request",
)


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroLink-training/1.0 (dataset preparation)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def stable_validation(prompt: str, ratio: float) -> bool:
    digest = hashlib.sha256(prompt.strip().casefold().encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12) < ratio


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = row.get("inputs")
    answer = row.get("targets")
    if row.get("language_code") not in {"fra", "fr"}:
        return None
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    prompt, answer = prompt.strip(), answer.strip()
    if len(prompt) < 5 or len(answer) < 2:
        return None
    if len(prompt) + len(answer) > 80_000:
        return None
    lowered = answer.casefold()
    if any(phrase in lowered for phrase in BLOCKED_PHRASES):
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "source": "CohereLabs/aya_dataset",
    }


def prepare(
    raw_dir: Path, output_dir: Path, validation_ratio: float
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / ARCHIVE_NAME
    download(DATASET_URL, parquet_path)
    checksum = hashlib.sha256()
    with parquet_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows_read = french_rows = duplicates = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=4096, columns=["inputs", "targets", "language_code", "annotation_type"]
    ):
        for raw_row in batch.to_pylist():
            rows_read += 1
            if raw_row.get("language_code") in {"fra", "fr"}:
                french_rows += 1
            row = convert_row(raw_row)
            if row is None:
                continue
            key = json.dumps(row["messages"][1:], ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            row["annotation_type"] = raw_row.get("annotation_type")
            target = (
                validation
                if stable_validation(row["messages"][1]["content"], validation_ratio)
                else train
            )
            target.append(row)

    if not train or not validation:
        raise RuntimeError(
            "Le filtrage Aya n'a pas produit des exemples train et validation. "
            "Vérifie les valeurs de language_code et les données source."
        )
    for name, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (output_dir / name).open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "CohereLabs/aya_dataset",
        "revision": DATASET_REVISION,
        "file": ARCHIVE_NAME,
        "file_sha256": checksum.hexdigest(),
        "license": "Apache-2.0",
        "language_filter": "language_code in {'fra', 'fr'}",
        "rows_read": rows_read,
        "french_rows_before_quality_filter": french_rows,
        "duplicates_removed": duplicates,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_ratio_by_prompt": validation_ratio,
        "demographics_config_used": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.05 <= args.validation_ratio <= 0.25:
        parser.error("--validation-ratio doit être compris entre 0,05 et 0,25")
    print(json.dumps(prepare(args.raw_dir, args.output_dir, args.validation_ratio),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
