#!/usr/bin/env python3
"""Refuse release unless both model tracks pass and a human approves the outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = ROOT / "generated-artifacts/neurolink/adapters/qwen3-1.7b-lora"
DEFAULT_EMBEDDING_MODEL = (
    ROOT / "generated-artifacts/neurolink/embedding-staging/multilingual-e5-small"
)
DEFAULT_RELEASE = ROOT / "generated-artifacts/neurolink/release"
DEFAULT_HUMAN_REVIEW = ROOT / "generated-artifacts/neurolink/reports/human_review.json"


def read_report(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Rapport manquant : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat-report",
        type=Path,
        default=ROOT / "generated-artifacts/neurolink/reports/chat.json",
    )
    parser.add_argument(
        "--embedding-report",
        type=Path,
        default=ROOT / "generated-artifacts/neurolink/reports/embeddings.json",
    )
    parser.add_argument("--human-review", type=Path, default=DEFAULT_HUMAN_REVIEW)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--embedding-model-dir", type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()

    chat = read_report(args.chat_report)
    embeddings = read_report(args.embedding_report)
    human = read_report(args.human_review)
    chat_cases = chat.get("cases", [])
    embedding_cases = embeddings.get("cases", [])
    failures = []
    if not chat.get("summary", {}).get("passed"):
        failures.append("tests de conversation échoués")
    if not embeddings.get("passed"):
        failures.append("tests d'embeddings échoués")
    if (
        len(chat_cases) < 20
        or len(embedding_cases) < 20
        or len({case.get("id") for case in chat_cases}) != len(chat_cases)
        or len({case.get("id") for case in embedding_cases}) != len(embedding_cases)
    ):
        failures.append("les deux benchmarks doivent contenir au moins 20 cas")
    if (
        human.get("chat_quality_approved") is not True
        or human.get("embedding_quality_approved") is not True
        or type(human.get("reviewed_chat_case_count")) is not int
        or human.get("reviewed_chat_case_count", 0) < len(chat_cases)
        or type(human.get("reviewed_embedding_case_count")) is not int
        or human.get("reviewed_embedding_case_count", 0) < len(embedding_cases)
    ):
        failures.append("validation humaine absente ou négative")
    if not args.adapter.is_dir():
        failures.append("adaptateur de conversation introuvable")
    if not args.embedding_model_dir.is_dir():
        failures.append("encodeur d'embeddings exporté introuvable")
    if failures:
        raise SystemExit("MODÈLE NON LIVRÉ : " + "; ".join(failures))

    args.release_dir.mkdir(parents=True, exist_ok=True)
    destination = args.release_dir / "neurolink-assistant"
    if destination.exists():
        raise SystemExit(
            f"Une livraison existe déjà à {destination}; archive-la manuellement "
            "avant de créer une nouvelle version."
        )
    staging = args.release_dir / ".neurolink-assistant-staging"
    if staging.exists():
        raise SystemExit(
            f"Un dossier de préparation existe déjà à {staging}; vérifie-le avant "
            "de poursuivre."
        )
    staging.mkdir()
    shutil.copytree(args.adapter, staging / "chat-adapter")
    shutil.copytree(args.embedding_model_dir, staging / "embedding-model")
    shutil.copy2(args.embedding_report, staging / "embedding-evaluation.json")
    shutil.copy2(args.chat_report, staging / "chat-evaluation.json")
    shutil.copy2(args.human_review, staging / "human-review.json")
    (staging / "RELEASE.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "chat_base_model": chat.get("base_model"),
                "chat_adapter": "chat-adapter",
                "embedding_model": embeddings.get("model"),
                "embedding_encoder": "embedding-model",
                "context_tokens": 10_000,
                "note": "Les poids ne sont copiés qu'après réussite de tous les contrôles.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    staging.rename(destination)
    print(f"MODÈLE VALIDÉ ET COPIÉ : {args.release_dir}")


if __name__ == "__main__":
    main()
