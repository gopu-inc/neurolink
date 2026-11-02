#!/usr/bin/env python3
"""Measure French semantic retrieval with the dedicated multilingual E5 encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_CASES = ROOT / "neurolink-training/evaluation/embedding_cases.jsonl"
DEFAULT_REPORT = ROOT / "generated-artifacts/neurolink/reports/embeddings.json"
DEFAULT_MODEL_OUTPUT = (
    ROOT / "generated-artifacts/neurolink/embedding-staging/multilingual-e5-small"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--minimum-recall-at-1", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.minimum_recall_at_1 <= 1:
        parser.error("--minimum-recall-at-1 doit être compris entre 0 et 1")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) < 20 or any(
        len(case.get("candidates", [])) < 3
        or not 0 <= case.get("positive_index", -1) < len(case.get("candidates", []))
        for case in cases
    ):
        raise SystemExit(
            "La suite d'embeddings doit contenir au moins 20 cas, avec trois "
            "passages candidats ou plus par cas."
        )
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        cache_folder=str(ROOT / "generated-artifacts/neurolink/huggingface-cache"),
    )
    queries = ["query: " + item["query"] for item in cases]
    passages = [
        "passage: " + candidate
        for item in cases
        for candidate in item["candidates"]
    ]
    query_vectors = model.encode(queries, normalize_embeddings=True)
    passage_vectors = model.encode(passages, normalize_embeddings=True)

    results = []
    offset = 0
    for index, case in enumerate(cases):
        count = len(case["candidates"])
        scores = np.asarray(passage_vectors[offset : offset + count]) @ np.asarray(
            query_vectors[index]
        )
        predicted = int(np.argmax(scores))
        results.append(
            {
                "id": case["id"],
                "expected_index": case["positive_index"],
                "predicted_index": predicted,
                "passed": predicted == case["positive_index"],
                "scores": [float(score) for score in scores],
            }
        )
        offset += count

    recall_at_1 = sum(item["passed"] for item in results) / max(1, len(results))
    finite = bool(np.isfinite(passage_vectors).all() and np.isfinite(query_vectors).all())
    unit_norms = bool(
        np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-3)
        and np.allclose(np.linalg.norm(passage_vectors, axis=1), 1.0, atol=1e-3)
    )
    report = {
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "license": "MIT",
        "case_count": len(results),
        "recall_at_1": recall_at_1,
        "minimum_recall_at_1": args.minimum_recall_at_1,
        "finite_vectors": finite,
        "unit_normalized": unit_norms,
        "passed": (
            finite and unit_norms and recall_at_1 >= args.minimum_recall_at_1
        ),
        "cases": results,
        "human_review_required": True,
    }
    if report["passed"]:
        if args.model_output.exists():
            raise SystemExit(
                f"Le modèle d'embeddings existe déjà à {args.model_output}; "
                "choisis un nouveau chemin pour éviter de l'écraser."
            )
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_output))
        report["saved_model_path"] = str(args.model_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "recall_at_1": recall_at_1,
                "minimum_recall_at_1": args.minimum_recall_at_1,
                "finite_vectors": finite,
                "unit_normalized": unit_norms,
                "passed": report["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
