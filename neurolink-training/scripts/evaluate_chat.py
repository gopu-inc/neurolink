#!/usr/bin/env python3
"""Run a small deterministic conversation gate against a trained adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_ADAPTER = ROOT / "generated-artifacts/neurolink/adapters/qwen3-1.7b-lora"
DEFAULT_CASES = ROOT / "neurolink-training/evaluation/chat_cases.jsonl"
DEFAULT_REPORT = ROOT / "generated-artifacts/neurolink/reports/chat.json"


def has_ngram_loop(text: str, n: int = 6, repeats: int = 3) -> bool:
    words = re.findall(r"\b[\wÀ-ÿ]+\b", text.lower())
    if len(words) < n * repeats:
        return False
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(count >= repeats for count in counts.values())


def build_prompt(case: dict, tokenizer):
    messages = [dict(message) for message in case["messages"]]
    long_context = case.get("long_context")
    if not long_context:
        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ),
            None,
        )

    question = messages[-1]["content"]
    minimum = int(long_context["minimum_prompt_tokens"])
    maximum = int(long_context["maximum_prompt_tokens"])
    fact = long_context["fact"]
    filler = long_context["filler_sentence"]
    low, high = 0, 5_000
    best_prompt = None
    best_count = 0
    while low <= high:
        repetitions = (low + high) // 2
        messages[-1]["content"] = (
            f"{fact}\n" + (f"{filler}\n" * repetitions) + question
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        token_count = int(prompt.shape[-1])
        if token_count <= maximum:
            best_prompt = prompt
            best_count = token_count
            low = repetitions + 1
        else:
            high = repetitions - 1
    if best_prompt is None or best_count < minimum:
        raise SystemExit(
            f"Impossible de construire le cas de contexte long entre "
            f"{minimum} et {maximum} tokens; mesure obtenue : {best_count}."
        )
    return best_prompt, best_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()
    if not args.adapter.is_dir():
        raise SystemExit("Adaptateur introuvable : lance d'abord scripts/train_chat.py.")

    os.environ.setdefault(
        "HF_HOME", str(ROOT / "generated-artifacts/neurolink/huggingface-cache")
    )
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit(
            "Évaluation arrêtée : aucun GPU CUDA disponible pour charger le modèle."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if (args.adapter / "tokenizer_config.json").exists() else BASE_MODEL,
        revision=None if (args.adapter / "tokenizer_config.json").exists() else BASE_REVISION,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    records = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_categories = {
        "factual",
        "memory",
        "conversation",
        "instruction",
        "uncertainty",
        "context",
    }
    if len(records) < 20 or not required_categories.issubset(
        {case.get("category") for case in records}
    ):
        raise SystemExit(
            "La suite de conversation doit contenir au moins 20 cas et couvrir "
            "les catégories factual, memory, conversation, instruction et uncertainty."
        )
    results = []
    for case in records:
        prompt, long_prompt_tokens = build_prompt(case, tokenizer)
        prompt = prompt.to(model.device)
        torch.manual_seed(42)
        with torch.inference_mode():
            generated = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0, prompt.shape[-1] :], skip_special_tokens=True
        ).strip()
        normalized = answer.casefold()
        contains = [term.casefold() in normalized for term in case.get("must_contain", [])]
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "answer": answer,
                "nonempty": bool(answer),
                "required_terms_pass": all(contains) if contains else None,
                "repetition_pass": not has_ngram_loop(answer),
                "context_length_pass": (
                    long_prompt_tokens is None
                    or 7_000 <= long_prompt_tokens <= 9_700
                ),
                "prompt_tokens": long_prompt_tokens,
            }
        )

    def category_pass(category: str) -> bool:
        group = [item for item in results if item["category"] == category]
        return bool(group) and all(
            item["nonempty"]
            and item["repetition_pass"]
            and item["context_length_pass"]
            and item["required_terms_pass"] is not False
            for item in group
        )

    factual = category_pass("factual")
    memory = category_pass("memory")
    category_results = {
        category: category_pass(category) for category in sorted(required_categories)
    }
    all_clean = all(
        item["nonempty"]
        and item["repetition_pass"]
        and item["context_length_pass"]
        for item in results
    )
    report = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter": str(args.adapter),
        "cases": results,
        "summary": {
            "case_count": len(results),
            "category_results": category_results,
            "factual_pass": factual,
            "memory_pass": memory,
            "all_nonempty_and_nonrepetitive": all_clean,
            "passed": all(category_results.values()) and all_clean,
            "human_review_required": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
