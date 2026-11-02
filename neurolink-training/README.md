# NeuroLink assistant training

This pipeline adapts a multilingual pretrained assistant and evaluates a separate
semantic embedding model. **No trained adapter or released model is included yet.**
The original NeuroLink source, corpus, and manifests are preserved in `../neurolink/`;
the uploaded archive remains available in the conversation files.

## Model choices

- Chat base: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), pinned to
  revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its model card lists Apache-2.0,
  1.7 billion parameters, and a 32,768-token context. The training default is 10,000
  tokens; no positional-embedding resize is needed.
- Embeddings: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
  pinned to `614241f622f53c4eeff9890bdc4f31cfecc418b3`, licensed MIT. It is a distinct,
  pretrained sentence encoder; the chat model's hidden states are not presented as a
  semantic embedding system.
- Chat data: French, human-annotated prompt-completion pairs from CohereLabs/Aya
  (Apache-2.0), plus top-ranked French conversation branches from reviewed OASST1 trees
  (Apache-2.0). Aya has 202,362 train pairs across 65 languages; the exact French yield
  after filtering is recorded in the generated manifest. OASST1's full release documents
  4,251 French messages; its useful branch count is recorded after filtering.

The selected chat model has **1.7 billion** parameters, not 2 million. That is the
intentional tradeoff for starting from a model that already understands language and
can converse. A 2-million-parameter model trained from zero is suitable for a narrow
demo, not a comparable general assistant.

## Dataset preparation

From the workspace root:

```sh
python neurolink-training/scripts/prepare_oasst1.py
python neurolink-training/scripts/prepare_aya.py
python neurolink-training/scripts/combine_sft.py
```

The scripts download pinned dataset revisions, keep French-only examples, filter review
flags and lower-ranked OASST alternatives, deduplicate prompts across sources, and keep
each source's held-out prompts out of training. Aya's separate demographics config is not
downloaded. Raw and processed files are kept under `generated-artifacts/neurolink/` and
excluded from Git and deployments.

These corpora are useful fine-tuning data, not proof of broad assistant quality. The
user's existing SFT corpus remains preserved separately; it should only be mixed in
after its examples have been checked for factual quality and rights.

## Fine-tuning

The recipe uses QLoRA for two epochs, trains loss only on assistant turns, keeps the
10,000-token sequence length, and saves adapters outside source control:

```sh
python neurolink-training/scripts/train_chat.py
```

This needs CUDA, a compatible bitsandbytes/PyTorch stack, and sufficient GPU memory.
The current runtime has no CUDA device, so the script stops before downloading model
weights or starting an un-runnable training job. It will not silently fall back to a
CPU run and claim success.

Install the optional dependencies in `requirements-training.txt` in a compatible GPU
environment. The installed `transformers`, `trl`, `peft`, `accelerate`, and
`bitsandbytes` versions must be kept together; TRL's `assistant_only_loss` support is
required so user prompts are not treated as target answers.

## Evaluation and release gate

```sh
python neurolink-training/scripts/evaluate_chat.py
python neurolink-training/scripts/evaluate_embeddings.py
python neurolink-training/scripts/release_gate.py \
  --chat-report generated-artifacts/neurolink/reports/chat.json \
  --embedding-report generated-artifacts/neurolink/reports/embeddings.json
```

The default human-review file must explicitly contain both approvals, for example:

```json
{
  "chat_quality_approved": true,
  "embedding_quality_approved": true
}
```

The 20-case chat suite checks French and English facts, multi-turn recall, task following,
uncertainty, repetition, and a prompt built to 7,000–9,700 tokens. The 20-case embedding
suite varies the correct result's position to prevent a fixed-position shortcut and checks
normalized vectors and French semantic retrieval. A passing embedding evaluation saves
the separate encoder under `generated-artifacts/neurolink/embedding-staging/`.
`release_gate.py` refuses to copy either model unless both measured thresholds pass and
a human has reviewed every case.

The human-review JSON must explicitly contain both approvals and the number of cases
reviewed, for example:

```json
{
  "chat_quality_approved": true,
  "embedding_quality_approved": true,
  "reviewed_chat_case_count": 20,
  "reviewed_embedding_case_count": 20
}
```

Only a passing chat adapter and a passing embedding encoder may be copied together into
`generated-artifacts/neurolink/release/neurolink-assistant/`. Passing these smoke suites
is a prerequisite, not proof of universal assistant quality; broader blind evaluation
should follow before public release.
