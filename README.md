# BankLaw AI

An Egyptian banking-law RAG (Retrieval-Augmented Generation) assistant.

Ingests 19 Central Bank of Egypt (CBE) legal PDFs, indexes them into a
vector database, and answers natural-language Arabic questions with
citation-backed answers.

## Architecture

```
PDF → text extraction → Arabic validation → OCR fallback → cleaning
   → structure/sections → metadata → documents → chunking → BGE-M3 embeddings
   → ChromaDB → semantic retrieval → BGE cross-encoder re-ranking
   → citation-aware context → prompt → Qwen3:8b (Ollama) → answer
```

## Entry points

| Command | Purpose |
|---------|---------|
| `streamlit run banklaw_ui.py` | Minimal Arabic web UI (question input, answer, citations, sources) |
| `python ask_banklaw.py "question"` | CLI: prints the answer and the sources list |
| `python -m banklaw_step1.index_corpus.py` | (Re)build the Chroma index from `data/` |
| `python -c "from banklaw_step1.query import ask; print(ask('...'))"` | Programmatic entry point |

## Data

- `data/` — 19 CBE PDFs (laws, customer protection, banking regulations,
  circulars) plus `corpus_manifest.json` categorizing each document.
- `data/chroma/` — persistent ChromaDB collection `banklaw_documents`
  (1013 chunks as of the last index build).

## Models

- Embeddings + re-ranking: `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`
  (1024-dim vectors).
- Answer generation: `qwen3:8b` via Ollama.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python -m banklaw_step1.index_corpus.py   # build the index once
```

Ollama must be running with `qwen3:8b` available (`ollama pull qwen3:8b`).

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

88 tests pass: pipeline unit tests, retrieval/reranking/evaluation tests
(including 10 golden queries), RAG quality/grounding tests, CLI UTF-8 printing,
UI error handling, and end-to-end integration tests.

## Verified end-to-end

Real Arabic queries — complaints, customer rights, credit conditions,
custody, dormant accounts, investment funds / money markets, bancassurance,
deposit banks, primary dealers, retail banking disclosure, and CBE Law
194/2020 — return non-empty, citation-grounded answers.