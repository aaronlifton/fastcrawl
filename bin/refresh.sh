#!/usr/bin/env bash
set -euo pipefail
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL must be set" >&2
  exit 1
fi

INPUT=${FASTCRAWL_REFRESH_INPUT:-data/wiki.jsonl}
MANIFEST=${FASTCRAWL_REFRESH_MANIFEST:-data/wiki_manifest.jsonl}
EMBEDDINGS=${FASTCRAWL_REFRESH_EMBEDDINGS:-data/wiki_embeddings.jsonl}
SCHEMA=${FASTCRAWL_REFRESH_SCHEMA:-public}
TABLE=${FASTCRAWL_REFRESH_TABLE:-wiki_chunks}
BATCH_SIZE=${FASTCRAWL_REFRESH_BATCH:-64}

# echo "== Freshness planner =="
# cargo run --bin freshness -- \
#   --current-manifest "$MANIFEST" \
#   --plan-output data/refresh_plan.jsonl \
#   --ledger-output data/embedding_ledger.jsonl || true
#
# echo "== Embedding changed chunks =="
# cargo run --bin embedder -- \
#   --provider openai \
#   --input "$INPUT" \
#   --manifest "$MANIFEST" \
#   --output "$EMBEDDINGS" \
#   --batch-size "$BATCH_SIZE" \
#   --only-changed

echo "== Loading vectors into Postgres =="
cargo run --bin pgvector_store -- \
  --input "$EMBEDDINGS" \
  --schema "$SCHEMA" \
  --table "$TABLE" \
  --batch-size 512 \
  --upsert \
  --database-url "$DATABASE_URL"

echo "== Ensuring FTS indexes =="
cargo run --bin fts_indexer -- \
  --database-url "$DATABASE_URL" \
  --schema "$SCHEMA" \
  --table "$TABLE"
