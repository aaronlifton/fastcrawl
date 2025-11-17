#!/bin/sh

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 1
fi

docker compose up -d
