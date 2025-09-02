#!/usr/bin/env bash
set -euo pipefail

echo "[build] initializing submodules (if any)"
git submodule update --init --recursive || true

echo "[build] installing Python dependencies"
if [ -f requirements.txt ]; then
  # prefer python3 so the build environment is explicit
  python3 -m pip install -r requirements.txt
fi

echo "[build] fetching vectorstore files into ${VECTORSTORE_PATH:-./app/vectorstore}"
python3 -m scripts.fetch_vectorstore --repo-id SamPease/TransAdviceAgent || true

echo "[build] build script finished"
