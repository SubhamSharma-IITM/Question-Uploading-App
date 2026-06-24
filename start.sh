#!/bin/bash
# ── Acadza Question Uploading App — Startup Bootstrap ──
# Copies Git LFS data files to persistent disk on first run,
# then starts the uvicorn server.

set -e

# Source and destination paths
SRC_DATA="/opt/render/project/src/data"
PERSISTENT="/data"

echo "=============================="
echo "  Acadza Startup Bootstrap"
echo "=============================="

# ── 1. Decode Vertex AI credentials from base64 env var ──
if [ -n "$VERTEX_AI_CREDENTIALS_B64" ]; then
    echo "[BOOT] Decoding Vertex AI service account..."
    echo "$VERTEX_AI_CREDENTIALS_B64" | base64 -d > /tmp/vertex_sa.json
    export VERTEX_AI_CREDENTIALS="/tmp/vertex_sa.json"
    export GOOGLE_APPLICATION_CREDENTIALS="/tmp/vertex_sa.json"
    echo "[BOOT] Vertex AI credentials written to /tmp/vertex_sa.json"
fi

# ── 2. Bootstrap persistent disk (first deploy only) ──
# SAFETY: NEVER overwrite live data files. Stage 10 absorptions update these
# continuously. The Git LFS copies are frozen snapshots from initial deploy.
# Only copy if the persistent disk file is completely MISSING (fresh disk).

# Create directories
mkdir -p "$PERSISTENT/sessions"
mkdir -p "$PERSISTENT/db_index_cache/db_reference"

# Question database — LIVE, updated by Stage 10 absorption
if [ ! -f "$PERSISTENT/allque030326.json" ]; then
    if [ -f "$SRC_DATA/allque030326.json" ] && [ "$(wc -c < "$SRC_DATA/allque030326.json")" -gt 1000 ]; then
        echo "[BOOT] First deploy: copying question database to persistent disk..."
        cp "$SRC_DATA/allque030326.json" "$PERSISTENT/allque030326.json"
        echo "[BOOT] Done ($(du -sh "$PERSISTENT/allque030326.json" | cut -f1))"
    else
        echo "[BOOT] WARNING: Question database missing from persistent disk AND source is missing/stub (LFS not fetched)."
        echo "[BOOT] WARNING: You must SCP the database manually: scp allque030326.json render:/data/"
    fi
else
    echo "[BOOT] Question database on persistent disk ($(du -sh "$PERSISTENT/allque030326.json" | cut -f1)) -- NOT overwriting"
fi

# FAISS index — LIVE, updated by Stage 10 absorption
if [ ! -f "$PERSISTENT/db_index_cache/db_reference/faiss.index" ]; then
    if [ -f "$SRC_DATA/db_index_cache/db_reference/faiss.index" ] && [ "$(wc -c < "$SRC_DATA/db_index_cache/db_reference/faiss.index")" -gt 1000 ]; then
        echo "[BOOT] First deploy: copying FAISS index to persistent disk..."
        cp "$SRC_DATA/db_index_cache/db_reference/faiss.index" "$PERSISTENT/db_index_cache/db_reference/faiss.index"
        echo "[BOOT] Done ($(du -sh "$PERSISTENT/db_index_cache/db_reference/faiss.index" | cut -f1))"
    else
        echo "[BOOT] WARNING: FAISS index missing from persistent disk AND source is missing/stub."
        echo "[BOOT] WARNING: You must SCP the index manually."
    fi
else
    echo "[BOOT] FAISS index on persistent disk ($(du -sh "$PERSISTENT/db_index_cache/db_reference/faiss.index" | cut -f1)) -- NOT overwriting"
fi

# FAISS metadata — LIVE, updated by Stage 10 absorption
if [ ! -f "$PERSISTENT/db_index_cache/db_reference/metadata.json" ]; then
    if [ -f "$SRC_DATA/db_index_cache/db_reference/metadata.json" ] && [ "$(wc -c < "$SRC_DATA/db_index_cache/db_reference/metadata.json")" -gt 1000 ]; then
        echo "[BOOT] First deploy: copying FAISS metadata to persistent disk..."
        cp "$SRC_DATA/db_index_cache/db_reference/metadata.json" "$PERSISTENT/db_index_cache/db_reference/metadata.json"
        echo "[BOOT] Done ($(du -sh "$PERSISTENT/db_index_cache/db_reference/metadata.json" | cut -f1))"
    else
        echo "[BOOT] WARNING: FAISS metadata missing from persistent disk AND source is missing/stub."
        echo "[BOOT] WARNING: You must SCP the metadata manually."
    fi
else
    echo "[BOOT] FAISS metadata on persistent disk ($(du -sh "$PERSISTENT/db_index_cache/db_reference/metadata.json" | cut -f1)) -- NOT overwriting"
fi

echo "[BOOT] Persistent disk contents:"
du -sh "$PERSISTENT"/* 2>/dev/null || echo "  (empty)"
echo "------------------------------"

# ── 2b. Verify env vars point to persistent disk ──
echo "[BOOT] DATA_DIR=$DATA_DIR"
echo "[BOOT] SESSIONS_DIR=$SESSIONS_DIR"
echo "[BOOT] DB_PATH=$DB_PATH"
echo "[BOOT] DB_INDEX_DIR=$DB_INDEX_DIR"

# Quick Python check: where will app.db actually be created?
python3 -c "
from app.config import settings
print(f'[BOOT] settings.data_dir     = {settings.data_dir}')
print(f'[BOOT] settings.sessions_dir = {settings.sessions_dir}')
print(f'[BOOT] settings.db_path      = {settings.db_path}')
print(f'[BOOT] app.db will be at     = {settings.data_dir}/app.db')
" 2>&1 || echo "[BOOT] WARNING: Could not verify settings"

# ── 3. Clear Python bytecode cache ──
find /opt/render/project/src/app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[BOOT] Cleared __pycache__"

# ── 4. Start uvicorn ──
echo "[BOOT] Starting uvicorn on port $PORT..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-10000}" \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level info
