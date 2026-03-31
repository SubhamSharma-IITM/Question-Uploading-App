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
# Copy large data files from Docker image to persistent disk
# Subsequent deploys skip this (files already exist)

# Create directories
mkdir -p "$PERSISTENT/sessions"
mkdir -p "$PERSISTENT/db_index_cache/db_reference"

# Question database (512MB) — Stage 8 cross-check + Stage 10 absorption
if [ ! -f "$PERSISTENT/allque030326.json" ]; then
    echo "[BOOT] Copying question database to persistent disk..."
    cp "$SRC_DATA/allque030326.json" "$PERSISTENT/allque030326.json"
    echo "[BOOT] Done ($(du -sh "$PERSISTENT/allque030326.json" | cut -f1))"
else
    echo "[BOOT] Question database already on persistent disk"
fi

# FAISS index (965MB) — Stage 8 duplicate detection + Stage 10 update
if [ ! -f "$PERSISTENT/db_index_cache/db_reference/faiss.index" ]; then
    echo "[BOOT] Copying FAISS index to persistent disk..."
    cp "$SRC_DATA/db_index_cache/db_reference/faiss.index" "$PERSISTENT/db_index_cache/db_reference/faiss.index"
    echo "[BOOT] Done ($(du -sh "$PERSISTENT/db_index_cache/db_reference/faiss.index" | cut -f1))"
else
    echo "[BOOT] FAISS index already on persistent disk"
fi

# FAISS metadata (8MB) — OID map for duplicate preview
if [ ! -f "$PERSISTENT/db_index_cache/db_reference/metadata.json" ]; then
    echo "[BOOT] Copying FAISS metadata to persistent disk..."
    cp "$SRC_DATA/db_index_cache/db_reference/metadata.json" "$PERSISTENT/db_index_cache/db_reference/metadata.json"
    echo "[BOOT] Done ($(du -sh "$PERSISTENT/db_index_cache/db_reference/metadata.json" | cut -f1))"
else
    echo "[BOOT] FAISS metadata already on persistent disk"
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
