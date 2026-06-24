# ── Acadza Question Uploading App — Production Dockerfile ──
# Python 3.11-slim with system deps for opencv, cairosvg, rdkit, faiss
FROM python:3.11-slim

# System dependencies:
# - libgl1 + libglib2.0: OpenCV headless
# - libcairo2 + libpango: cairosvg SVG->PNG
# - libffi-dev: cffi (pycairo build)
# - gcc g++: native extensions (faiss-cpu, numpy, etc.)
# - curl: health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    gcc \
    g++ \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/render/project/src

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install rdkit via pip (older version that has Python 3.11 wheels)
# Stage 5 SMILES rendering gracefully degrades if unavailable
RUN pip install --no-cache-dir rdkit-pypi==2023.3.1b1 || echo "rdkit install failed — SMILES rendering disabled"

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Render sets PORT env var (default 10000)
ENV PORT=10000

EXPOSE ${PORT}

# Startup: bootstrap data to persistent disk, then run uvicorn
CMD ["./start.sh"]
