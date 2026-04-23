# ── Stage 1: Export ECAPA-TDNN to ONNX ──────────────────────────────────
FROM python:3.11-slim AS exporter

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 120 \
    "torch==2.0.1" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir speechbrain==1.0.2

COPY scripts/export_onnx.py /export_onnx.py
RUN mkdir -p /model && python /export_onnx.py

# ── Stage 2: Runtime (lightweight, ONNX only) ──────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy exported ONNX model from builder
COPY --from=exporter /model/ecapa_tdnn.onnx /app/model/ecapa_tdnn.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Embeddings volume — persisted on the host
VOLUME ["/data/embeddings"]

CMD ["python", "-m", "src.main"]
