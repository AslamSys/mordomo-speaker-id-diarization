FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download SpeechBrain ECAPA-TDNN model at build time
RUN python -c "from speechbrain.inference.speaker import EncoderClassifier; \
    EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='/tmp/speechbrain_cache', run_opts={'device':'cpu'})"

COPY src/ src/

CMD ["python", "-m", "src.main"]
