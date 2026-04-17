import os


class Config:
    # NATS
    nats_url: str = os.getenv("NATS_URL", "nats://nats:4222")

    # ZeroMQ (audio from VAD)
    zmq_vad_url: str = os.getenv("ZMQ_VAD_URL", "tcp://audio-capture-vad:5555")
    zmq_topic: str = os.getenv("ZMQ_TOPIC", "audio.raw")

    # Audio
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    frame_duration_ms: int = int(os.getenv("FRAME_DURATION_MS", "30"))

    # Speaker recognition
    embeddings_path: str = os.getenv("EMBEDDINGS_PATH", "/data/embeddings")
    recognition_threshold: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.70"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
    )

    # ONNX model path (exported ECAPA-TDNN)
    onnx_model_path: str = os.getenv(
        "ONNX_MODEL_PATH", "/app/model/ecapa_tdnn.onnx"
    )

    # Diarization
    max_speakers: int = int(os.getenv("MAX_SPEAKERS", "3"))
    min_segment_ms: int = int(os.getenv("MIN_SEGMENT_MS", "1000"))
    overlap_threshold: float = float(os.getenv("OVERLAP_THRESHOLD", "0.5"))
    min_overlap_duration_s: float = float(os.getenv("MIN_OVERLAP_DURATION_S", "0.5"))

    # Buffer: how many seconds of audio to accumulate before running identification
    segment_duration_s: float = float(os.getenv("SEGMENT_DURATION_S", "2.0"))

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def frames_per_segment(self) -> int:
        return int(self.segment_duration_s * 1000 / self.frame_duration_ms)


config = Config()
