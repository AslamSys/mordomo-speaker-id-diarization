"""
Speaker diarization + identification engine.
Uses ECAPA-TDNN via ONNX Runtime for speaker embeddings (ARM64 compatible, lightweight).
Accumulates audio frames, extracts embeddings, identifies speakers, detects overlap.
"""
import logging
import time

import numpy as np
import onnxruntime as ort

from src.config import config
from src.embeddings import EmbeddingStore, cosine_similarity

logger = logging.getLogger("diarizer")


class SpeakerDiarizer:
    def __init__(self, store: EmbeddingStore):
        self.store = store
        self._session: ort.InferenceSession | None = None
        self._prev_speaker_id: str | None = None

    def load_model(self):
        model_path = config.onnx_model_path
        logger.info(f"Loading ONNX embedding model: {model_path}")
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX embedding model loaded")

    def extract_embedding(self, pcm_bytes: bytes) -> np.ndarray:
        """Extract speaker embedding from raw PCM int16 audio."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = samples.reshape(1, -1)  # (1, T)

        # Resample if needed (model expects 16kHz)
        if config.sample_rate != 16000:
            # Simple linear interpolation resampling
            ratio = 16000 / config.sample_rate
            n_out = int(samples.shape[0] * ratio)
            indices = np.linspace(0, samples.shape[0] - 1, n_out)
            resampled = np.interp(indices, np.arange(samples.shape[0]), samples)
            waveform = resampled.reshape(1, -1)

        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {input_names[0]: waveform.astype(np.float32)}
        if len(input_names) > 1:
            feed[input_names[1]] = np.array([1.0], dtype=np.float32)

        outputs = self._session.run(None, feed)
        return outputs[0].squeeze()

    def identify_segment(
        self, pcm_bytes: bytes, timestamp: float, conversation_id: str
    ) -> dict | None:
        """
        Identify the speaker of an audio segment.
        Returns a diarization result dict or None if segment too short.
        """
        min_samples = int(config.min_segment_ms * config.sample_rate / 1000) * 2  # int16 = 2 bytes
        if len(pcm_bytes) < min_samples:
            return None

        embedding = self.extract_embedding(pcm_bytes)
        speaker_id, confidence, recognized = self.store.identify(embedding)

        # Detect speaker change
        speaker_changed = (
            self._prev_speaker_id is not None
            and self._prev_speaker_id != speaker_id
        )
        self._prev_speaker_id = speaker_id

        duration_s = len(pcm_bytes) / 2 / config.sample_rate  # int16 = 2 bytes/sample

        return {
            "speaker_id": speaker_id,
            "recognized": recognized,
            "confidence": round(confidence, 3),
            "start_time": round(timestamp - duration_s, 3),
            "end_time": round(timestamp, 3),
            "overlap_detected": False,
            "speaker_changed": speaker_changed,
            "timestamp": timestamp,
            "conversation_id": conversation_id,
        }

    def detect_overlap(
        self, pcm_bytes: bytes
    ) -> tuple[bool, list[np.ndarray] | None]:
        """
        Detect if multiple speakers are present in the audio segment.
        Uses a simple approach: split segment in half, compare embeddings.
        If they differ significantly, overlap is likely.
        Returns (overlap_detected, [embedding_a, embedding_b] or None).
        """
        min_samples = int(config.min_overlap_duration_s * config.sample_rate) * 2
        if len(pcm_bytes) < min_samples * 2:
            return False, None

        mid = len(pcm_bytes) // 2
        # Align to int16 boundary
        mid = mid - (mid % 2)

        emb_a = self.extract_embedding(pcm_bytes[:mid])
        emb_b = self.extract_embedding(pcm_bytes[mid:])

        sim = cosine_similarity(emb_a, emb_b)

        # Low similarity between halves suggests different speakers (overlap)
        if sim < config.overlap_threshold:
            logger.info(f"Overlap detected: similarity={sim:.3f} < {config.overlap_threshold}")
            return True, [emb_a, emb_b]

        return False, None

    def reset(self):
        """Reset state for new conversation."""
        self._prev_speaker_id = None
