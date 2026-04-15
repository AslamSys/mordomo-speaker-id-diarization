"""
Enrolled speaker embeddings — loads .npy files from shared volume,
compares live embeddings via cosine similarity, hot-reloads on changes.
"""
import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("embeddings")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingStore:
    """
    Read-only store for enrolled speaker embeddings.
    Speaker Verification writes them; we only read.
    """

    def __init__(self, embeddings_path: str, threshold: float = 0.70):
        self.path = Path(embeddings_path)
        self.threshold = threshold
        self._enrolled: dict[str, np.ndarray] = {}
        self._last_scan: float = 0.0
        self._scan_interval: float = 5.0  # seconds
        self._load_all()

    def _load_all(self):
        if not self.path.exists():
            logger.warning(f"Embeddings path does not exist: {self.path}")
            return
        count = 0
        for f in self.path.glob("*.npy"):
            user_id = f.stem  # user_1.npy -> user_1
            try:
                self._enrolled[user_id] = np.load(f)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        self._last_scan = time.monotonic()
        logger.info(f"Loaded {count} enrolled embeddings from {self.path}")

    def _maybe_reload(self):
        now = time.monotonic()
        if now - self._last_scan > self._scan_interval:
            self._load_all()

    def identify(self, embedding: np.ndarray) -> tuple[str, float, bool]:
        """
        Compare embedding against all enrolled speakers.
        Returns (speaker_id, confidence, recognized).
        """
        self._maybe_reload()

        if not self._enrolled:
            uid = "unknown_" + hashlib.md5(embedding.tobytes()[:64]).hexdigest()[:8]
            return uid, 0.0, False

        best_id = ""
        best_sim = -1.0
        for user_id, enrolled in self._enrolled.items():
            sim = cosine_similarity(embedding, enrolled)
            if sim > best_sim:
                best_sim = sim
                best_id = user_id

        if best_sim >= self.threshold:
            return best_id, best_sim, True

        uid = "unknown_" + hashlib.md5(embedding.tobytes()[:64]).hexdigest()[:8]
        return uid, best_sim, False

    @property
    def enrolled_count(self) -> int:
        return len(self._enrolled)
