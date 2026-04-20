"""
Main entry point — state machine for speaker identification/diarization.

States:
  IDLE        → wake_word.detected    → BUFFERING
  BUFFERING   → speaker.verified      → ANALYZING
  BUFFERING   → speaker.rejected      → IDLE (discard buffer)
  ANALYZING   → conversation.ended    → IDLE (reset)

Audio arrives via ZMQ SUB from audio-capture-vad.
Results published to NATS: speech.diarized.{speaker_id}
"""
import asyncio
import base64
import enum
import json
import logging
import threading
import time
import uuid

import nats
import zmq

from src.config import config
from src.diarizer import SpeakerDiarizer
from src.embeddings import EmbeddingStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("speaker-id-diarization")


class State(enum.Enum):
    IDLE = "idle"
    BUFFERING = "buffering"
    ANALYZING = "analyzing"


class DiarizationService:
    def __init__(self):
        self.state = State.IDLE
        self.conversation_id: str = ""
        self._buffer: bytearray = bytearray()
        self._frame_count: int = 0
        self._nc: nats.NATS | None = None
        self._lock = threading.Lock()

        # Components
        self.store = EmbeddingStore(config.embeddings_path, config.recognition_threshold)
        self.diarizer = SpeakerDiarizer(self.store)

    # ── NATS event handlers ────────────────────────────────────────────

    async def _on_wake_word(self, msg):
        data = json.loads(msg.data)
        with self._lock:
            if self.state != State.IDLE:
                logger.debug("Ignoring wake_word — not IDLE")
                return
            self.state = State.BUFFERING
            self.conversation_id = data.get("session_id", str(uuid.uuid4()))
            self._buffer.clear()
            self._frame_count = 0
        logger.info(f"IDLE → BUFFERING (conversation={self.conversation_id})")

    async def _on_speaker_verified(self, msg):
        with self._lock:
            if self.state != State.BUFFERING:
                return
            self.state = State.ANALYZING
        logger.info("BUFFERING → ANALYZING (gate open)")

        # Process any buffered audio immediately
        await self._process_buffer()

    async def _on_speaker_rejected(self, msg):
        with self._lock:
            if self.state != State.BUFFERING:
                return
            buf_size = len(self._buffer)
            self.state = State.IDLE
            self._buffer.clear()
            self._frame_count = 0
            self.diarizer.reset()
        logger.info(f"BUFFERING → IDLE (rejected, discarded {buf_size} bytes)")

    async def _on_conversation_ended(self, msg):
        with self._lock:
            prev = self.state
            self.state = State.IDLE
            self._buffer.clear()
            self._frame_count = 0
            self.diarizer.reset()
        logger.info(f"{prev.value} → IDLE (conversation ended)")

    # ── Audio processing ───────────────────────────────────────────────

    def _ingest_frame(self, pcm_bytes: bytes):
        """Called from ZMQ thread — thread-safe buffer append."""
        with self._lock:
            if self.state == State.IDLE:
                return
            self._buffer.extend(pcm_bytes)
            self._frame_count += 1

    async def _process_buffer(self):
        """Process accumulated audio buffer."""
        with self._lock:
            if self.state != State.ANALYZING or len(self._buffer) == 0:
                return
            audio = bytes(self._buffer)
            self._buffer.clear()
            self._frame_count = 0

        now = time.time()

        # Check for overlap
        overlap, _ = self.diarizer.detect_overlap(audio)
        if overlap:
            await self._publish_overlap(audio, now)

        # Identify speaker
        result = self.diarizer.identify_segment(
            audio, now, self.conversation_id
        )
        if result:
            result["overlap_detected"] = overlap
            subject = f"speech.diarized.{result['speaker_id']}"
            await self._nc.publish(subject, json.dumps(result).encode())
            logger.info(
                f"Published {subject}: "
                f"recognized={result['recognized']}, "
                f"confidence={result['confidence']}, "
                f"changed={result['speaker_changed']}"
            )

    async def _publish_overlap(self, audio: bytes, timestamp: float):
        """Trigger source-separation when overlap detected."""
        payload = {
            "audio": base64.b64encode(audio).decode(),
            "sample_rate": config.sample_rate,
            "timestamp": timestamp,
            "conversation_id": self.conversation_id,
        }
        await self._nc.publish("audio.overlap_detected", json.dumps(payload).encode())
        logger.info("Published audio.overlap_detected → source-separation")

    # ── ZMQ audio receiver (blocking, runs in thread) ──────────────────

    def _zmq_loop(self, loop: asyncio.AbstractEventLoop):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(config.zmq_vad_url)
        sub.setsockopt_string(zmq.SUBSCRIBE, config.zmq_topic)
        logger.info(f"ZMQ SUB connected: {config.zmq_vad_url} topic={config.zmq_topic}")

        while True:
            try:
                topic, pcm = sub.recv_multipart()
                self._ingest_frame(pcm)

                # When enough frames accumulated and in ANALYZING, schedule processing
                with self._lock:
                    should_process = (
                        self.state == State.ANALYZING
                        and self._frame_count >= config.frames_per_segment
                    )

                if should_process:
                    asyncio.run_coroutine_threadsafe(self._process_buffer(), loop)

            except zmq.ZMQError as e:
                logger.error(f"ZMQ error: {e}")
                time.sleep(1)

    # ── NATS heartbeat ─────────────────────────────────────────────────

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(30)
            payload = {
                "service": "speaker-id-diarization",
                "state": self.state.value,
                "enrolled_speakers": self.store.enrolled_count,
                "conversation_id": self.conversation_id or None,
                "timestamp": time.time(),
            }
            await self._nc.publish(
                "speaker.diarization.status", json.dumps(payload).encode()
            )

    # ── Main ───────────────────────────────────────────────────────────

    async def run(self):
        # Load ML model
        self.diarizer.load_model()

        # Connect NATS
        async def _error_cb(e): logger.error(f"NATS error: {e}")
        async def _reconnected_cb(): logger.warning("NATS reconnected")
        async def _disconnected_cb(): logger.warning("NATS disconnected")

        self._nc = await nats.connect(
            config.nats_url,
            error_cb=_error_cb,
            reconnected_cb=_reconnected_cb,
            disconnected_cb=_disconnected_cb,
        )
        logger.info(f"Connected to NATS: {config.nats_url}")

        # Subscribe to events
        await self._nc.subscribe("wake_word.detected", cb=self._on_wake_word)
        await self._nc.subscribe("speaker.verified", cb=self._on_speaker_verified)
        await self._nc.subscribe("speaker.rejected", cb=self._on_speaker_rejected)
        await self._nc.subscribe("conversation.ended", cb=self._on_conversation_ended)
        logger.info("Subscribed to NATS events")

        # Start heartbeat
        asyncio.create_task(self._heartbeat())

        # Start ZMQ audio receiver in background thread
        loop = asyncio.get_event_loop()
        zmq_thread = threading.Thread(
            target=self._zmq_loop, args=(loop,), daemon=True
        )
        zmq_thread.start()
        logger.info("ZMQ audio receiver started")

        # Also process buffer periodically while in ANALYZING
        # (in case frames_per_segment threshold is not hit exactly)
        while True:
            await asyncio.sleep(config.segment_duration_s)
            if self.state == State.ANALYZING:
                await self._process_buffer()


async def main():
    svc = DiarizationService()
    await svc.run()


if __name__ == "__main__":
    asyncio.run(main())
