"""Streaming TTS for Kokoro (https://github.com/hexgrad/kokoro).

A standalone, framework-agnostic module: it produces audio chunks from text and
knows nothing about any particular server, transport, or bot framework. Feed it
a string or a stream of text (LLM tokens, stdin, a queue) and it yields PCM as
soon as each piece is rendered.

Kokoro-82M is non-autoregressive (StyleTTS2-style): one forward pass renders a
whole chunk of phonemes at once, with no token-by-token seam to stream from.
So latency here is purely a function of *chunk size* — time-to-first-audio is
the render time of the first chunk. Two things follow, and they are what this
module implements:

  * Split long text at punctuation, with a deliberately short first chunk, so
    the opening clause is rendered and emitted while the rest is still being
    synthesized. Short text passes through whole — splitting it would only add
    prosody seams (each chunk is synthesized independently and gets its own
    sentence-final contour) without meaningfully improving latency.
  * When the input is itself a stream, buffer to sentence-ish boundaries before
    synthesizing, flushing the first boundary early and later ones lazily.

Everything is 24kHz mono, which is fixed by the model.

Install:
    pip install kokoro soundfile        # plus the espeak-ng binary for G2P
    # apt-get install espeak-ng  /  brew install espeak-ng

Library use:
    tts = KokoroStreamingTTS(voice="af_heart")
    for chunk in tts.stream("Hello there. This is Kokoro streaming."):
        sink.write(chunk.pcm_bytes)          # s16le mono 24kHz

    # from a token stream (e.g. an LLM), sentence-buffered internally
    for chunk in tts.stream_text_stream(token_iterator):
        sink.write(chunk.pcm_bytes)

    # async variants run synthesis in a worker thread
    async for chunk in tts.astream("Hello there."):
        await sink.send(chunk.pcm_bytes)

CLI:
    python -m nemotron_speech.kokoro_streaming_tts "Hello there." -o out.wav
    python -m nemotron_speech.kokoro_streaming_tts --stdin -o out.wav
    python -m nemotron_speech.kokoro_streaming_tts --list-voices
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
import threading
import time
import wave
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # fixed by the Kokoro decoder
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "af_heart"

# Kokoro voice names are <lang><gender>_<name>; the first character is the
# language code, which also selects the G2P pipeline.
LANG_NAMES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# Split at any punctuation a speaker would pause on — sentence-final *and*
# clause-internal. Splitting on sentence ends alone leaves the opening chunk as
# a whole sentence, which is the dominant term in time-to-first-audio; clause
# breaks give the packer somewhere earlier to cut. Pieces are merged back up to
# max_chunk_chars afterwards, so the extra granularity only costs anything at
# the start of an utterance.
_PUNCT_SPLIT = re.compile(r"(?<=[.!?…;:—,])\s+")
_WORD_SPLIT = re.compile(r"\s+")

# A boundary in a *streaming* text feed. The first flush may cut at a clause
# (speech starts a clause earlier, and we stop waiting on the producer to
# finish a long opening sentence); later flushes wait for sentence ends so
# prosody has full context.
_SENTENCE_BOUNDARY = re.compile(r"[.!?…](?=[\s\"')\]]|$)|\n")
_CLAUSE_BOUNDARY = re.compile(r"[.!?…;:—,](?=[\s\"')\]]|$)|\n")


@dataclass(slots=True)
class ChunkConfig:
    """How text is carved into synthesis chunks.

    Kokoro renders a chunk in one forward pass at RTF ~0.1 on a small GPU, so
    time-to-first-audio ≈ 0.1 × (duration of the first chunk). English runs
    ~15 characters of text per second of speech, which makes the arithmetic
    concrete: a 300-char first chunk is ~20s of audio and ~2s of latency; a
    40-char one is ~2.6s of audio and ~250ms. Hence a small `first_chunk_chars`
    and a large `max_chunk_chars` — after the first chunk there is an audio
    cushion playing, so later chunks should be big for the sake of prosody.

    `split_threshold` is the "don't bother" line: text shorter than this
    renders fast enough that splitting would only add seams.
    """

    split_threshold: int = 90
    first_chunk_chars: int = 40
    max_chunk_chars: int = 300

    # Ceiling on the first chunk when there is no punctuation to cut at before
    # it. A word-boundary cut mid-clause gets a wrong contour from the model,
    # so it is only worth it to escape the alternative: a run-on sentence
    # rendered whole, where latency is the full utterance. 0 disables it (the
    # first chunk then follows punctuation only, however long that takes).
    first_chunk_hard_cap: int = 120

    # Streaming-input buffering: release at a boundary once `min_flush_chars`
    # have accumulated (`first_flush_chars` for the very first release), or at
    # a word boundary once `force_flush_chars` accumulate with none in sight.
    # first_flush_chars=1 means the first boundary always wins, however short:
    # "Hi there." should start playing immediately.
    first_flush_chars: int = 1
    min_flush_chars: int = 120
    force_flush_chars: int = 400

    # Let the first release cut at a clause boundary (comma, semicolon) rather
    # than waiting for a sentence end. Big TTFA win on utterances that open
    # with a long sentence; costs a slightly odd contour on that one fragment.
    first_flush_on_clause: bool = True


@dataclass(slots=True)
class AudioChunk:
    """One rendered piece of audio, plus what produced it."""

    audio: np.ndarray  # float32 mono, [-1, 1], SAMPLE_RATE
    text: str
    index: int
    phonemes: str = ""
    latency_ms: float = 0.0

    @property
    def pcm_bytes(self) -> bytes:
        """s16le mono PCM — what most audio sinks want."""
        return (np.clip(self.audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def split_for_streaming(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Carve `text` into chunks: short first, larger after. Pure function.

    Cuts land on punctuation. Text with no punctuation to cut at is left whole
    unless it exceeds `max_chunk_chars`, in which case it is broken at word
    boundaries — a mid-clause cut gets a wrong (falling) contour from the
    model, so it is a last resort rather than a latency tactic.
    """
    config = config or ChunkConfig()
    text = text.strip()
    if not text:
        return []
    if len(text) <= config.split_threshold:
        return [text]

    pieces = [p for p in _PUNCT_SPLIT.split(text) if p.strip()]
    if len(pieces) == 1:
        if len(text) <= config.max_chunk_chars:
            return _apply_first_chunk_cap([text], config)
        pieces = [p for p in _WORD_SPLIT.split(text) if p]

    chunks: list[str] = []
    current = ""
    limit = config.first_chunk_chars
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            chunks.append(current)
            current = piece
            limit = config.max_chunk_chars
        else:
            current = f"{current} {piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return _apply_first_chunk_cap(chunks, config)


def _apply_first_chunk_cap(chunks: list[str], config: ChunkConfig) -> list[str]:
    """Break an over-long first chunk at a word boundary.

    Only reached when the opening clause ran past `first_chunk_hard_cap` with
    no punctuation to cut at — the choice is a mid-clause seam or paying the
    whole clause's render time before any audio.
    """
    cap = config.first_chunk_hard_cap
    if not chunks or cap <= 0 or len(chunks[0]) <= cap:
        return chunks
    head = chunks[0]
    cut = head.rfind(" ", 0, cap + 1)
    if cut <= 0:
        return chunks  # a single enormous word; nothing sensible to do
    return [head[:cut].rstrip(), head[cut:].lstrip(), *chunks[1:]]


class SentenceBuffer:
    """Accumulates streamed text and releases synthesis-sized pieces.

    The first release is deliberately eager (a short opening clause gets audio
    playing sooner); later ones wait for more text so prosody has context.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._buf = ""
        self._flushed = 0

    @property
    def pending(self) -> str:
        return self._buf

    def push(self, text: str) -> list[str]:
        """Add text; return whatever is ready to synthesize (possibly none)."""
        self._buf += text
        ready: list[str] = []
        while True:
            piece = self._take()
            if piece is None:
                break
            ready.append(piece)
        return ready

    def flush(self) -> str | None:
        """Release whatever is left, at end of stream."""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._flushed += 1
            return rest
        return None

    def _take(self) -> str | None:
        cfg = self.config
        is_first = self._flushed == 0
        threshold = cfg.first_flush_chars if is_first else cfg.min_flush_chars
        boundary = (
            _CLAUSE_BOUNDARY if is_first and cfg.first_flush_on_clause else _SENTENCE_BOUNDARY
        )

        cut = -1
        for match in boundary.finditer(self._buf):
            if match.end() >= threshold:
                cut = match.end()
                break
        if cut < 0 and len(self._buf) >= cfg.force_flush_chars:
            # No boundary in a long run of text: fall back to the last space so
            # we at least don't cut a word in half.
            space = self._buf.rfind(" ", 0, cfg.force_flush_chars)
            cut = space if space > 0 else cfg.force_flush_chars
        if cut < 0:
            return None

        piece, self._buf = self._buf[:cut], self._buf[cut:].lstrip()
        piece = piece.strip()
        if not piece:
            return None
        self._flushed += 1
        return piece


class KokoroStreamingTTS:
    """Kokoro-82M with chunk-level streaming synthesis.

    Thread-safe for sequential use from multiple threads (a lock serializes
    inference); concurrent callers are queued, not interleaved.

    Args:
        voice: default voice name, e.g. "af_heart". Its first letter picks the
            language pipeline. Pass a different one per call to override.
        speed: default speaking rate multiplier.
        device: "cuda" / "mps" / "cpu"; autodetected when None.
        repo_id: HF repo for weights and voice packs.
        config: chunking/buffering knobs (see ChunkConfig).
        trim_silence: strip the padding silence Kokoro leaves around each chunk
            so joined chunks don't have audible gaps at the seams. The lead-in
            of the first chunk and the release of the last are preserved.
        lazy: defer model loading until first use (default loads eagerly, which
            is usually what you want — the first render is otherwise slow).
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        device: str | None = None,
        repo_id: str = DEFAULT_REPO_ID,
        config: ChunkConfig | None = None,
        trim_silence: bool = True,
        lazy: bool = False,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.repo_id = repo_id
        self.config = config or ChunkConfig()
        self.trim_silence = trim_silence
        self.device = device or _autodetect_device()

        self._model: Any = None
        self._pipelines: dict[str, Any] = {}
        self._voice_packs: dict[str, torch.FloatTensor] = {}
        self._lock = threading.Lock()

        if not lazy:
            self.load()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load weights and the default voice. Idempotent."""
        if self._model is not None:
            return
        from kokoro import KModel

        logger.info("Loading Kokoro (%s) on %s", self.repo_id, self.device)
        start = time.perf_counter()
        model = KModel(repo_id=self.repo_id).to(self.device).eval()
        self._model = model
        self._get_voice_pack(self.voice)  # pulls the pack + builds the pipeline
        logger.info("Kokoro loaded in %.1fs", time.perf_counter() - start)

    def warmup(self, text: str = "Warm up the decoder and the phoneme pipeline.") -> float:
        """Run one synthesis to pay JIT/lexicon costs up front. Returns seconds."""
        self.load()
        start = time.perf_counter()
        for _ in self.stream(text):
            pass
        elapsed = time.perf_counter() - start
        logger.info("Warmup took %.2fs", elapsed)
        return elapsed

    def list_voices(self) -> list[str]:
        """Voice names available in the repo (network call, cached by HF hub)."""
        from huggingface_hub import list_repo_files

        return sorted(
            f.removeprefix("voices/").removesuffix(".pt")
            for f in list_repo_files(self.repo_id)
            if f.startswith("voices/") and f.endswith(".pt")
        )

    # -- synthesis ---------------------------------------------------------

    def stream(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield audio chunks for `text`, first chunk as early as possible.

        `cancel` is polled between chunks; a single forward pass is short
        enough (tens of ms on GPU) that finer-grained cancellation would buy
        nothing but complexity.
        """
        import torch

        self.load()
        voice = voice or self.voice
        speed = self.speed if speed is None else speed

        pipeline = self._get_pipeline(voice)
        pack = self._get_voice_pack(voice)
        chunks = split_for_streaming(text, self.config)
        start = time.perf_counter()

        for i, chunk_text in enumerate(chunks):
            if cancel is not None and cancel():
                return
            is_first = i == 0
            is_last = i == len(chunks) - 1

            with self._lock, torch.inference_mode():
                # split_pattern=None: chunking is ours. The pipeline still
                # applies its own hard 510-phoneme cap on top, and can yield
                # more than one result for a very long chunk — emit each.
                results = list(pipeline(chunk_text, voice=pack, speed=speed, split_pattern=None))

            for j, result in enumerate(results):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().float().numpy()
                if self.trim_silence:
                    audio = _trim_silence(
                        audio,
                        head=not (is_first and j == 0),
                        tail=not (is_last and j == len(results) - 1),
                    )
                if audio.size == 0:
                    continue
                yield AudioChunk(
                    audio=audio,
                    text=chunk_text,
                    index=i,
                    phonemes=result.phonemes or "",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

    def stream_text_stream(
        self,
        text_stream: Iterable[str],
        voice: str | None = None,
        speed: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[AudioChunk]:
        """Synthesize a *stream* of text (LLM tokens, stdin, …) as it arrives.

        Text is buffered to sentence-ish boundaries by SentenceBuffer, so
        synthesis of the opening clause overlaps with the producer still
        generating the rest.
        """
        buffer = SentenceBuffer(self.config)
        index = 0
        for piece in text_stream:
            if cancel is not None and cancel():
                return
            for ready in buffer.push(piece):
                for chunk in self.stream(ready, voice, speed, cancel):
                    yield _reindex(chunk, index)
                    index += 1
        rest = buffer.flush()
        if rest and not (cancel is not None and cancel()):
            for chunk in self.stream(rest, voice, speed, cancel):
                yield _reindex(chunk, index)
                index += 1

    def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> np.ndarray:
        """Render `text` fully and return one float32 waveform."""
        parts = [c.audio for c in self.stream(text, voice, speed)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def save_wav(
        self,
        text: str,
        path: str | Path,
        voice: str | None = None,
        speed: float | None = None,
    ) -> Path:
        """Render `text` to a 24kHz mono 16-bit WAV file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, self.synthesize(text, voice, speed))
        return out

    # -- async -------------------------------------------------------------

    async def astream(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Async `stream`: synthesis runs in a worker thread, so the event loop
        stays responsive (to a cancel signal, a socket, an interruption)."""
        async for chunk in _to_async(lambda: self.stream(text, voice, speed, cancel)):
            yield chunk

    async def astream_text_stream(
        self,
        text_stream: AsyncIterator[str],
        voice: str | None = None,
        speed: float | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Async counterpart of `stream_text_stream`, for an async text source.

        Each buffered piece is synthesized off-loop while the producer keeps
        filling the buffer.
        """
        buffer = SentenceBuffer(self.config)
        index = 0
        async for piece in text_stream:
            if cancel is not None and cancel():
                return
            for ready in buffer.push(piece):
                async for chunk in self.astream(ready, voice, speed, cancel):
                    yield _reindex(chunk, index)
                    index += 1
        rest = buffer.flush()
        if rest and not (cancel is not None and cancel()):
            async for chunk in self.astream(rest, voice, speed, cancel):
                yield _reindex(chunk, index)
                index += 1

    # -- internals ---------------------------------------------------------

    def _get_pipeline(self, voice: str) -> Any:
        """G2P pipeline for the voice's language, cached per language code."""
        from kokoro import KPipeline

        lang_code = voice[0]
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            pipeline = KPipeline(lang_code=lang_code, repo_id=self.repo_id, model=self._model)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def _get_voice_pack(self, voice: str) -> torch.FloatTensor:
        """Voice embedding pack, cached per voice (hub download on first use)."""
        pack = self._voice_packs.get(voice)
        if pack is None:
            pack = self._get_pipeline(voice).load_voice(voice)
            self._voice_packs[voice] = pack
        return pack


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SILENCE_THRESHOLD = 0.005
_SILENCE_KEEP_SAMPLES = int(0.02 * SAMPLE_RATE)  # 20ms of margin


def _trim_silence(audio: np.ndarray, head: bool, tail: bool) -> np.ndarray:
    """Trim near-silence from the ends, keeping a small margin."""
    loud = np.flatnonzero(np.abs(audio) > _SILENCE_THRESHOLD)
    if loud.size == 0:
        return audio[:0]
    start = max(0, int(loud[0]) - _SILENCE_KEEP_SAMPLES) if head else 0
    end = min(len(audio), int(loud[-1]) + 1 + _SILENCE_KEEP_SAMPLES) if tail else len(audio)
    return audio[start:end]


def _autodetect_device() -> str:
    try:
        import torch
    except ImportError:  # let the real error surface at load()
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _reindex(chunk: AudioChunk, index: int) -> AudioChunk:
    chunk.index = index
    return chunk


async def _to_async(make_iter: Callable[[], Iterator[AudioChunk]]) -> AsyncIterator[AudioChunk]:
    """Run a blocking chunk generator in a thread, yielding chunks as they land."""
    loop = asyncio.get_running_loop()
    out: asyncio.Queue[AudioChunk | BaseException | None] = asyncio.Queue()

    def produce() -> None:
        try:
            for chunk in make_iter():
                loop.call_soon_threadsafe(out.put_nowait, chunk)
        except BaseException as e:  # noqa: BLE001 - re-raised on the loop side
            loop.call_soon_threadsafe(out.put_nowait, e)
            return
        loop.call_soon_threadsafe(out.put_nowait, None)

    producer = loop.run_in_executor(None, produce)
    try:
        while True:
            item = await out.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        with contextlib.suppress(Exception):
            await producer


def write_wav(
    dest: str | Path | IO[bytes], audio: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> None:
    """Write a float32 [-1, 1] mono waveform as a 16-bit WAV (stdlib only).

    `dest` is a path or an open binary file object (e.g. io.BytesIO, to get the
    bytes back without touching disk).
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    target = str(dest) if isinstance(dest, (str, Path)) else dest
    with wave.open(target, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm)


@dataclass(slots=True)
class StreamStats:
    """Timing summary for one stream, for benchmarking chunk settings."""

    ttfa_ms: float = 0.0  # time to first audio
    total_ms: float = 0.0
    audio_ms: float = 0.0
    chunks: int = 0
    chunk_texts: list[str] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """Real-time factor: <1 means faster than playback."""
        return self.total_ms / max(self.audio_ms, 1e-6)

    def __str__(self) -> str:
        return (
            f"TTFA {self.ttfa_ms:.0f}ms | {self.audio_ms / 1000:.1f}s audio in "
            f"{self.total_ms / 1000:.2f}s (RTF {self.rtf:.3f}) | {self.chunks} chunks"
        )


def measure(chunks: Iterable[AudioChunk]) -> tuple[np.ndarray, StreamStats]:
    """Drain a chunk stream, returning the joined audio plus timings."""
    stats = StreamStats()
    parts: list[np.ndarray] = []
    start = time.perf_counter()
    for chunk in chunks:
        if not parts:
            stats.ttfa_ms = (time.perf_counter() - start) * 1000
        parts.append(chunk.audio)
        stats.chunks += 1
        stats.chunk_texts.append(chunk.text)
    stats.total_ms = (time.perf_counter() - start) * 1000
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    stats.audio_ms = len(audio) / SAMPLE_RATE * 1000
    return audio, stats


def _stdin_stream(chunk_size: int = 16) -> Iterator[str]:
    """Read stdin as a text stream, to exercise the streaming-input path."""
    while True:
        piece = sys.stdin.read(chunk_size)
        if not piece:
            return
        yield piece


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kokoro_streaming_tts",
        description="Streaming TTS with Kokoro-82M (24kHz mono).",
    )
    parser.add_argument("text", nargs="?", help="text to speak (omit with --stdin)")
    parser.add_argument("-o", "--output", default="kokoro_out.wav", help="output WAV path")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE)
    parser.add_argument("-s", "--speed", type=float, default=1.0)
    parser.add_argument("-d", "--device", default=None, help="cuda | mps | cpu")
    parser.add_argument("--stdin", action="store_true", help="stream text from stdin")
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--no-trim", action="store_true", help="keep inter-chunk silence")
    parser.add_argument("--warmup", action="store_true", help="warm up before timing")
    # ChunkConfig uses slots=True, so defaults live on an instance, not the class.
    defaults = ChunkConfig()
    parser.add_argument("--first-chunk-chars", type=int, default=defaults.first_chunk_chars)
    parser.add_argument("--max-chunk-chars", type=int, default=defaults.max_chunk_chars)
    parser.add_argument("--split-threshold", type=int, default=defaults.split_threshold)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_voices:
        tts = KokoroStreamingTTS(voice=args.voice, device=args.device, lazy=True)
        for voice in tts.list_voices():
            lang = LANG_NAMES.get(voice[0], "?")
            print(f"{voice:<20} {lang}")
        return 0

    if not args.text and not args.stdin:
        parser.error("provide TEXT or --stdin")

    tts = KokoroStreamingTTS(
        voice=args.voice,
        speed=args.speed,
        device=args.device,
        trim_silence=not args.no_trim,
        config=ChunkConfig(
            split_threshold=args.split_threshold,
            first_chunk_chars=args.first_chunk_chars,
            max_chunk_chars=args.max_chunk_chars,
        ),
    )
    if args.warmup:
        tts.warmup()

    if args.stdin:
        stream = tts.stream_text_stream(_stdin_stream())
    else:
        stream = tts.stream(args.text)

    audio, stats = measure(stream)
    if audio.size == 0:
        print("No audio generated (empty input?)", file=sys.stderr)
        return 1

    write_wav(args.output, audio)
    print(stats)
    print(f"Wrote {args.output}  (play: ffplay -autoexit {args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
