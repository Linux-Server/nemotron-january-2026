#
# LiveKit streaming TTS plugin for the self-hosted Chatterbox TTS server
# (Resemble AI Chatterbox Turbo, deployed on Modal).
#
# The Chatterbox server speaks the exact same WebSocket protocol as the Magpie
# server (init/text/close/cancel -> binary PCM + stream_created /
# segment_complete / done / error), so this is a thin configuration of the
# Magpie adapter in `magpie_tts.py`:
#
#   - Output is 24 kHz s16le mono (Magpie is 22 kHz).
#   - "voice" names a prompt .wav in the chatterbox-tts-voices Modal volume
#     for voice cloning; empty string (or an unknown name) uses the default
#     built-in voice.
#   - The Magpie-specific "language" / "mode" / "preset" fields the adapter
#     sends are accepted and ignored by the Chatterbox server.
#
# Chatterbox has no token-level streaming: each `text` segment is synthesized
# in full on the GPU, then streamed back as a burst of PCM chunks. The Magpie
# adapter's segment shaping (few-words head -> first sentence -> tail) is
# exactly what keeps TTFB low here, since the first segment's synthesis time
# scales with its length.

from __future__ import annotations

import aiohttp

from magpie_tts import TTS as _MagpieProtocolTTS

# Chatterbox (S3Gen vocoder) output sample rate.
CHATTERBOX_SAMPLE_RATE = 24000


class TTS(_MagpieProtocolTTS):
    def __init__(
        self,
        *,
        url: str,
        # Name of a prompt .wav in the chatterbox-tts-voices Modal volume;
        # empty string uses the model's default voice.
        voice: str = "",
        # Each segment arrives as a burst (it is fully synthesized before any
        # bytes are sent), so the prebuffer fills almost instantly once audio
        # starts — but it must stay below the head segment's audio length, or
        # playback silently waits for the *second* segment's synthesis.
        prebuffer_ms: int = 200,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            url=url,
            voice=voice,
            sample_rate=CHATTERBOX_SAMPLE_RATE,
            prebuffer_ms=prebuffer_ms,
            http_session=http_session,
        )

    @property
    def provider(self) -> str:
        return "chatterbox-modal"
