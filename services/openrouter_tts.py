#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter Qwen-Audio-3.0-TTS-Flash service for Pipecat.

MERaLiON-OmniVoice-Hokkien-TTS was evaluated first but dropped as the
default: local GPU inference was too slow for real-time voice turns (see
scripts/archive/serve_meralion_tts.py and download_meralion_tts.py).

Hokkien caveat: qwen/qwen-audio-3.0-tts-flash's system voices only cover
Mandarin Chinese and English (see Alibaba's Qwen-Audio-TTS voice list —
longanhuan_v3.6 and loongjohn, the two voices exposed on OpenRouter, are
Mandarin/English only; the one dialect voice this model family offers is
Sichuanese, not Hokkien/Minnan). OpenRouter's /audio/speech schema for this
model exposes only `model`, `input`, `voice`, `response_format`, and `speed`
— there is no language/dialect/instruction parameter to request Hokkien
pronunciation. Feeding it Hokkien Hanji text will be read aloud with Mandarin
pronunciation, not Hokkien. There is no reliable way to prompt this specific
model into speaking Hokkien; plain written Chinese is the practical ceiling.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSStoppedFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class OpenRouterTTSSettings(TTSSettings):
    """Settings for OpenRouterTTSService."""

    pass


class OpenRouterTTSService(TTSService):
    """Text-to-speech via OpenRouter's qwen/qwen-audio-3.0-tts-flash endpoint."""

    Settings = OpenRouterTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen-audio-3.0-tts-flash",
        voice: str = "longanhuan_v3.6",
        pcm_sample_rate: int = 24000,
        http_referer: str | None = None,
        title: str | None = None,
        request_timeout: float = 30.0,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the OpenRouter TTS service.

        Args:
            api_key: OpenRouter API key.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            base_url: OpenRouter's OpenAI-compatible base URL.
            model: OpenRouter TTS model slug.
            voice: Provider-specific voice identifier (Mandarin/English only
                for qwen-audio-3.0-tts-flash — see module docstring).
            pcm_sample_rate: Sample rate of the raw PCM audio DashScope returns.
                Not documented by OpenRouter; 24000 matches DashScope's own
                Qwen-Audio-TTS quickstart examples. Override if audio sounds
                pitch-shifted/sped up.
            http_referer: Optional site URL sent for OpenRouter app rankings.
            title: Optional site name sent for OpenRouter app rankings.
            request_timeout: HTTP request timeout in seconds.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(model=None, voice=None, language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._pcm_sample_rate = pcm_sample_rate
        self._http_referer = http_referer
        self._title = title
        self._request_timeout = aiohttp.ClientTimeout(total=request_timeout)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenRouter's TTS endpoint.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._title:
            headers["X-Title"] = self._title

        payload = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": "pcm",
        }

        try:
            async with self._session.post(
                f"{self._base_url}/audio/speech",
                json=payload,
                headers=headers,
                timeout=self._request_timeout,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    yield TTSStoppedFrame(context_id=context_id)
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=self._pcm_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
