#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MERaLiON speech-translation service for Pipecat.

Named accurately: the MERaLiON-3-3B-ASR-CTM endpoint is exposed here as a
*translation* service, not a transcription service. It accepts VAD-closed
Hokkien (or other supported-language) audio segments and returns English
text, which is emitted as a ``TranscriptionFrame`` so it can plug directly
into Pipecat's existing STT-shaped pipeline slot.
"""

import base64
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


class MERaLiONSpeechTranslationService(SegmentedSTTService):
    """Translates spoken audio (e.g. Hokkien) into English text via MERaLiON.

    Uses the MERaLiON ``/v1/audio/translations`` endpoint. ``SegmentedSTTService``
    buffers raw PCM between VAD start/stop events and hands ``run_stt`` a
    complete WAV-encoded segment, which is base64-encoded as a data URI and
    posted to the translation endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.meralion.ai",
        model: str = "MERaLiON/MERaLiON-3-3B-ASR-CTM",
        source_language: str = "",
        target_language: str = "English",
        request_timeout: float = 90.0,
        **kwargs,
    ):
        """Initialize the MERaLiON speech-translation service.

        Args:
            api_key: MERaLiON API key.
            base_url: MERaLiON API base URL.
            model: MERaLiON translation model name.
            source_language: Source language hint, empty string for auto-detect.
            target_language: Target language for translation (always English by default).
            request_timeout: HTTP request timeout in seconds.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        settings = STTSettings(model=model, language=None)
        super().__init__(settings=settings, **kwargs)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._source_language = source_language
        self._target_language = target_language
        self._client = httpx.AsyncClient(timeout=request_timeout)

    def can_generate_metrics(self) -> bool:
        """This service supports metric generation."""
        return True

    async def set_language(self, language: Language):
        """Update the source-language hint sent to the translation endpoint.

        The endpoint always outputs English; only the source hint changes.
        """
        logger.info(f"MERaLiON source language hint updated to: [{language}]")
        self._source_language = language.value if language else ""

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a translation result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Translate a WAV-encoded audio segment into English text.

        Args:
            audio: Complete WAV-encoded audio for one VAD-closed utterance.

        Yields:
            Frame: A TranscriptionFrame with the English translation, or an
                ErrorFrame if the request fails.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            text = await self._translate(audio)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            if text:
                await self._handle_transcription(text, True, Language.EN)
                logger.debug(f"MERaLiON translation: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language=Language.EN,
                )
            else:
                logger.warning("Received empty translation from MERaLiON API")

        except Exception as e:
            logger.error(f"MERaLiON speech-translation request failed: {e}")
            yield ErrorFrame(error=f"MERaLiON speech-translation error: {e}")

    async def _translate(self, audio: bytes) -> str:
        """POST a WAV audio segment to the MERaLiON translation endpoint.

        Args:
            audio: WAV-encoded audio bytes.

        Returns:
            The translated English text.
        """
        audio_b64 = base64.b64encode(audio).decode("ascii")

        response = await self._client.post(
            f"{self._base_url}/v1/audio/translations",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "audio_url": f"data:audio/wav;base64,{audio_b64}",
                "model": self._settings.model,
                "stream": False,
                "translation_params": {
                    "source_language": self._source_language,
                    "target_language": self._target_language,
                },
            },
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Unexpected MERaLiON response schema: {data}")

        return choices[0]["message"]["content"].strip()

    async def cleanup(self):
        """Release the underlying HTTP client."""
        await super().cleanup()
        await self._client.aclose()
