#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Buffers a complete grounded English LLM turn and renders it as Singapore
Chinese via MERaLiON-3-10B before it reaches the TTS service.

Partial English token fragments must never reach the renderer: this processor
swallows ``LLMTextFrame`` chunks between
``LLMFullResponseStartFrame``/``LLMFullResponseEndFrame``, renders the
buffered whole turn once, and emits two frames in place of the original text:

- The original English text, tagged ``skip_tts=True`` so
  ``LLMAssistantAggregator`` (downstream, after TTS) still records the
  verified English answer in conversation context, while the TTS service
  passes it straight through unspoken.
- The rendered Singapore Chinese text, tagged ``append_to_context=False`` so it
  is spoken by TTS but never written to context (avoiding duplicate/garbled
  assistant messages).
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from services.meralion_chinese_render import MERaLiONChineseRenderClient

DEFAULT_FALLBACK_TEXT = "歹势，我这时讲袂出话，请汝再问一遍。"


class ChineseRenderer(FrameProcessor):
    """Converts a buffered grounded English LLM turn into spoken Singapore Chinese Hanji."""

    def __init__(
        self,
        render_client: MERaLiONChineseRenderClient,
        *,
        fallback_text: str = DEFAULT_FALLBACK_TEXT,
        **kwargs,
    ):
        """Initialize the renderer.

        Args:
            render_client: Client used to call the MERaLiON-3-10B renderer.
            fallback_text: Static Chinese text spoken if rendering fails
                (never fabricate an audio response from invented content).
        """
        super().__init__(**kwargs)
        self._render_client = render_client
        self._fallback_text = fallback_text
        self._buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush(direction)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _flush(self, direction: FrameDirection) -> None:
        english_text = "".join(self._buffer).strip()
        self._buffer = []

        if not english_text:
            return

        # Preserve the verified English answer in context; never spoken.
        english_frame = TextFrame(text=english_text)
        english_frame.skip_tts = True
        await self.push_frame(english_frame, direction)

        try:
            rendered_text = await self._render_client.render(english_text)
        except Exception as exc:
            logger.error("MERaLiON Singapore Chinese rendering failed: {}", exc)
            rendered_text = ""

        if not rendered_text:
            logger.warning("Empty/failed rendering; using fallback message for TTS")
            rendered_text = self._fallback_text

        logger.info("Rendered turn: '{}'", rendered_text.replace("\n", " ")[:160])

        rendered_frame = TextFrame(text=rendered_text)
        rendered_frame.append_to_context = False
        await self.push_frame(rendered_frame, direction)
