#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart — Voice bot.

Required AI services:
 - MERaLiON speech translation API (Hokkien audio → English text)
 - OpenRouter gpt-oss-120b (LLM) — served via OpenAI-compatible HTTP API
 - OpenRouter nvidia/nemotron-3-embed-1b:free (embeddings, for local retrieval)
 - MERaLiON-3-10B (Singapore Chinese renderer, enabled by default; see ENABLE_CHINESE_RENDERER)
 - OpenRouter qwen/qwen-audio-3.0-tts-flash (Text-to-Speech; Mandarin/English only, see services/openrouter_tts.py)

Run the bot using::

    uv run bots/bot.py

Before first run, build the retrieval index (see scripts/build_index.py)::

    uv run scripts/build_index.py

Pipeline::

    transport.input()
        ↓
    stt (MERaLiONSpeechTranslationService)
        ↓
    user_aggregator (LLMUserAggregator)
        ↓
    pre_llm_tagger (UserTurnPreLLMTagger)
        — also runs deterministic hybrid retrieval over knowledge_base/ and
          injects a <retrieved_evidence> block before every LLM call —
        ↓
    llm (OpenAILLMService → OpenRouter gpt-oss-120b)
        ↓
    chinese_renderer (ChineseRenderer → MERaLiON-3-10B)
        — buffers the complete English answer, renders it as Singapore
          Chinese, and keeps the verified English text in context —
        ↓
    tts (OpenRouterTTSService)
        ↓
    transport.output()
        ↓
    assistant_aggregator (LLMAssistantAggregator)
"""

import asyncio
import os
import re
import sys
from pathlib import Path

import aiohttp

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for non-ASCII in logs).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure the project root (pipecat-quickstart/) is on sys.path so local
# packages such as `processors/` are importable regardless of how the bot
# is launched (uv run, python -m, direct script invocation, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from loguru import logger

print("Starting Pipecat bot...")
print("Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from processors.retrieval_tool import format_evidence_block
from processors.sso_tool import make_find_nearest_sso_tool
from processors.chinese_renderer import ChineseRenderer
from retrieval.embeddings import OpenRouterEmbeddingClient
from retrieval.index import SchemeIndex
from retrieval.reranker import OpenRouterRerankClient
from retrieval.sso_lookup import SSODirectory
from services.meralion_speech_translation import MERaLiONSpeechTranslationService
from services.meralion_chinese_render import MERaLiONChineseRenderClient
from services.openrouter_tts import OpenRouterTTSService

if os.name != "nt":
    try:
        from pipecat.transports.daily.transport import DailyParams
    except ModuleNotFoundError:
        DailyParams = None
else:
    DailyParams = None

logger.info("✅ All components loaded successfully!")

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid %s='%s'; falling back to %s", name, raw_value, default)
        return default


def _env_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid %s='%s'; falling back to %s", name, raw_value, default)
        return default


def _env_optional_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid %s='%s'; ignoring value", name, raw_value)
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in ("1", "true", "yes", "on")


def _build_openrouter_settings() -> dict:
    """Collect generation parameters supported by OpenRouter's OpenAI-compatible API."""
    settings: dict = {}

    temperature = _env_optional_float("OPENROUTER_TEMPERATURE")
    if temperature is not None:
        settings["temperature"] = temperature

    top_p = _env_optional_float("OPENROUTER_TOP_P")
    if top_p is not None:
        settings["top_p"] = top_p

    max_tokens = _env_int("OPENROUTER_MAX_TOKENS")
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens

    presence_penalty = _env_optional_float("OPENROUTER_PRESENCE_PENALTY")
    if presence_penalty is not None:
        settings["presence_penalty"] = presence_penalty

    frequency_penalty = _env_optional_float("OPENROUTER_FREQUENCY_PENALTY")
    if frequency_penalty is not None:
        settings["frequency_penalty"] = frequency_penalty

    reasoning_effort = (os.getenv("OPENROUTER_REASONING_EFFORT") or "low").strip()
    if reasoning_effort:
        # OpenRouter's chat-completions schema nests this under a "reasoning"
        # object (`{"reasoning": {"effort": ...}}`), not a flat top-level
        # "reasoning_effort" key — the latter is silently ignored, leaving
        # gpt-oss-120b at its provider-default (often "medium") reasoning
        # effort, which can add 30-45s of thinking time per turn. And since
        # "reasoning" isn't a param the OpenAI SDK's create() recognizes
        # either, it must be passed via extra_body so the SDK forwards it
        # unvalidated instead of raising "unexpected keyword argument".
        settings["extra"] = {"extra_body": {"reasoning": {"effort": reasoning_effort}}}

    return settings


def _resolve_system_prompt() -> str:
    system_prompt_file = (os.getenv("OPENROUTER_SYSTEM_PROMPT_FILE") or "").strip()
    inline_prompt = (os.getenv("OPENROUTER_SYSTEM_PROMPT") or "").strip()

    if system_prompt_file:
        prompt_path = Path(system_prompt_file).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = (ENV_PATH.parent / prompt_path).resolve()

        try:
            prompt_from_file = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(
                "Unable to load OPENROUTER_SYSTEM_PROMPT_FILE='{}': {}",
                prompt_path,
                exc,
            )
            return inline_prompt

        if not prompt_from_file:
            logger.warning(
                "OPENROUTER_SYSTEM_PROMPT_FILE='{}' is empty; falling back to OPENROUTER_SYSTEM_PROMPT",
                prompt_path,
            )
            return inline_prompt

        if inline_prompt:
            logger.warning(
                "Both OPENROUTER_SYSTEM_PROMPT_FILE and OPENROUTER_SYSTEM_PROMPT are set; using file prompt"
            )

        logger.info("Loaded system prompt from file: {}", prompt_path)
        return prompt_from_file

    return inline_prompt


def _load_render_prompt_template(relative_path: str) -> str:
    """Load the MERaLiON-3-10B Chinese-renderer prompt template from disk."""
    prompt_path = Path(relative_path).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (ENV_PATH.parent / prompt_path).resolve()
    return prompt_path.read_text(encoding="utf-8")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")
    logger.info("Loaded env from: {}", ENV_PATH)

    aiohttp_session = aiohttp.ClientSession()

    openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
    logger.info("Using OpenRouter model: {} @ {}", openrouter_model, openrouter_base_url)
    openrouter_settings = _build_openrouter_settings()
    logger.info("OpenRouter generation settings: {}", openrouter_settings)
    openrouter_tts_model = os.getenv("OPENROUTER_TTS_MODEL", "qwen/qwen-audio-3.0-tts-flash")
    openrouter_tts_voice = os.getenv("OPENROUTER_TTS_VOICE", "longanhuan_v3.6")
    system_prompt = _resolve_system_prompt()
    if system_prompt:
        logger.info("Custom system prompt enabled")

    _last_counted_msg_index: int = -1
    _conversation_started = False
    user_turn_texts: list[str] = []
    """Raw text of every user turn so far, oldest first — windowed into the retrieval query so facts stated across turns (citizenship, area, condition, ...) compound instead of being forgotten each turn."""
    vad_stop_secs = _env_float("VAD_STOP_SECS", 0.8)
    user_turn_stop_timeout = _env_float("USER_TURN_STOP_TIMEOUT", 10.0)
    user_speech_timeout = _env_float("USER_SPEECH_TIMEOUT", 4.5)

    meralion_api_key = os.environ["MERALION_API_KEY"]
    meralion_base_url = os.getenv("MERALION_API_BASE_URL", "https://api.meralion.ai")
    meralion_translation_model = os.getenv(
        "MERALION_TRANSLATION_MODEL", "MERaLiON/MERaLiON-3-3B-ASR-CTM"
    )
    logger.info(
        "Using MERaLiON speech translation model: {} @ {}",
        meralion_translation_model,
        meralion_base_url,
    )
    stt = MERaLiONSpeechTranslationService(
        api_key=meralion_api_key,
        base_url=meralion_base_url,
        model=meralion_translation_model,
    )

    # --- MERaLiON-3-10B Singapore Chinese renderer -------------------------------
    # Narrow transformation only (STREAMLINED_PIPELINE_CONTEXT.md section 3.5):
    # verified grounded English answer -> Singapore Chinese. Runs on the
    # complete buffered LLM response (see processors/chinese_renderer.py) so no
    # partial English fragments ever reach the renderer or the TTS model.
    enable_chinese_renderer = _env_bool("ENABLE_CHINESE_RENDERER", True)
    chinese_renderer: ChineseRenderer | None = None
    chinese_render_client: MERaLiONChineseRenderClient | None = None
    if enable_chinese_renderer:
        render_prompt_file = os.getenv(
            "MERALION_RENDER_PROMPT_FILE", "prompt/english_to_singapore-chinese.md"
        )
        render_model = os.getenv("MERALION_RENDER_MODEL", "MERaLiON/MERaLiON-3-10B")
        logger.info(
            "Using MERaLiON Singapore Chinese renderer model: {} @ {} (prompt={})",
            render_model,
            meralion_base_url,
            render_prompt_file,
        )
        chinese_render_client = MERaLiONChineseRenderClient(
            api_key=meralion_api_key,
            prompt_template=_load_render_prompt_template(render_prompt_file),
            base_url=meralion_base_url,
            model=render_model,
            temperature=_env_float("MERALION_RENDER_TEMPERATURE", 0.2),
        )
        chinese_renderer = ChineseRenderer(chinese_render_client)
    else:
        logger.warning(
            "ENABLE_CHINESE_RENDERER=false: grounded English answers are sent "
            "directly to the TTS model, skipping the MERaLiON-3-10B "
            "Hanji-rendering step."
        )

    logger.info("Using OpenRouter TTS model: {} (voice={})", openrouter_tts_model, openrouter_tts_voice)
    tts = OpenRouterTTSService(
        api_key=os.getenv("OPENROUTER_TTS_API_KEY") or openrouter_api_key,
        aiohttp_session=aiohttp_session,
        base_url=os.getenv("OPENROUTER_TTS_BASE_URL", openrouter_base_url),
        model=openrouter_tts_model,
        voice=openrouter_tts_voice,
        pcm_sample_rate=_env_int("OPENROUTER_TTS_PCM_SAMPLE_RATE", 24000),
    )

    # --- Local retrieval over the curated SupportGoWhere knowledge base ---------
    # Retrieval-first policy (STREAMLINED_PIPELINE_CONTEXT.md section 2): evidence
    # is fetched deterministically for every user turn (see
    # `_update_retrieval_evidence` below) rather than via LLM tool calling, so the
    # grounded-answer call never depends on gpt-oss-120b successfully emitting a
    # tool call.
    retrieval_data_dir = _PROJECT_ROOT / os.getenv("RETRIEVAL_DATA_DIR", "data")
    retrieval_top_k = _env_int("RETRIEVAL_TOP_K", 3)
    retrieval_min_score = _env_float("RETRIEVAL_MIN_SCORE", 0.0)
    retrieval_alpha = _env_float("RETRIEVAL_HYBRID_ALPHA", 0.7)
    retrieval_context_window_turns = _env_int("RETRIEVAL_CONTEXT_WINDOW_TURNS", 6)
    enable_sso_lookup_tool = _env_bool("ENABLE_SSO_LOOKUP_TOOL", True)

    reranker_model = (os.getenv("RERANKER_MODEL") or "").strip() or None
    rerank_candidates = _env_int("RERANK_CANDIDATES", 10)
    rerank_client: OpenRouterRerankClient | None = None
    if reranker_model:
        rerank_client = OpenRouterRerankClient(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
            model=reranker_model,
        )
        logger.info(
            "Using OpenRouter reranker model: {} (candidates={}, top_k={})",
            reranker_model,
            rerank_candidates,
            retrieval_top_k,
        )

    embed_client = OpenRouterEmbeddingClient(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"),
        query_prefix=os.getenv("OPENROUTER_EMBED_QUERY_PREFIX", ""),
        passage_prefix=os.getenv("OPENROUTER_EMBED_PASSAGE_PREFIX", ""),
    )

    scheme_index: SchemeIndex | None = None
    try:
        scheme_index = SchemeIndex.load(retrieval_data_dir)
        logger.info(
            "Loaded retrieval index: {} schemes (embedding_model={})",
            scheme_index.scheme_count,
            scheme_index.manifest.get("embedding_model"),
        )
    except FileNotFoundError:
        logger.warning(
            "No retrieval index found at {}. Run `uv run scripts/build_index.py` "
            "first — retrieval will be skipped until then.",
            retrieval_data_dir,
        )

    llm = OpenAILLMService(
        api_key=openrouter_api_key,
        base_url=openrouter_base_url,
        model=openrouter_model,
        params=OpenAILLMService.InputParams(**openrouter_settings) if openrouter_settings else None,
    )

    tools = []
    if enable_sso_lookup_tool:
        sso_locations_path = _PROJECT_ROOT / os.getenv(
            "SSO_LOCATIONS_FILE", "knowledge_base/social_service_offices/sso_locations.json"
        )
        sso_directory = SSODirectory.load(sso_locations_path)
        tools.append(make_find_nearest_sso_tool(sso_directory))
        logger.info("find_nearest_sso tool enabled (data: {})", sso_locations_path)
        logger.warning(
            "ENABLE_SSO_LOOKUP_TOOL=true: follow-up tool calling is enabled on "
            "openai/gpt-oss-120b via OpenRouter, which has a documented "
            "broken-streaming-tool-call issue on at least one OpenRouter upstream "
            "provider. Set ENABLE_SSO_LOOKUP_TOOL=false if find_nearest_sso calls "
            "fail to stream correctly."
        )

    context = LLMContext(tools=tools) if tools else LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs)),
            user_turn_stop_timeout=user_turn_stop_timeout,
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=user_speech_timeout,
                    )
                ],
            ),
        ),
    )

    logger.info(
        "Turn detection config | vad_stop_secs={} | user_speech_timeout={} | user_turn_stop_timeout={}",
        vad_stop_secs,
        user_speech_timeout,
        user_turn_stop_timeout,
    )

    _evidence_re = re.compile(r'<retrieved_evidence>(.*?)</retrieved_evidence>', re.DOTALL)

    def _update_retrieved_evidence_metadata(evidence_text: str) -> None:
        """Atomically replace the live <retrieved_evidence> block with fresh evidence."""
        new_block = f"<retrieved_evidence>\n{evidence_text}\n</retrieved_evidence>"
        messages = context.get_messages()
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                content = messages[i]["content"]
                if _evidence_re.search(content):
                    messages[i]["content"] = _evidence_re.sub(new_block, content, count=1)
                else:
                    messages[i]["content"] = content.rstrip("\n") + f"\n\n{new_block}"
                return
        # No system message found — create one.
        context.add_message({"role": "system", "content": new_block})

    async def _update_retrieval_evidence(query_text: str) -> None:
        """Deterministic retrieval-first evidence lookup, windowed over recent user turns.

        ``query_text`` is the last ``RETRIEVAL_CONTEXT_WINDOW_TURNS`` user turns
        concatenated (see ``_tag_latest_user_message_if_needed``), not just the
        latest turn — so a fact stated a turn or two ago (e.g. "I'm a Singapore
        Citizen", "I'm unemployed") still contributes to this turn's search
        instead of being dropped once superseded in the evidence block. Runs
        before every LLM call (see STREAMLINED_PIPELINE_CONTEXT.md section 2's
        "Important optimization") so the grounded answer never depends on the LLM
        successfully emitting a tool call.
        """
        if scheme_index is None or not query_text.strip():
            _update_retrieved_evidence_metadata(
                "No verified scheme evidence was found for this turn."
            )
            return

        try:
            query_embedding = await embed_client.embed_query(query_text)
            fetch_k = max(retrieval_top_k, rerank_candidates) if rerank_client else retrieval_top_k
            results = scheme_index.search(
                query_text,
                query_embedding,
                top_k=fetch_k,
                min_score=retrieval_min_score,
                alpha=retrieval_alpha,
            )
        except Exception as exc:
            logger.warning("Retrieval failed for this turn, continuing without evidence: {}", exc)
            _update_retrieved_evidence_metadata("Retrieval was unavailable for this turn.")
            return

        if rerank_client and results:
            try:
                documents = [r.scheme.embedding_text or r.scheme.description for r in results]
                reranked = await rerank_client.rerank(query_text, documents, top_n=retrieval_top_k)
                results = [results[i] for i, _ in reranked]
            except Exception as exc:
                # :free-tier reranker availability/rate limits are less reliable than
                # the core hybrid search — degrade to plain hybrid ranking, not to no evidence.
                logger.warning("Reranking failed for this turn, falling back to hybrid ranking: {}", exc)
                results = results[:retrieval_top_k]
        else:
            results = results[:retrieval_top_k]

        evidence_text = format_evidence_block(results)
        logger.info(
            "Retrieval-first evidence for turn {} | top_score={} | schemes={}",
            len(user_turn_texts),
            f"{results[0].hybrid_score:.2f}" if results else "n/a",
            [r.scheme.scheme_id for r in results],
        )

        _update_retrieved_evidence_metadata(evidence_text)

    async def _tag_latest_user_message_if_needed() -> tuple[bool, str]:
        nonlocal _last_counted_msg_index
        messages = context.get_messages()
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue

            content = messages[i].get("content", "")
            if i == _last_counted_msg_index:
                return False, content

            _last_counted_msg_index = i
            user_turn_texts.append(content)
            windowed_query = "\n".join(user_turn_texts[-retrieval_context_window_turns:])
            await _update_retrieval_evidence(windowed_query)
            return True, content

        return False, ""

    class UserTurnPreLLMTagger(FrameProcessor):
        """Ensures latest user message is tagged before LLM consumes context."""

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if direction != FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)
                return

            # Chat transports can enqueue text and trigger the LLM quickly.
            # Tag here so the same-turn LLM request includes [UserTurn N].
            tagged, tagged_content = await _tag_latest_user_message_if_needed()
            if tagged:
                logger.info(
                    "User turn metadata updated in system prompt | count={} | text='{}'",
                    len(user_turn_texts),
                    tagged_content.replace("\n", " ")[:120],
                )

            await self.push_frame(frame, direction)

    pre_llm_tagger = UserTurnPreLLMTagger()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator, strategy, message: UserTurnStoppedMessage
    ):
        tagged, tagged_content = await _tag_latest_user_message_if_needed()

        logger.info(
            "User turn stop event | tagged={} | count={} | text='{}'",
            tagged,
            len(user_turn_texts),
            (tagged_content or message.content or "").replace("\n", " ")[:120],
        )

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator):
        # Fallback for transports where typed chat does not fire user turn stop.
        tagged, tagged_content = await _tag_latest_user_message_if_needed()
        if not tagged:
            # Some transports append chat text to context slightly after this event.
            await asyncio.sleep(0)
            tagged, tagged_content = await _tag_latest_user_message_if_needed()

        if not tagged:
            return

        logger.info(
            "User turn fallback tag applied before assistant response | count={} | text='{}'",
            len(user_turn_texts),
            tagged_content.replace("\n", " ")[:120],
        )

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator, message: AssistantTurnStoppedMessage
    ):
        # Final fallback for typed chat that lands in context after turn start.
        tagged, tagged_content = await _tag_latest_user_message_if_needed()
        if not tagged:
            return

        logger.info(
            "User turn late fallback tag applied after assistant response | count={} | text='{}'",
            len(user_turn_texts),
            tagged_content.replace("\n", " ")[:120],
        )

    pipeline_stages = [
        transport.input(),   # Transport user input
        stt,                 # Speech-to-Text
        user_aggregator,     # Aggregate user speech into context
        pre_llm_tagger,      # Ensure [UserTurn N] tag exists before LLM request
        llm,                 # LLM
    ]
    if chinese_renderer is not None:
        pipeline_stages.append(chinese_renderer)  # English -> Singapore Chinese
    pipeline_stages += [
        tts,                 # TTS
        transport.output(),  # Transport bot output
        assistant_aggregator,  # Assistant spoken responses
    ]
    pipeline = Pipeline(pipeline_stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal _conversation_started
        logger.info("Client connected")
        if _conversation_started:
            # WebRTC renegotiation fires on_client_connected again on the same
            # session. Skip re-initialization so the pipeline continues without
            # restarting the conversation.
            logger.info("WebRTC renegotiation detected — skipping re-initialization")
            return
        _conversation_started = True
        if system_prompt:
            context.add_message({"role": "system", "content": system_prompt})
        # Kick off the conversation.
        context.add_message(
            {
                "role": "user",
                "content": (
                    "Say a short, warm hello and ask what support or help they're "
                    "looking for today."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    try:
        await runner.run(task)
    finally:
        await aiohttp_session.close()
        await embed_client.aclose()
        if rerank_client is not None:
            await rerank_client.aclose()
        if chinese_render_client is not None:
            await chinese_render_client.aclose()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    if DailyParams is not None:
        transport_params["daily"] = lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    else:
        logger.warning(
            "Daily transport is unavailable in this environment. "
            "Use -t webrtc or run in WSL/Linux for Daily."
        )

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
