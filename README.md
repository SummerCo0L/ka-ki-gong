# Ka-Ki-Gong: Hokkien Voice Access to Singapore Senior Support Schemes

Ka-Ki-Gong ("speak for yourself" in Hokkien) is a voice assistant prototype that
helps Singapore Hokkien-speaking seniors discover and understand caregiving and
senior-support schemes. A spoken Hokkien question is translated to English,
answered using a curated, locally-retrieved snapshot of
[SupportGoWhere](https://supportgowhere.life.gov.sg/) scheme pages, and spoken
back in Singapore Chinese through a cloud text-to-speech model.

This is an accessibility research prototype, not an official integration with
SupportGoWhere, LifeSG, GovTech, MSF, MOH, AIC, or any other government agency.
It does not confirm eligibility or submit applications. See
[STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) for the full
design specification this implementation follows.

## Status

**Implemented and wired into `bots/bot.py`:**

- Hokkien (or other supported language) speech → English text, via the
  MERaLiON speech-translation API ([services/meralion_speech_translation.py](services/meralion_speech_translation.py))
- Local hybrid retrieval (dense + lexical) over 303 curated SupportGoWhere
  scheme records in [knowledge_base/](knowledge_base/), run deterministically
  before every turn ([retrieval/](retrieval/), [processors/retrieval_tool.py](processors/retrieval_tool.py))
- Grounded English answer generation via `openai/gpt-oss-120b` through
  OpenRouter, with retrieved evidence — including every scraped section
  (eligibility, how to apply, phone number, etc.), not just highlights —
  injected into the system prompt each turn
- MERaLiON-3-10B English → Singapore Chinese rendering, run once per complete
  buffered assistant turn before TTS
  ([services/meralion_chinese_render.py](services/meralion_chinese_render.py),
  [processors/chinese_renderer.py](processors/chinese_renderer.py),
  [prompt/english_to_singapore-chinese.md](prompt/english_to_singapore-chinese.md)).
  The verified English answer is still what's stored in conversation context;
  only the spoken output is rendered as Singapore Chinese. Toggle with
  `ENABLE_CHINESE_RENDERER` (default `true`; set to `false` to send the
  grounded English answer directly to TTS as before).
- Text-to-speech via OpenRouter's `qwen/qwen-audio-3.0-tts-flash`
  ([services/openrouter_tts.py](services/openrouter_tts.py)). Note: this
  model's system voices only cover Mandarin Chinese and English — there is no
  Hokkien/Minnan voice, so spoken output is Mandarin-pronounced Singapore
  Chinese, not Hokkien.
- Optional `find_nearest_sso` follow-up tool
  ([processors/sso_tool.py](processors/sso_tool.py),
  [retrieval/sso_lookup.py](retrieval/sso_lookup.py)) that looks up the
  Social Service Office nearest an area the user names, from a transcribed
  MSF contact list
  ([knowledge_base/social_service_offices/](knowledge_base/social_service_offices/)).
  Toggle with `ENABLE_SSO_LOOKUP_TOOL` (default `true`). Always falls back to
  the ComCare Hotline (1800-222-0000) when the match is only approximate or
  not found.


## Architecture

Orchestrated by [Pipecat](https://github.com/pipecat-ai/pipecat), using its
browser-based WebRTC transport for interactive testing — no bespoke front-end
required.

```
transport.input()
    ↓
stt (MERaLiONSpeechTranslationService)      Hokkien audio → English text
    ↓
user_aggregator (LLMUserAggregator)         Collects transcribed text into a full user turn
    ↓
pre_llm_tagger (UserTurnPreLLMTagger)       Runs hybrid retrieval over knowledge_base/ and
    ↓                                       injects a <retrieved_evidence> block, per turn
llm (OpenAILLMService → OpenRouter gpt-oss-120b)   Grounded English answer
    ↓
chinese_renderer (ChineseRenderer)          Buffers the complete answer, renders it as
    ↓                                       Singapore Chinese via MERaLiON-3-10B
tts (OpenRouterTTSService)                   Text-to-speech (see Status above)
    ↓
transport.output()
    ↓
assistant_aggregator (LLMAssistantAggregator)
```

| Stage | Role |
|---|---|
| **MERaLiONSpeechTranslationService** | Segmented STT-shaped service; buffers VAD-closed audio, POSTs it to the MERaLiON `/v1/audio/translations` endpoint, emits English text |
| **Silero VAD** | Detects end-of-turn from the audio stream to drive turn-taking |
| **User/Assistant Aggregators** | Collect transcribed/generated text into full conversation turns |
| **Pre-LLM Tagger** | Before every LLM call: (1) rewrites a `<session_state><user_turn>` block, and (2) runs local hybrid (dense + lexical) retrieval over `knowledge_base/` and rewrites a `<retrieved_evidence>` block in the system prompt — both live-edited in place so the LLM always sees current values |
| **Local retrieval** | In-memory hybrid search: `hybrid_score = alpha * dense_cosine + (1 - alpha) * lexical_BM25`, over embeddings precomputed by `scripts/build_index.py` (`nvidia/nemotron-3-embed-1b:free` via OpenRouter) and a small dependency-free BM25 index — no vector database |
| **LLM** | `openai/gpt-oss-120b` via OpenRouter's OpenAI-compatible endpoint, prompted with [prompt/PROMPT_TEMPLATE_senior_support_grounded_answer.md](prompt/PROMPT_TEMPLATE_senior_support_grounded_answer.md) |
| **ChineseRenderer** | Buffers the complete grounded English answer (never partial fragments), calls MERaLiON-3-10B with [prompt/english_to_singapore-chinese.md](prompt/english_to_singapore-chinese.md) to render it as Singapore Chinese, and keeps the verified English text (not the rendered text) in conversation context |
| **OpenRouterTTSService** | Talks to OpenRouter's `qwen/qwen-audio-3.0-tts-flash` endpoint ([services/openrouter_tts.py](services/openrouter_tts.py)); Mandarin/English voices only |
| **find_nearest_sso** (optional tool) | LLM-callable lookup of the Social Service Office nearest an area the user names, over a transcribed MSF contact list; falls back to the ComCare Hotline when unsure ([processors/sso_tool.py](processors/sso_tool.py), [retrieval/sso_lookup.py](retrieval/sso_lookup.py)) |

An optional follow-up `search_senior_support` LLM tool exists
([processors/retrieval_tool.py](processors/retrieval_tool.py)) but is disabled
by default (`ENABLE_LLM_RETRIEVAL_TOOL=false`) — see [Limitations](#limitations)
for why.

## Repository layout

```
bots/bot.py                 Pipecat pipeline entry point
services/                   Pipecat service wrappers (MERaLiON speech translation, MERaLiON Chinese
                             renderer, OpenRouter TTS)
retrieval/                  Embedding client, BM25, hybrid SchemeIndex, SSO area-name lookup
processors/retrieval_tool.py     Evidence formatting + optional follow-up tool
processors/sso_tool.py           Optional find_nearest_sso follow-up tool
processors/chinese_renderer.py   Full-response English → Singapore Chinese buffering processor
prompt/                      System prompts (grounded answer, Chinese renderer)
knowledge_base/<category>/   Curated SupportGoWhere scheme snapshot (schemes.json + md/)
knowledge_base/social_service_offices/   MSF Social Service Office contact list (sso_locations.json + md/)
data/                        Precomputed retrieval index (embeddings.npy, schemes.json, manifest.json)
scripts/                     Offline tooling (index building, KB scraping)
scripts/samples/             Standalone request examples against the MERaLiON/OpenRouter APIs
models/                      Local model weights (gitignored; downloaded, not committed)
```

## Prerequisites

- **Python 3.10+** and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An **OpenRouter** API key ([openrouter.ai](https://openrouter.ai)) for
  `openai/gpt-oss-120b` (grounded-answer LLM), `nvidia/nemotron-3-embed-1b:free`
  (retrieval embeddings), and `qwen/qwen-audio-3.0-tts-flash` (text-to-speech)
- A **MERaLiON API key** for the speech-translation and Singapore Chinese
  rendering endpoints (`https://api.meralion.ai`)

## Setup

```bash
# 1. Copy the config and fill in your API keys
cp env.example_cpp .env

# 2. Install Python dependencies
uv sync

# 3. Build the local retrieval index over knowledge_base/
#    (a prebuilt data/ is already checked in — 303 schemes, nemotron-3-embed-1b —
#    only rerun this if knowledge_base/ changes)
uv run scripts/build_index.py
```

## Running the Bot

```bash
uv run bots/bot.py
```

Open **http://localhost:7860** in your browser and click **Connect**.

> First run may take 20–30 s while Pipecat downloads the Silero VAD weights.

### Key `.env` settings

See [env.example_cpp](env.example_cpp) for the full, commented list. The ones
you must set before first run:

```ini
OPENROUTER_API_KEY=              # gpt-oss-120b + embeddings + TTS
MERALION_API_KEY=                # speech translation + Chinese rendering
```

Everything else (retrieval `top_k`/threshold/alpha, VAD timings, system prompt
file, generation parameters, TTS voice) has a working default and can be tuned
afterwards.

---

## Data

`knowledge_base/<category>/schemes.json` + `md/` is a curated local snapshot of
SupportGoWhere scheme pages, collected by `scripts/scrape_supportgowhere_schemes.py`
against SupportGoWhere's public list/detail JSON APIs. See
[STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) section 7 for
scraping approach, provenance, and licensing notes.

## Limitations

- The Singapore Chinese rendering step adds one extra API call (and its
  latency) per assistant turn, and buffers the complete grounded English
  answer before rendering — no partial/streaming rendering yet (see
  [STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) section 5).
  If rendering fails or returns empty output, a static fallback message is
  spoken instead of fabricating content
  ([processors/chinese_renderer.py](processors/chinese_renderer.py)).
- `qwen/qwen-audio-3.0-tts-flash`'s system voices only cover Mandarin Chinese
  and English — there is no Hokkien/Minnan voice or dialect parameter exposed
  by OpenRouter for this model, so spoken output is Mandarin pronunciation,
  not genuine Hokkien (see [services/openrouter_tts.py](services/openrouter_tts.py)).
- The optional `search_senior_support` follow-up tool is off by default:
  Pipecat's CHANGELOG documents a broken-streaming-tool-call bug specific to
  `openai/gpt-oss-120b` on at least one OpenRouter upstream provider, and
  OpenRouter can route this model to different backing providers per request.
  Deterministic per-turn retrieval does not depend on this and is unaffected.
  See [processors/retrieval_tool.py](processors/retrieval_tool.py).
  See also [STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md)
  section 20 for further known limitations (Hokkien writing standardization,
  retrieval scores not being eligibility probabilities, source staleness, etc.).
