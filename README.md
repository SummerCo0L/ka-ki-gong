# Ka-Ki-Gong: Hokkien Voice Access to Singapore Senior Support Schemes

Ka-Ki-Gong ("speak for yourself" in Hokkien) is a voice assistant prototype that
helps Singapore dialect-speaking seniors (e.g., Hokkien) seek help and the relevant resources  and schemes through a voice assistant. A spoken dialect question (e.g., Hokkien) is translated to English, and answered using a curated, locally-retrieved snapshot of [SupportGoWhere](https://supportgowhere.life.gov.sg/) scheme pages, and spoken back in Singapore Chinese through a cloud text-to-speech model.

This is an accessibility research prototype, not an official integration with
SupportGoWhere, LifeSG, GovTech, MSF, MOH, AIC, or any other government agency.
It does not confirm eligibility or submit applications. See
[STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) for the full
design specification this implementation follows.

## Motivation

- Some Singapore seniors who need government support cannot read or speak
  English fluently, which locks them out of text-heavy, English-first
  portals even when a scheme that could help them already exists.
- Rather than building a new benefits database from scratch, this project
  taps on [SupportGoWhere](https://supportgowhere.life.gov.sg/)'s own existing
  scheme search/content, but refines the interaction to be palatable for a
  senior who can't read or speak English: a spoken dialect (e.g. Hokkien)
  question in, a spoken Singapore Chinese answer out.
- The goal is deliberately narrow: **not** to complete an application or
  relay every clause of a scheme through the assistant, but to point the
  caller to the relevant hotline, address, or scheme name to pursue next and
  let them (or a caregiver) take it from there — see Guardrails/Limitations
  for why the assistant never confirms eligibility or submits anything on
  the user's behalf.

## Status

**Implemented and wired into `bots/bot.py`:**

- Hokkien (or other supported language & dialect) speech → English text, via the
  MERaLiON speech-translation API ([services/meralion_speech_translation.py](services/meralion_speech_translation.py))
- Local hybrid retrieval (dense + lexical) over 303 curated SupportGoWhere
  scheme records in [knowledge_base/](knowledge_base/), run deterministically
  before every turn ([retrieval/](retrieval/), [processors/retrieval_tool.py](processors/retrieval_tool.py)),
  with `cohere/rerank-4-fast` cross-encoder reranking pass over the
  top hybrid candidates before the top-k are kept (`RERANKER_MODEL`; see
  [eval/README.md](eval/README.md) for its measured impact)
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
pre_llm_tagger (UserTurnPreLLMTagger)       Runs hybrid retrieval (+ rerank) over
    ↓                                       knowledge_base/ and injects a <retrieved_evidence> block
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
| **Pre-LLM Tagger** | Before every LLM call: (1) rewrites a `<session_state><user_turn>` block, and (2) runs local hybrid (dense + lexical) retrieval (+ optional reranking) over `knowledge_base/` and rewrites a `<retrieved_evidence>` block in the system prompt — both live-edited in place so the LLM always sees current values |
| **Local retrieval** | In-memory hybrid search: `hybrid_score = alpha * dense_cosine + (1 - alpha) * lexical_BM25`, over embeddings precomputed by `scripts/build_index.py` (`openai/text-embedding-3-small` via OpenRouter, `OPENROUTER_EMBEDDING_MODEL`) and a small dependency-free BM25 index — no vector database |
| **Reranker (optional)** | `cohere/rerank-4-fast` cross-encoder over the top `RERANK_CANDIDATES` hybrid results, keeping the top `RETRIEVAL_TOP_K`; falls back to plain hybrid ranking if the rerank call fails (`retrieval/reranker.py`) |
| **LLM** | `openai/gpt-oss-120b` via OpenRouter's OpenAI-compatible endpoint, prompted with [prompt/PROMPT_TEMPLATE_senior_support_grounded_answer.md](prompt/PROMPT_TEMPLATE_senior_support_grounded_answer.md) |
| **ChineseRenderer** | Buffers the complete grounded English answer (never partial fragments), calls MERaLiON-3-10B with [prompt/english_to_singapore-chinese.md](prompt/english_to_singapore-chinese.md) to render it as Singapore Chinese, and keeps the verified English text (not the rendered text) in conversation context |
| **OpenRouterTTSService** | Talks to OpenRouter's `qwen/qwen-audio-3.0-tts-flash` endpoint ([services/openrouter_tts.py](services/openrouter_tts.py)); Mandarin/English voices only |
| **find_nearest_sso** (optional tool) | LLM-callable lookup of the Social Service Office nearest an area the user names, over a transcribed MSF contact list; falls back to the ComCare Hotline when unsure ([processors/sso_tool.py](processors/sso_tool.py), [retrieval/sso_lookup.py](retrieval/sso_lookup.py)) |

## Repository layout

```
bots/bot.py                 Pipecat pipeline entry point
Dockerfile, docker-compose.yml, .dockerignore   Container build (see Running the Bot)
services/                   Pipecat service wrappers (MERaLiON speech translation, MERaLiON Chinese
                             renderer, OpenRouter TTS)
retrieval/                  Embedding client, BM25, hybrid SchemeIndex, SSO area-name lookup
processors/retrieval_tool.py     Evidence formatting for retrieved schemes
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
  — or just [Docker](https://docs.docker.com/get-docker/) if you'd rather run
  the container (see Running the Bot); Python/uv are only needed to run
  `scripts/build_index.py`/`scripts/scrape_supportgowhere_schemes.py`
  locally, since a prebuilt `data/` and `knowledge_base/` are already checked in
- An **OpenRouter** API key: sign up for free at [openrouter.ai](https://openrouter.ai),
  then create a key under [Settings → Keys](https://openrouter.ai/settings/keys).
  Used for `openai/gpt-oss-120b` (grounded-answer LLM), `openai/text-embedding-3-small`
  (retrieval embeddings), `cohere/rerank-4-fast` (optional reranking), and
  `qwen/qwen-audio-3.0-tts-flash` (text-to-speech) — none of these four have
  a free tier, so **because this project only uses paid models, top up at
  least $2** at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits)
  (card/AliPay/crypto; a small processing fee applies) before testing
  end-to-end — per the per-turn cost breakdown in Target User, Environment &
  Deployment Considerations below, that covers on the order of a couple
  hundred voice turns plus running `scripts/build_index.py`/
  `scripts/evaluate_retrieval.py`.
- A **MERaLiON API key** for the speech-translation and Singapore Chinese
  rendering endpoints (`https://api.meralion.ai`). Register a free key at
  [meralion.org/api-console](https://meralion.org/api-console) ("My Key" tab
  → generate a key; usage is shown under "API Tiers"/"Usage"). The free tier
  is rate-limited to **5 requests/minute and 1,000 requests/month** (as shown
  on the API Tiers tab at the time of writing — check the console for the
  current limits). Review [meralion.org/terms](https://meralion.org/terms)
  before use — see Licensing below.

## Setup

```bash
# 1. Copy the config and fill in your API keys (see Prerequisites above for
#    how to obtain each). OPENROUTER_API_KEY must be set before step 4, since
#    build_index.py calls OpenRouter's embedding API.
cp env.example_cpp .env

# 2. Install Python dependencies
uv sync

# 3. (Optional — a curated knowledge_base/ is already checked in) re-scrape
#    SupportGoWhere to refresh the knowledge base — no API key needed:
uv run scripts/scrape_supportgowhere_schemes.py

# 4. Build the local retrieval index over knowledge_base/ — embeds every
#    scheme via OpenRouter (requires OPENROUTER_API_KEY from step 1).
#    (a prebuilt data/ is already checked in — 303 schemes, text-embedding-3-small —
#    only rerun this if knowledge_base/ changed, e.g. after step 3)
uv run scripts/build_index.py
```

## Running the Bot

### Option A — locally with uv

```bash
uv run bots/bot.py
```

Open **http://localhost:7860** in your browser and click **Connect**.

### Option B — Docker

```bash
# Build and run (reads OPENROUTER_API_KEY / MERALION_API_KEY from .env)
docker compose up --build
```

or without Compose:

```bash
docker build -t ka-ki-gong .
docker run --rm -p 7860:7860 --env-file .env ka-ki-gong
```

Open **http://localhost:7860** in your browser and click **Connect**. The
image bakes in the prebuilt `data/` retrieval index and `knowledge_base/`
snapshot already checked into this repo — only the two API keys need to be
supplied at run time, via `.env`/`--env-file`, never baked into the image.
See [Dockerfile](Dockerfile) for what's included; the archived local
Hokkien-TTS downloader/server (`scripts/archive/`, torch/omnivoice) is
deliberately excluded from the image — see Development Process below.

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

## Evaluation

### Why retrieval, and why benchmarking

The pipeline has several components that could each be evaluated in
isolation — speech translation, Chinese rendering, TTS — but this project's
evaluation effort was deliberately concentrated on **retrieval**, not the
others. STT/translation and TTS/rendering are comparatively direct,
single-input-to-single-output transformations; retrieval is the stage that
decides which of the 300+ real scheme records the grounded answer is even
allowed to draw from, so a retrieval miss produces a confidently-wrong
answer no matter how well the LLM writes it up — it's the most
failure-prone stage upstream of every answer, and the one worth spending
the evaluation budget on.

**Benchmarking** (Recall@k / MRR against a labeled, self-curated
ground-truth set) was chosen over **LLM-as-judge** as the primary method
because retrieval has an intuitive, checkable correct answer: for a given
query, one or more specific schemes in the knowledge base are the right
reference, and the system either surfaced them or it didn't. An LLM judge
can score how plausible or well-written a retrieved result *looks*, but it
has no independent way to confirm that result is genuinely the correct
source — doing that honestly would mean handing it the full knowledge base
as context on every judgment call (impractical at 300+ scheme records per
turn), and at that point it would just be re-running retrieval itself
rather than checking it. A fixed labeled ground truth sidesteps this: it's
set once, and every run is scored against it the same way, objectively and
reproducibly, so hybrid-search weights, embedding models, and a reranking
stage can all be compared on the same fixed set as the knowledge base grows.
LLM-as-judge is still used, but only as a supplementary signal on top of
the labeled results (see below and Development Process), not as a
substitute for ground truth.

See [eval/README.md](eval/README.md)
for the full methodology: how the 133-query, 179-scheme-id set was built,
then corrected using an LLM-assisted audit of a real retrieval run's results
— and why fixing ground truth *before* tuning the reranker matters for
reading metric changes honestly.

The query set itself was AI-generated, then refined across iterations rather
than trusted as-is: schemes were first filtered to only the 167 tagged
`"Seniors"` in `profiles` (the bot's actual target population), Claude Sonnet
5 wrote one or more casual, indirect queries per scheme simulating how a
senior might actually ask, deliberately ambiguous ones were kept
multi-labeled instead of forced to a single answer, and near-duplicate
queries were consolidated down to 133. Crucially, the ground truth was
corrected *after* a first real evaluation run, not just re-read in
isolation: an LLM was shown each query's actual top-5 retrieved results
next to the schemes' real eligibility text and asked to judge whether an
unlabeled result was genuinely also correct — 62 of the 133 queries' labels
were expanded this way, applied programmatically rather than hand-edited.

```bash
uv run scripts/evaluate_retrieval.py                        # hybrid + reranking (current defaults)
uv run scripts/evaluate_retrieval.py --reranker-model none  # hybrid search only, no reranking
uv run scripts/evaluate_retrieval.py --llm-judge            # add supplementary LLM-as-judge scoring
```

Latest measured result (`eval/results/retrieval_eval_20260906T100946Z.json`):
**Recall@1 93.2%, Recall@3/5/10 97.7%, MRR 0.955**. Generation quality
(groundedness, no invented facts) and Hokkien-rendering quality are not yet
covered by an automated eval harness — see Limitations.

## Data

`knowledge_base/<category>/schemes.json` + `md/` is a curated local snapshot of
SupportGoWhere scheme pages, collected by `scripts/scrape_supportgowhere_schemes.py`
against SupportGoWhere's public, unauthenticated list/detail JSON APIs (a
Singapore Government service). See
[STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) section 7
for the full scraping approach.

- **Provenance & licensing**: all scraped content is public government
  information already published for citizen consumption. At the time of
  scraping, SupportGoWhere's `robots.txt` allowed `User-agent: *` with no
  `Disallow`; robots permission is not itself a content licence, so this
  snapshot is used only for a non-commercial accessibility research
  prototype, and every scheme record keeps `source_url`/`source_updated_at`
  so an answer can be traced back to the official page.
- **Privacy**: scheme records describe government programmes, not
  individuals — no personal or citizen data is scraped or stored. The
  `knowledge_base/social_service_offices/sso_locations.json` contact list is
  similarly institutional (office addresses/phone numbers). The pipeline does
  not log raw audio or user-identifying information by default.
- **Synthetic content**: each scheme record's `example_needs` field (a few
  natural-language phrasings of who might need the scheme, used as a
  retrieval aid) is author/AI-generated, not scraped source text, and is kept
  in a clearly separate field from the scraped content.
- **Staleness**: the corpus is a one-time snapshot, not a live feed — see
  Limitations below.
- The retrieval evaluation set ([eval/](eval/)) is a separate, self-curated,
  AI-assisted synthetic dataset — see [eval/README.md](eval/README.md) for
  how its queries and ground-truth labels were constructed and audited.

## Licensing & Terms of Service

- **SupportGoWhere scraping** (`scripts/scrape_supportgowhere_schemes.py`):
  [supportgowhere.life.gov.sg/robots.txt](https://supportgowhere.life.gov.sg/robots.txt)
  contains only `User-agent: *` with no `Disallow` line — scraping the public
  list/detail JSON APIs is consistent with that file. Robots permission is
  not a content licence, so this snapshot is used only for a non-commercial
  accessibility research prototype (see Data above for provenance details).
- **MERaLiON API** ([meralion.org/terms](https://meralion.org/terms)): the
  API is provided "as is"/"as available" by A*STAR I²R (IMDA-funded), with no
  warranty on output accuracy, no guaranteed availability, and Singapore law
  governing use. Callers are responsible for the lawfulness of any audio/text
  submitted (consent, data protection) — this prototype does not submit any
  user-identifying data beyond the audio needed for that turn's translation
  or rendering call (see Data → Privacy above), and does not treat MERaLiON
  outputs as verified facts without the retrieval-grounding step.
- **OpenRouter-hosted models** (all also subject to
  [openrouter.ai/terms](https://openrouter.ai/terms)):
  - `openai/gpt-oss-120b` is [Apache-2.0-licensed, open-weight](https://huggingface.co/openai/gpt-oss-120b)
    — accessed here via OpenRouter's hosted API, not self-hosted or
    redistributed.
  - `openai/text-embedding-3-small`, `cohere/rerank-4-fast`, and
    `qwen/qwen-audio-3.0-tts-flash` are proprietary, provider-hosted models
    (OpenAI, Cohere, and Alibaba/Qwen respectively) with no published open
    weights; usage is governed by each provider's own API terms as
    surfaced through OpenRouter, not a redistributable licence.

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
- See [STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md)
  section 20 for further known limitations (Hokkien writing standardization,
  retrieval scores not being eligibility probabilities, source staleness, etc.).
- No generation- or rendering-quality eval harness yet (see Evaluation above)
  — only retrieval is benchmarked with labeled metrics today.
- The Docker image only packages the WebRTC demo path; all AI inference still
  depends on the external OpenRouter and MERaLiON APIs being reachable and
  keyed correctly at container startup (see Running the Bot).

---

## Target User, Environment & Deployment Considerations

**Target user**: Singapore-resident Hokkien-speaking seniors (and caregivers
assisting them) who find text-heavy portals like SupportGoWhere hard to
navigate. **Environment**: a browser-based WebRTC demo (Pipecat's built-in
transport), run locally or on a single small server — this is a research
prototype, not a hardened multi-tenant deployment; there is no
authentication, session-isolation hardening, or PDPA-compliant data-retention
pipeline implemented.

**Projected inference cost** (OpenRouter list prices, checked Sep 2026):
`gpt-oss-120b` ≈ $0.09 / $0.37 per million input/output tokens (weighted
average across providers), `text-embedding-3-small` $0.02/M tokens,
`cohere/rerank-4-fast` $0.002/search, `qwen-audio-3.0-tts-flash` $15/M
characters. Given the retrieval-first design (one embed + one optional
rerank + one LLM call + one TTS call per turn), the OpenRouter-metered
portion of an average voice turn costs well under US$0.01. The two MERaLiON
calls (speech translation, Chinese rendering) run on a separately-hosted
research API whose per-call pricing isn't published, so total cost-at-scale
isn't fully known.

**Monitoring metrics post-deployment**: retrieval Recall@1/@3/MRR, re-measured
by re-running `scripts/evaluate_retrieval.py` against an expanding query set
as real usage surfaces new phrasings (see Evaluation); end-to-end turn
latency per pipeline stage; the TTS/Chinese-rendering fallback rate (how
often the static fallback message is spoken instead of a real rendering).

**Deployment risk**: knowledge-base staleness — schemes are a dated snapshot
scraped once, not a live feed, so eligibility criteria, amounts, or contact
details can silently go out of date with no automatic refresh or staleness
alert currently in place.

## Development Process

This prototype was built iteratively, with several earlier designs discarded
once evaluated — see [STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md)
for the full spec this implementation follows:

- **TTS backend switch**: MERaLiON-OmniVoice-Hokkien-TTS (genuine Hokkien
  pronunciation via local GPU inference) was the original choice — the
  archived downloader/server scripts remain at `scripts/archive/`. It was
  dropped in favor of OpenRouter's `qwen/qwen-audio-3.0-tts-flash` once local
  inference proved too slow for real-time voice turns, trading genuine
  Hokkien pronunciation for Mandarin-pronounced Singapore Chinese (called out
  throughout this README and in section 3.6 of the pipeline spec).
- **Retrieval-first, not tool-call-first**: an earlier design exposed
  retrieval as an LLM-callable tool. It was replaced with deterministic
  per-turn retrieval (always runs before the LLM call) after the similarly
  LLM-callable `find_nearest_sso` tool proved to intermittently fail to
  stream correctly through OpenRouter/gpt-oss-120b — removing that same
  failure mode from the more critical retrieval step.
- **Embedding model swap**: retrieval initially defaulted to the free
  `nvidia/nemotron-3-embed-1b:free`, but its 50-requests/day free-tier cap
  made repeated index rebuilds and evaluation runs impractical; the default
  was switched to the paid `openai/text-embedding-3-small` ($0.02/M tokens).
- **Ground-truth-first evaluation**: the retrieval eval set's labels were
  corrected using an LLM-assisted audit of a real run's results *before*
  tuning a reranking stage — see [eval/README.md](eval/README.md) for why
  that ordering matters (a jump in Recall@k can be an artifact of previously
  incomplete ground truth, not a genuine retrieval improvement). Reranking
  (`cohere/rerank-4-fast`) was added last, once hybrid search was measured
  against corrected labels, so its contribution could be attributed cleanly.
- **LLM-as-judge discarded as the primary eval method**: an LLM-scored
  relevance judgment was tried, but labeled Recall@k/MRR against the
  hand-corrected ground truth set was kept as the primary metric instead —
  a fixed, reproducible ground truth is easier to trust and compare across
  runs than another LLM's subjective (and run-to-run inconsistent) score.
  LLM-as-judge remains available only as an optional supplementary signal
  (`--llm-judge` in `scripts/evaluate_retrieval.py`), not the headline result.
- **Dependency trimming for Docker**: `torch`/`torchaudio`/`omnivoice`/
  `soundfile`/`huggingface_hub` were moved out of the base `uv sync` install
  into an opt-in `archive-hokkien-tts` extra (see [pyproject.toml](pyproject.toml))
  once Docker packaging was added, since none of them are imported by the
  live pipeline — only by the archived local Hokkien-TTS downloader/server.
  This keeps the image small and avoids needing the `pytorch-cu126` wheel
  index inside the container at all, matching the pipeline spec's guidance
  against packaging large model weights into the main application image
  ([STREAMLINED_PIPELINE_CONTEXT.md](STREAMLINED_PIPELINE_CONTEXT.md) section 14).

## Using AI Coding Agents

This project was built with GitHub Copilot (Claude Sonnet 5) as the primary
coding agent, used across pipeline implementation, retrieval/evaluation
tooling, debugging, and documentation.

> TODO before submission: export and commit the Copilot chat/session logs
> for this project (see `session_store_sql`/chronicle tooling) to a location
> in this repo, and link it here.

## Demo Video

> TODO before submission: add the 3–5 minute demo video URL here.
