# Ka-Ki-Gong: Hokkien Voice Access to Singapore Senior Support Schemes

## Coding Agent Context and Implementation Specification

## 1. Project Summary

Build a Dockerized voice assistant prototype that helps Singapore Hokkien-speaking seniors discover and understand relevant caregiving and senior-support schemes.

The assistant accepts a spoken Hokkien question, translates it into English, retrieves evidence from a small curated knowledge base of senior-relevant SupportGoWhere scheme pages, generates a grounded English answer, renders that answer as conversational Singapore Hokkien written in Chinese characters, and synthesizes the response as Hokkien speech.

This is an accessibility layer and research prototype. It is not an official integration with SupportGoWhere, LifeSG, GovTech, MSF, MOH, AIC, or any other government agency.

### Primary user

A Singapore senior who is most comfortable speaking and listening in Hokkien and may have difficulty navigating text-heavy government support portals.

### Secondary user

A caregiver assisting a Hokkien-speaking senior.

### Prototype scope

Focus only on a corpus of schemes relevant initially sourced from:
"https://supportgowhere.life.gov.sg/"

---

## 2. Recommended Streamlined Pipeline

```text
Hokkien speech
    ↓
Pipecat transport input and voice activity detection
    ↓
MERaLiON-3B-ASR-CTM Translation API
Hokkien audio → English text
    ↓
Local retrieval over curated SupportGoWhere scheme records
    ↓
gpt-oss-120b through OpenRouter
Retrieved evidence + English user query → grounded English answer
    ↓
Full-response buffering
    ↓
MERaLiON-3-10B
Grounded English answer → Singapore Hokkien Hanji
    ↓
OpenRouter TTS (qwen/qwen-audio-3.0-tts-flash)
Chinese text → Mandarin-pronounced audio
(originally MERaLiON-OmniVoice-Hokkien-TTS; dropped for real-time latency —
see section 3.6)
    ↓
Pipecat transport output
```

### Important optimization

Use deterministic retrieval before the gpt-oss call for the initial question. This avoids using two gpt-oss calls merely to select a retrieval tool.

The default initial-turn flow should be:

```text
English query
→ local retrieval
→ one gpt-oss call with retrieved evidence
→ grounded answer
```

Function calling can remain available for follow-up turns where the LLM decides that it needs to search again.

### Expected inference calls for a normal initial question

1. One MERaLiON speech-translation call
2. One local retrieval operation, not an LLM call
3. One gpt-oss-120b call
4. One MERaLiON-3-10B Hokkien-rendering call
5. One Hokkien TTS inference call

---

## 3. Responsibilities of Each Component

### 3.1 Pipecat

Pipecat is the real-time orchestration layer. It should handle:

- Browser microphone input
- Voice activity detection and turn segmentation
- Audio frame transport
- Calling the inbound speech service
- Conversation context
- Sending grounded English text through the Hokkien renderer
- Sending Hokkien text to TTS
- Returning audio to the browser
- Latency and error-event logging

Pipecat is not the knowledge base.

### 3.2 MERaLiON-3B-ASR-CTM Translation API

Use the original Hokkien audio as input and request English text output.

Although the model name contains ASR, the API exposes a speech-translation request. The production code must therefore name the service accurately, for example:

```text
MERaLiONSpeechTranslationService
```

rather than implying that it only performs transcription.

The service should operate like a Pipecat segmented HTTP STT service:

```text
InputAudioRawFrame
→ local VAD closes the utterance
→ convert raw PCM to 16 kHz mono WAV
→ encode WAV as a base64 data URI
→ POST to MERaLiON translation endpoint
→ parse the English result
→ emit a TranscriptionFrame containing English text
```

Use asynchronous HTTP, preferably `httpx.AsyncClient`. Do not use synchronous `requests` inside the live Pipecat event loop.

### 3.3 Local retrieval

The knowledge base should be a curated local snapshot of senior-relevant SupportGoWhere scheme information.

For the MVP:

- Use 8 to 20 schemes
- Store structured JSON records
- Precompute embeddings
- Keep embeddings in memory
- Do not operate a vector database server
- Retrieve the top three records
- Apply a minimum relevance threshold
- Return source URLs and source update dates

Recommended embedding model:

```text
intfloat/multilingual-e5-small
```

Use the required prefixes:

```text
query: <user query>
passage: <scheme record>
```

Recommended retrieval method:

```text
hybrid_score = 0.7 * dense_cosine_similarity + 0.3 * lexical_score
```

The exact weights should be configurable and evaluated rather than hard-coded as an unquestioned truth.

Suggested storage:

```text
data/
  schemes.json
  embeddings.npy
  manifest.json
```

### 3.4 gpt-oss-120b through OpenRouter

Use gpt-oss-120b for:

- Conversation management
- Interpreting retrieved scheme evidence
- Producing a short, grounded English answer
- Asking a concise clarification question when the user's request is ambiguous
- Optional function calling for follow-up retrieval

Do not use gpt-oss-120b for inbound speech translation.

Do not require gpt-oss-120b to produce Hokkien unless it is being used as an experimental baseline.

For an initial scheme question, retrieval should occur before the LLM call. Supply the translated English query and top retrieved passages in one request.

Use low reasoning effort if supported because this is a latency-sensitive voice task and does not require deep reasoning.

### 3.5 MERaLiON-3-10B Hokkien renderer

Use MERaLiON-3-10B only for the narrow transformation:

```text
Verified grounded English answer
→ conversational Singapore Hokkien written in Chinese characters
```

It must not answer the user's question again, search for schemes, or change factual content.

The renderer should preserve:

- Official scheme names
- Agency names
- Amounts
- Dates
- Percentages
- Telephone numbers
- URLs
- Qualifications such as `may`, `up to`, and `subject to assessment`

Keep official terms in English where appropriate, including:

- CHAS
- AIC
- CPF
- MediSave
- MediFund
- ComCare
- CareShield Life
- SupportGoWhere
- MOH
- MSF
- HDB
- polyclinic

### 3.6 Text-to-speech: OpenRouter qwen/qwen-audio-3.0-tts-flash

MERaLiON-OmniVoice-Hokkien-TTS was the original choice for this stage — it accepts Singapore Hokkien Hanji directly (`language="nan"`) and produces genuine Hokkien-pronounced audio. It was dropped as the default because local GPU inference was too slow for real-time voice turns; the downloader and server scripts remain archived at `scripts/archive/download_meralion_tts.py` and `scripts/archive/serve_meralion_tts.py` for a future revisit.

The current default is OpenRouter's `qwen/qwen-audio-3.0-tts-flash`, a cloud endpoint with acceptable latency. Its system voices only cover Mandarin Chinese and English — there is no Hokkien/Minnan voice — so Chinese Hanji input is read aloud with Mandarin pronunciation, not genuine Hokkien (see `services/openrouter_tts.py` for the full caveat).

Expected input and output:

```text
Input: Singapore Chinese Hanji text, Mandarin pronunciation only
Output: 24 kHz PCM audio
```

---

## 4. Why the Mandarin Intermediate Step Is Not the Default

An earlier design included:

```text
Grounded English
→ gpt-oss English-to-Mandarin
→ MERaLiON Mandarin-to-Hokkien
→ TTS
```

This is not the recommended default because every translation step can introduce:

- Lost qualifications
- Altered numbers
- Incorrect eligibility claims
- Missing application steps
- More latency
- More cost

The default should be:

```text
Grounded English
→ MERaLiON-3-10B English-to-Hokkien Hanji
→ OpenRouter TTS (Mandarin-pronounced; Hokkien TTS was dropped for
  real-time latency, see section 3.6)
```

Keep the Mandarin intermediate route only as an experimental variant if evaluation shows that it materially improves Hokkien naturalness or TTS intelligibility.

---

## 5. Pipecat Pipeline Shape

Conceptually:

```python
pipeline = Pipeline([
    transport.input(),
    meralion_speech_translation,
    context_aggregator.user(),
    retrieval_and_grounded_answer_processor,
    hokkien_response_buffer_and_renderer,
    openrouter_tts,  # previously meralion_hokkien_tts; dropped for latency
    transport.output(),
    context_aggregator.assistant(),
])
```

Actual Pipecat imports and method signatures must follow the installed Pipecat version. Do not copy an outdated example without checking the installed package API.

### Audio settings

Set pipeline-wide sample rates consistently:

```text
Audio input: 16 kHz mono
Audio output: 24 kHz mono
```

Incoming speech segments must be encoded as valid WAV files, not raw PCM bytes falsely labelled as WAV.

### VAD settings

Use local VAD because the MERaLiON API consumes completed audio segments over HTTP.

Seniors may pause while speaking, so begin with a more generous end-of-turn silence threshold, approximately 800 to 1,200 milliseconds, then tune it using actual recordings.

### Full-response buffering

Do not stream partial English token fragments into the Hokkien renderer.

Incorrect:

```text
Long-Term Care
Subsidies may
help with nursing
home costs
```

Correct:

```text
gpt-oss streams or produces the complete English response
→ buffer complete sentence or response
→ call MERaLiON-3-10B once
→ emit Hokkien text
→ send to Hokkien TTS
```

For the MVP, buffer the complete response. Sentence-level streaming can be a later optimization.

---

## 6. Conversation Context

Store the translated English user turn and grounded English assistant answer in the main LLM conversation context.

```text
User context: English translation of Hokkien speech
Assistant context: Grounded English answer
```

Treat Hokkien text generation as an output-only transformation.

Do not replace the English assistant message in the reasoning context with the Hokkien Hanji output. Keeping the internal context in English makes retrieval and follow-up tool calls more predictable.

Optionally store the following as tracing metadata:

```json
{
  "source_audio_id": "...",
  "english_translation": "...",
  "retrieved_scheme_ids": ["..."],
  "grounded_answer_en": "...",
  "rendered_hokkien": "...",
  "tts_audio_id": "..."
}
```

Do not log raw audio or sensitive personal data by default.

---

## 7. Knowledge Base Preparation

### Source

Start from the SupportGoWhere caregiving-support topic and selected senior-relevant scheme pages.

The crawler should run only during dataset preparation, not during user conversations.

### Scraping approach

Because topic pages may load scheme cards dynamically, use Playwright to:

1. Open the topic page
2. Wait for rendered content
3. Scroll to trigger lazy loading
4. Collect unique links containing `/schemes/`
5. Visit each scheme page politely
6. Save the rendered HTML
7. Extract clean text
8. Convert each page to a structured record

Keep concurrency low, for example one to three requests, and add delays.

At the time inspected, the robots file contained `User-agent: *` without a `Disallow` directive. Nevertheless, robots rules are not a content licence. Review and document terms of use, attribution, provenance, and refresh limitations.

### Do not embed raw HTML

Remove:

- Navigation
- Footer
- Scripts
- Styles
- Repeated scam warnings
- Accessibility labels
- Unrelated links

### Suggested scheme schema

```json
{
  "scheme_id": "Sucth2ke",
  "slug": "long-term-care-subsidies",
  "name": "Long-Term Care Subsidies",
  "summary": "Means-tested subsidies for long-term care services.",
  "audiences": ["senior", "caregiver", "long-term-care-recipient"],
  "categories": ["healthcare", "caregiving", "long-term-care"],
  "highlights": [],
  "eligibility": [],
  "benefits": [],
  "application_steps": [],
  "help_information": [],
  "example_needs": [
    "Help paying for a nursing home",
    "Home nursing is too expensive",
    "Need community rehabilitation after discharge"
  ],
  "source_url": "https://supportgowhere.life.gov.sg/schemes/...",
  "source_updated_at": "YYYY-MM-DD",
  "retrieved_at": "ISO-8601 timestamp",
  "content_sha256": "..."
}
```

Embed one complete structured record per short scheme. Avoid unnecessary chunking unless evaluation reveals that whole-record retrieval is inadequate.

Synthetic `example_needs` should be clearly identified as author-created retrieval aids rather than source text.

---

## 8. Retrieval Function

Expose one optional LLM tool:

```text
search_senior_support
```

Suggested signature:

```python
async def search_senior_support(
    params,
    query: str,
    category: str | None = None,
):
    """Search verified Singapore senior-support scheme records.

    Args:
        query: A concise English description of the user's support need.
        category: Optional category such as healthcare, caregiving,
            mobility, financial_support, or long_term_care.
    """
```

Suggested tool result:

```json
{
  "results": [
    {
      "scheme_id": "long-term-care-subsidies",
      "title": "Long-Term Care Subsidies",
      "relevant_passage": "...",
      "application_steps": ["..."],
      "source_url": "https://supportgowhere.life.gov.sg/...",
      "source_updated_at": "YYYY-MM-DD",
      "retrieval_score": 0.89
    }
  ],
  "retrieval_confident": true
}
```

### Initial-turn policy

Automatically retrieve for the initial substantive support question. This usually requires only one gpt-oss call afterward.

### Follow-up policy

The LLM may call the search function for follow-up questions requiring new evidence. It should not call the tool for:

- Please repeat
- Speak more slowly
- Thank you
- I meant my mother, not my wife

It should ask a clarification question before searching when the request is too vague, such as:

```text
I need money.
My mother needs help.
The medical one.
```

---

## 9. Grounded Answer Prompt Requirements

The gpt-oss system prompt should enforce:

```text
You are a voice assistant helping Singapore seniors understand publicly
available support-scheme information.

Rules:
1. Use only facts supplied in the retrieved evidence.
2. Never state that the user is definitely eligible.
3. Do not infer age, citizenship, income, diagnosis, disability, household
   size, or other personal details.
4. Preserve qualifiers such as may, up to, depending on, and subject to
   assessment.
5. If the evidence is insufficient, say that the information could not be
   verified.
6. Ask one short clarification question when the user's need is ambiguous.
7. Keep the spoken answer to no more than three short sentences.
8. State what the scheme may help with.
9. Give one practical next step.
10. Avoid reading long URLs or detailed thresholds aloud unless requested.
11. Do not claim to apply for, approve, renew, or submit a scheme.
```

Suggested output schema:

```json
{
  "answer_en": "...",
  "source_ids": ["..."],
  "needs_clarification": false,
  "clarification_question_en": null,
  "abstained": false
}
```

Use structured output if the OpenRouter endpoint supports it reliably. Otherwise validate JSON with Pydantic and retry once.

---

## 10. Singapore Hokkien Rendering Prompt

Use the following behavior for MERaLiON-3-10B:

```text
You are a Singapore Hokkien language renderer for a voice-based
support-scheme assistant.

Convert the verified English answer into natural conversational Singapore
Hokkien written using Chinese characters or Hanji. The output is sent to the
text-to-speech stage (originally MERaLiON-OmniVoice-Hokkien-TTS with
language="nan"; now OpenRouter's qwen/qwen-audio-3.0-tts-flash, which reads
Chinese text with Mandarin pronunciation — see section 3.6).

Rules:
1. Preserve the original meaning exactly.
2. Translate only the supplied answer.
3. Do not answer, explain, recommend, or add facts.
4. Do not remove facts.
5. Preserve qualifying language such as may, might, up to, depending on,
   subject to assessment, and check whether.
6. Never change may qualify into qualifies.
7. Never change up to 80% into 80%.
8. Preserve all numbers, dates, percentages, monetary amounts, telephone
   numbers, URLs, and proper names exactly.
9. Keep official names in English where appropriate, including CHAS, AIC,
   CPF, MediSave, MediFund, ComCare, CareShield Life, SupportGoWhere, MOH,
   MSF, HDB, and polyclinic.
10. Use conversational Singapore Hokkien vocabulary and grammar rather than
    formal written Mandarin.
11. Chinese characters or Hanji should be used for Hokkien wording.
12. English and Malay loanwords are acceptable where natural in Singapore
    Hokkien.
13. Prefer wording understandable to an elderly Singapore Hokkien speaker.
14. Use no more than three short spoken sentences.
15. If there is no reliable Hokkien equivalent for a technical term, retain
    the original English term.
16. Return only the Hokkien text without labels, explanations, Markdown,
    romanization, or quotation marks.

Verified English answer:
{{answer_en}}
```

### Reference language patterns

Use validated examples from the Hokkien TTS model card as few-shot references. Important patterns include:

```text
今天的天气好吗？
→ 今日个天气好无？

如果下雨记得收衣服。
→ 若落雨记得收衫。

今天外面一直下雨，大家有带雨伞吗？
→ 今日外口一直落雨，大家有带雨伞无？

他突然肚子痛得很厉害，我们赶紧送他去看医生。
→ 伊突然腹肚痛真厉害，咱紧送佗去老君处。

我喜欢吃红豆冰。
→ 我 suka 食红豆冰。

大家好，吃饱了吗？今天心情好吗？
→ 大家好，食饱未？今仔日心情好无？
```

The full implementation prompt may include all model-card examples, but keep runtime token usage and latency in mind.

---

## 11. Factual Consistency Checks Before TTS

Before sending rendered Hokkien to TTS, extract protected tokens from the English answer and require them to remain in the Hokkien output.

Protected content should include:

- Agency acronyms
- Scheme names where retained in English
- Dollar amounts
- Percentages
- Years and dates
- Telephone numbers
- URLs

Example patterns:

```python
PROTECTED_PATTERNS = [
    r"\b(?:CHAS|AIC|CPF|MediSave|MediFund|ComCare|MOH|MSF|HDB)\b",
    r"\$\s?[\d,]+(?:\.\d+)?",
    r"\b\d+(?:\.\d+)?%",
    r"\b(?:19|20)\d{2}\b",
    r"\b\d{4}\s?\d{4}\b",
    r"https?://\S+",
]
```

If a protected item disappears or changes:

1. Retry rendering once with a stricter prompt
2. If it still fails, use a safe fallback
3. Do not silently speak altered factual information

A safe fallback could retain the official term in English within otherwise simple Hokkien text.

---

## 12. Error Handling

Handle the following gracefully:

### MERaLiON inbound speech errors

- Invalid API key
- Authentication failure
- Unsupported audio
- Timeout
- Empty translation
- Rate limit
- Provider 5xx
- Unexpected response schema

User-facing fallback:

```text
Sorry, I could not hear that clearly. Please try again.
```

Render this as a prevalidated Hokkien phrase rather than dynamically generating it during an outage.

### Retrieval errors

- Empty index
- No result above threshold
- Missing source text
- Corrupt embeddings

Fallback:

```text
I could not find verified information for that question. Please try another
wording or check SupportGoWhere directly.
```

### OpenRouter errors

- Missing authentication header
- Invalid key
- Insufficient credits
- Model unavailable
- Provider timeout
- Malformed JSON
- Missing `choices`

Every OpenRouter request must include:

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
```

Do not assume a successful response. Check the HTTP status and validate the response structure before accessing `choices[0].message`.

### Hokkien rendering or TTS errors

- Empty output
- Mandarin-like output
- Protected content changed
- TTS generation failure
- Invalid audio output

Fallback options:

- Retry once
- Use a validated static Hokkien error message
- Display the grounded English and Chinese text visually
- Never fabricate an audio response

---

## 13. Environment Variables and Secrets

Suggested `.env.example`:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-oss-120b
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

MERALION_API_KEY=
MERALION_API_BASE_URL=https://api.meralion.ai
MERALION_TRANSLATION_MODEL=MERaLiON/MERaLiON-3-3B-ASR-CTM
MERALION_RENDER_MODEL=MERaLiON/MERaLiON-3-10B

TTS_MODEL=qwen/qwen-audio-3.0-tts-flash
# Originally MERaLiON/MERaLiON-OmniVoice-Hokkien-TTS; dropped for real-time
# latency in favor of this OpenRouter endpoint (see section 3.6).

RETRIEVAL_TOP_K=3
RETRIEVAL_MIN_SCORE=0.0
LOG_LEVEL=INFO
```

Never commit `.env` or actual API keys.

Suggested `.gitignore` entries:

```gitignore
.env
*.wav
__pycache__/
.pytest_cache/
.venv/
```

If demo audio should be committed, place only consented or synthetic test fixtures in a dedicated fixture directory and override the blanket WAV exclusion as needed.

---

## 14. Docker and Reproducibility

The Dockerized solution should contain:

- Pipecat application
- Web application or transport server
- Local retrieval code
- Curated scheme JSON
- Precomputed embedding matrix
- Embedding model dependency or a documented build-time download
- Evaluation scripts
- Tests

External services:

- MERaLiON speech translation API
- OpenRouter gpt-oss-120b API
- MERaLiON-3-10B API
- Hokkien TTS, whether remote or separately hosted

Do not package large MERaLiON or gpt-oss weights directly into the main application image.

The README must explain clearly that valid API keys are required for the full live pipeline.

Provide a deterministic smoke-test or mock mode so reviewers can verify application wiring without paid credentials:

```text
APP_MODE=mock
```

Mock mode may use:

- A bundled test audio clip
- Cached inbound English translation
- Real local retrieval
- Cached or deterministic grounded response
- Cached Hokkien rendering
- Cached TTS audio

Clearly label all cached outputs. Do not present mock-mode outputs as live inference.

---

## 15. Evaluation Plan

The assessment emphasizes evaluation. Evaluate each stage separately and then end to end.

### 15.1 Inbound speech translation

Prepare 15 to 25 recorded Hokkien queries relevant to senior support.

Include:

- Nursing-home costs
- Home care
- Mobility limitations
- Medical expenses
- CHAS
- MediSave
- Caregiver support
- Living alone
- Utility or daily expenses
- Ambiguous requests
- Hokkien mixed with English scheme names

Metrics:

- Support-intent accuracy
- Explicit-fact preservation
- Relationship accuracy, for example mother versus father
- Care-setting accuracy, for example home care versus nursing home
- Negation accuracy
- Invented-fact rate
- End-of-speech to translation latency

Do not evaluate only translation fluency.

### 15.2 Retrieval

Create a ground-truth mapping from each test query to one or more relevant schemes.

Metrics:

- Recall@1
- Recall@3
- Mean reciprocal rank if useful
- No-result or abstention accuracy
- Retrieval latency

Ablation:

```text
Lexical only
versus
Dense only
versus
Hybrid retrieval
```

### 15.3 Grounded answer generation

Evaluate:

- Correct scheme
- Answer supported by retrieved evidence
- No invented facts
- Qualification preservation
- Useful next action
- No unsupported eligibility claim
- Appropriate clarification
- Appropriate abstention
- Response length suitable for speech
- Latency and cost

### 15.4 Hokkien rendering and TTS

Compare at least:

```text
A. Standard Chinese directly into Hokkien TTS
B. English → MERaLiON-3-10B Hokkien Hanji → TTS
C. Optional: English → Mandarin → Hokkien Hanji → TTS
```

Use a fluent Singapore Hokkien speaker if available.

Score:

- Meaning preservation
- Singapore Hokkien naturalness
- Senior comprehensibility
- Mandarin leakage
- Preservation of official names and numbers
- TTS intelligibility
- Audio naturalness
- Latency

### 15.5 End-to-end task success

A task succeeds when the user:

1. Is understood correctly
2. Receives a relevant scheme or an appropriate clarification
3. Gets an answer supported by source evidence
4. Receives one practical next step
5. Can understand the synthesized Hokkien response

Report honest limitations and failure examples.

---

## 16. Guardrails and Product Boundaries

The assistant must:

- Provide information, not official eligibility decisions
- Cite or display source URLs
- Show source update dates where available
- Avoid collecting NRIC, Singpass credentials, bank credentials, or detailed medical records
- Avoid submitting applications
- Avoid promising approvals
- Ask short clarification questions
- Abstain when evidence is insufficient
- Keep answers short for voice delivery
- Preserve uncertainty and conditions

The UI should state:

```text
This prototype provides general information from a dated snapshot of public
support-scheme pages. It does not confirm eligibility or submit applications.
Please verify important details with the official source or responsible agency.
```

---

## 17. Suggested Repository Structure

```text
ka-ki-gong/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── app/
│   ├── main.py
│   ├── config.py
│   ├── pipeline.py
│   ├── transports/
│   ├── services/
│   │   ├── meralion_translation.py
│   │   ├── openrouter_llm.py
│   │   ├── meralion_hokkien_renderer.py
│   │   └── meralion_hokkien_tts.py
│   ├── retrieval/
│   │   ├── index.py
│   │   ├── hybrid_search.py
│   │   └── schemas.py
│   ├── processors/
│   │   ├── grounded_answer.py
│   │   └── hokkien_response_buffer.py
│   └── prompts/
│       ├── grounded_answer.txt
│       └── hokkien_renderer.txt
├── data/
│   ├── schemes.json
│   ├── embeddings.npy
│   └── manifest.json
├── scripts/
│   ├── scrape_supportgowhere.py
│   ├── structure_schemes.py
│   ├── build_index.py
│   ├── test_meralion.py
│   └── test_openrouter.py
├── evaluation/
│   ├── cases.jsonl
│   ├── evaluate_translation.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_grounding.py
│   └── results/
└── tests/
    ├── test_retrieval.py
    ├── test_response_parsing.py
    ├── test_protected_terms.py
    └── test_pipeline_smoke.py
```

The coding agent may simplify this structure if appropriate, but it should preserve clear separation among provider clients, retrieval, prompts, orchestration, and evaluation.

---

## 18. Implementation Priorities for a Two-Day Build

### Must have

1. Working microphone or test-audio input
2. Hokkien-audio-to-English call
3. Curated local scheme corpus
4. Working retrieval
5. Grounded English answer
6. English-to-Hokkien Hanji rendering
7. Hokkien TTS playback
8. Source display
9. Graceful error handling
10. Docker instructions
11. Small evaluation dataset and metrics

### Nice to have

- Original Hokkien transcript displayed alongside English translation
- Function calling on follow-up turns
- Sentence-level output streaming
- User correction of translated text
- Multi-turn profile collection
- Additional support topics

### Do not build in this prototype

- Singpass integration
- Automatic application submission
- Appointment booking
- Calendar reminders
- WhatsApp or email sending
- Eligibility approval logic
- Full SupportGoWhere replication
- Multi-agent framework without demonstrated need
- Production scraping during live conversations

---

## 19. Key Technical Decisions to Explain in README

1. **Narrow senior-support corpus:** improves relevance and makes evaluation feasible.
2. **MERaLiON inbound translation:** selected for Singapore Hokkien and code-switching support.
3. **Retrieval-first initial turn:** reduces gpt-oss calls, latency, and cost.
4. **gpt-oss-120b via OpenRouter:** selected for tool use, instruction following, structured output, and grounded answer generation. Acknowledge that it is oversized for the narrow task and compare against operational requirements.
5. **MERaLiON-3-10B renderer:** used only for Singapore Hokkien Hanji generation.
6. **Dedicated Hokkien TTS:** generates spoken Singapore Hokkien from Hokkien Hanji.
7. **No vector database:** unnecessary for a tiny corpus; NumPy-based in-memory retrieval is simpler and reproducible.
8. **Source snapshot:** faster and safer than runtime scraping, but introduces staleness risk.
9. **English internal context:** improves retrieval and follow-up consistency.
10. **Full-response buffering:** chosen for translation fidelity, with latency as the trade-off.

---

## 20. Known Limitations

- Hokkien writing is not fully standardized.
- Singapore Hokkien differs from Taiwanese Hokkien.
- Speech translation may sound fluent while dropping or inventing facts.
- Hokkien rendering may introduce Mandarin-like grammar.
- Government information can become outdated.
- Retrieval scores are not eligibility probabilities.
- A small corpus cannot answer every support question.
- External API availability, cost, and latency affect the experience.
- The assistant cannot verify identity, income, citizenship, or eligibility.
- Hokkien TTS quality must be evaluated on synthesized audio, not text alone.
- The prototype depends on review by fluent Singapore Hokkien speakers for a rigorous language-quality claim.

---

## 21. Definition of Done

The prototype is complete when a reviewer can:

1. Follow the README on a clean machine
2. Supply API keys through environment variables
3. Start the application using Docker
4. Speak or submit a prerecorded Hokkien query
5. See the English translation
6. See the retrieved scheme and official source link
7. Receive a short grounded English answer
8. See the rendered Hokkien Hanji
9. Hear the answer synthesized in Hokkien
10. Run the evaluation scripts
11. Inspect clear limitations and failure cases
12. Run a mock smoke test without live credentials

---

## 22. Immediate Coding-Agent Instructions

Implement the smallest vertical slice first:

```text
prerecorded Hokkien WAV
→ MERaLiON translation API
→ local retrieval
→ OpenRouter grounded answer
→ MERaLiON-3-10B Hokkien rendering
→ Hokkien TTS
→ save output WAV and trace JSON
```

Only after this works should the implementation be inserted into a live Pipecat microphone pipeline.

For every provider call:

- Use async HTTP in the live application
- Set explicit timeouts
- Check HTTP status
- Validate JSON structure
- Log latency without logging secrets
- Retry at most once for transient failures
- Return a safe user-facing fallback

Do not add features that are not required for the vertical slice until the end-to-end path and tests are working.
