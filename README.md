# arnie-lab

A restaurant assistant that actually does things: it reads a menu, checks table
availability and books, cancels and looks up reservations against a real SQLite
database — through tool calling, streamed, across five model providers.

Built to explore what changes when an LLM stops answering questions and starts
taking actions: how tool calls arrive over a stream, what small models send
instead of what the schema asks for, what a turn really costs, and what has to
happen before a model is allowed to write to a database.

---

## Three tabs

**Chat** — the assistant. Answers stream token by token; every tool call shows
up as a collapsible bubble with the real arguments, the result and how long it
took. A model picker switches provider mid-conversation. Dish photos, spoken
replies and voice input are one toggle each.

**Telemetry** — every completed turn with tokens, cached tokens, provider cost,
latency, throughput, rounds and which tools ran. Session totals and a per-model
breakdown.

**Arena** — the same prompt against up to four models in parallel, streaming
side by side, ranked by time to first token.

---

## What running it actually taught me

These are measurements from the app, not claims from a pricing page.

**Speed and price do not move together.** One prompt, four providers, measured
in the Arena tab:

| Model | First token | Total | Output tokens | Tok/s | Cost |
|---|--:|--:|--:|--:|--:|
| GPT-OSS 120B · Groq | **0.53 s** | 0.7 s | 104 | **150** | 0.0079 ¢ |
| GPT-4.1 nano · OpenAI | 0.95 s | 1.5 s | 63 | 43 | **0.0030 ¢** |
| Gemini 3.1 Flash Lite · Google | 1.07 s | 1.3 s | 66 | 50 | 0.0108 ¢ |
| GPT-4.1 mini · OpenAI | 1.51 s | 2.0 s | 54 | 27 | 0.0106 ¢ |

Groq reached the first token three times faster than GPT-4.1 mini and produced
twice the output in a third of the time. `nano` cost a third of `mini` for a
comparable answer. In the booking conversation, Gemini spent 46% more tokens
than GPT-4.1 mini and still came out cheaper.

**Small models send the wrong types.** Llama 3.2 sends `party_size: "2"` and
`zone: ""` where the schema says integer and enum. Rejecting that is correct and
useless, so arguments are coerced against the declared schema — `"2"` → `2`,
`"4.0"` → `4`, `""` → absent, `"Salón"` → `"salon"` — and only genuinely
impossible values come back as an explanation the model can act on. It also
sometimes writes the tool call as plain text instead of using the tool channel,
with malformed JSON. That one is not worth papering over.

**Prompt caching is free money, and easy to lose.** The system prompt carries
the current date and time so the model can resolve "el viernes" into a real
date. Rebuild it every turn and the prefix changes every minute, which means it
can never be served from the provider's cache. Built once per conversation, the
second turn onwards runs about half price:

```
turn 1:  2048 in /  63 out /    0 cached · 0.0920 ¢
turn 2:  2336 in /  65 out / 2048 cached · 0.0424 ¢
```

**A prompt gap looks exactly like a bug.** Changing an existing booking, GPT-4.1
mini created a second reservation and left the first one live — two tables held
for one person. Gemini cancelled and rebooked. The fix was a sentence in the
system prompt, not code.

---

## Architecture

```
app.py                  Gradio Blocks: three tabs, event wiring, nothing else
assistant/
  config.py             business profile + model registry with declared capabilities
  db.py                 SQLite: menu, availability, reservations, opening hours
  tools.py              six tools, schemas derived from the registry, argument coercion
  tool_loop.py          streaming + tool calling + approval protocol
  llm.py                LiteLLM gateway, usage and cost, provider error mapping
  media.py              image generation with disk cache, TTS, transcription
  telemetry.py          per-turn accounting
  arena.py              parallel comparison across providers
  prompts.py            system prompts
```

The tool loop emits typed events (`TextDelta`, `ToolStarted`, `ToolFinished`,
`ApprovalRequested`, `TurnFinished`, …) rather than writing to a UI, so the same
loop drives Gradio, the CLI smoke script and the tests.

### Things the loop guarantees

- **Every `tool_call` gets exactly one `role: "tool"` reply**, including the ones
  whose arguments failed to parse. A missing reply makes the *next* request
  invalid, and the failure surfaces one turn later, somewhere else.
- **A tool failure is content, not an exception.** The model receives the error
  text and offers an alternative. Any exception a tool can raise — a bad
  signature, an image provider outage — is caught and returned as tool output.
- **Arguments arrive fragmented.** In a streamed response the id, the function
  name and the JSON arguments are split across chunks and must be reassembled by
  index before anything can run.
- **`max_rounds` bounds the loop.** Small models do get stuck calling the same
  tool forever.
- **Writes can require approval.** With the toggle on, the loop yields
  `ApprovalRequested` and pauses; the driver answers with `generator.send(True)`
  or `send(False)`. A driver that merely iterates sends `None`, which denies —
  failing closed is the only safe default for something that mutates data.

---

## Model registry

Not every model can do everything, so each one declares what it supports:

| Model | Tools | Notes |
|---|---|---|
| GPT-4.1 mini · OpenAI | yes | default |
| GPT-4.1 nano · OpenAI | yes | cheapest cloud option |
| Gemini 3.1 Flash Lite · Google | yes | free tier |
| GPT-OSS 120B · Groq | yes | fastest; free tier caps tokens per minute |
| Llama 3.2 3B · Ollama | yes | local, free, loose with types |
| DeepSeek-R1 1.5B · Ollama | **no** | local; chats but cannot look anything up |

The picker only lists models whose credentials are present, hides the local ones
when Ollama is unreachable, and warns when the selected model cannot use tools.

---

## Running it

```bash
git clone <this repo> && cd arnie-lab
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env     # add at least OPENAI_API_KEY
.venv/bin/python -m assistant.config     # what is available right now
.venv/bin/python app.py                  # http://localhost:7860
```

Optional and free: `GOOGLE_API_KEY` ([AI Studio](https://aistudio.google.com/api-keys))
and `GROQ_API_KEY` ([Groq console](https://console.groq.com/keys)). Local models
need [Ollama](https://ollama.com) with `ollama pull llama3.2`.

A scripted three-turn conversation against real providers:

```bash
.venv/bin/python scripts/smoke.py gpt-4.1-mini gemini-flash-lite groq-oss llama3.2
```

## Tests

```bash
.venv/bin/python -m pytest
```

Over a hundred tests, **no API key, no network, no cost**. The booking rules run
against a temporary SQLite database; the tool loop runs against a scripted
backend that replays canned streaming chunks, which is what makes it possible to
test fragmented arguments, parallel tool calls, chained rounds, runaway loops,
rate limits and the approval protocol without spending anything. The media tests
stub the OpenAI client, including one that asserts an off-menu dish **never
reaches the image API**.

## Cost

Image generation is the only per-call cost, and it is bounded three ways: only
dishes on the menu can be drawn, every image is cached on disk under its slug,
and `ARNIE_IMAGES=off` disables it entirely — which is what a public deployment
running on a personal key wants. Token cost comes from LiteLLM's price map, so
it is the provider's real number; when LiteLLM has no entry (local models, very
new ids) the turn is reported as `n/d` rather than counted as zero. Image, speech
and transcription calls are reported as counts, never as invented money.

## Deploying to Hugging Face Spaces

Create a Gradio Space, push this repo, and add `OPENAI_API_KEY` (plus any
others) as Space secrets. `requirements.txt` is there for Spaces, which does not
read `pyproject.toml`. Local models disappear from the picker automatically
because Ollama is not reachable there. Set `ARNIE_IMAGES=off` unless you want
visitors generating images on your key.

---

Written while working through Ed Donner's
[LLM Engineering](https://github.com/ed-donner/llm_engineering) course — the
ideas come from weeks 1 and 2, the code is my own.
