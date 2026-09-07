# llm-tool-calling-lab

A restaurant assistant that actually does things: it reads a menu, checks table
availability and books, cancels and looks up reservations against a real SQLite
database — through tool calling, streamed, across five model providers.

Built to explore what changes when an LLM stops answering questions and starts
taking actions: how tool calls arrive over a stream, what small models send
instead of what the schema asks for, what a turn really costs, and what has to
happen before a model is allowed to write to a database.

---

## Four tabs

**Chat** — the assistant. Answers stream token by token; every tool call shows
up as a collapsible bubble with the real arguments, the result and how long it
took. A model picker switches provider mid-conversation, and a second one sets
where a turn escalates when the first model is out of its depth. Dish photos,
spoken replies and voice input are one toggle each.

**Telemetry** — every completed turn with tokens, cached tokens, provider cost,
latency, throughput, rounds, which tools ran, and the route it took if it
changed models mid-turn. Session totals, a per-model breakdown, and how often
the routing actually fired.

**Arena** — the same prompt against up to four models in parallel, streaming
side by side, ranked by time to first token.

**Bench** — the same *conversation* under three routing policies, scored
against the database rather than against what the assistant said. The table
fills in a row at a time as each policy plays through; the verdict only appears
once every arm has finished.

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
  tool_loop.py          streaming + tool calling + approval protocol + escalation
  routing.py            which model finishes a turn, decided on evidence
  llm.py                LiteLLM gateway, usage and cost, provider error mapping
  media.py              image generation with disk cache, TTS, transcription
  telemetry.py          per-turn accounting
  arena.py              parallel comparison across providers
  bench.py              the same conversation under three routing policies
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
- **A turn can change hands mid-flight.** The draft model starts it, a stronger
  one finishes it when the draft gives evidence it should not — see below.
- **Writes can require approval.** With the toggle on, the loop yields
  `ApprovalRequested` and pauses; the driver answers with `generator.send(True)`
  or `send(False)`. A driver that merely iterates sends `None`, which denies —
  failing closed is the only safe default for something that mutates data.

---

## Picking a model per turn, without a model to pick it

The usual way to route between a cheap model and an expensive one is to put a
classifier in front: ask a small model whether the request is simple, then
dispatch. Two things make that a bad trade here.

**It pays latency on every turn.** The classification hop is serial, so it
lands in front of the first token — the number the Arena tab exists to measure
— to save a fraction of a cent.

**And it guesses.** What makes a turn hard in this app is not the surface of
the text. `"sí, dale"` is three words and a write to the database. `"cambiámela
para las 22"` needs a cancel *and* a rebook, which is the case GPT-4.1 mini was
already caught getting wrong. Classifying either correctly requires the
conversation state, so the classifier is neither small nor reliable — and a
misroute here does not fail loudly, it books a second table.

So the decision runs the other way around. **Every turn starts on the draft
model and changes hands the moment the draft produces evidence it should not be
finishing this one.** The evidence is a tool call that already came back over
the stream — a real name with real arguments — not a prediction:

| Trigger | What it catches |
|---|---|
| `write` | The draft asked for `make_reservation` or `cancel_reservation`. |
| `bad_arguments` | Arguments that do not parse, or that no coercion can rescue. |
| `repeat` | A call it already ran this turn — a stuck model, one round early. |
| `runaway` | It burned its round budget. The backstop for when `repeat` misses. |

A menu question never triggers anything, so the common case costs nothing
extra. The check runs *before* the round is committed to the transcript, which
matters more than it looks: a discarded round that left its `tool_calls` behind
would invalidate the very next request.

The honest cost: the round that triggered the hand-off is thrown away. Its
tokens were spent and the Telemetry tab counts them against the draft, where
they belong. A turn escalates at most once, so the worst case is bounded.

What this does **not** claim is a saving. What a turn would have cost had it
stayed on the draft is a counterfactual, and this repo does not invent numbers
— the escalation rate and the trigger breakdown are in the Telemetry tab, and
the comparison that settles it is running the same conversation under both
policies.

Escalation composes with the confirmation toggle rather than replacing it:
routing decides *who* runs the write, the approval pause decides *whether* it
runs at all — now with the stronger model's arguments to look at.

## Does the routing actually pay? The bench

An argument is worth what the measurement behind it is worth, so the routing
gets one. `assistant/bench.py` plays the same conversation under three
policies — always the strong model, always the draft, and routed — and compares
them.

The design decision that makes it worth having: **it scores the database, not
the answer.** A bench that weighed only money and speed would crown the
cheapest model every time, which is the wrong answer for exactly the reason
this repo already documented by hand — a model that replies *"listo, ya te la
cambié"* and leaves both tables booked has failed, however good the sentence
was. So each scenario declares the rows that have to exist when the
conversation ends, and an arm that lands anywhere else has lost at any price.

Three scenarios: a menu question that must book nothing, a straight booking,
and a change to an already-confirmed reservation — the one where GPT-4.1 mini
was caught holding two tables for one person.

It runs from the **Bench** tab, or from the command line:

```bash
.venv/bin/python scripts/bench.py --draft groq-oss --strong gpt-4.1-mini
```

Each arm gets its own throwaway database, because sharing one would let the
first arm's booking occupy the table the next arm is about to ask for and the
bench would be measuring the order they ran in. They run one after another
rather than in parallel like the Arena: three tool-heavy conversations at once
is how you discover what a free tier's tokens-per-minute cap feels like.

**What the shape of the result already says**, before any provider is called,
because it follows from the design rather than from a measurement:

- On a read-only conversation the routed arm never escalates, so it costs
  exactly what the draft costs and reaches the same state. Routing is free.
- On a write-heavy conversation the routed arm escalates on nearly every turn,
  and each hand-off pays for the draft round it threw away *on top of* the full
  strong-model turn. Against **always-strong** that is a straight loss; what it
  buys is over **always-draft**, and what it buys is correctness, not money.

So the honest headline is that routing is not a saving, it is a **mix bet**: it
pays on conversations that are mostly questions and loses on conversations that
are mostly bookings. Which way a real deployment falls is a question about the
customers, not about the models — and the bench is how you answer it for yours
instead of guessing.

The absolute numbers belong to whoever runs it, on their keys, on the day they
run it, so they are not reproduced here. `--markdown` prints the table ready to
paste, and the tab has the same thing behind an accordion.

The Bench tab is the most expensive button in the app — one click spends on
every model at once — so it has its own switch, `ARNIE_BENCH=off`, and the tab
disappears when it is set. Same reasoning as `ARNIE_IMAGES`, more urgently.

## Model registry

Not every model can do everything, so each one declares what it supports:

| Model | Tools | Writes | Notes |
|---|---|---|---|
| GPT-4.1 mini · OpenAI | yes | **yes** | default |
| GPT-4.1 nano · OpenAI | yes | no | cheapest cloud option; skips tool calls it should make |
| Gemini 3.1 Flash Lite · Google | yes | **yes** | free tier; the one that rebooked correctly |
| GPT-OSS 120B · Groq | yes | no | fastest; the intended draft. Free tier caps tokens/minute |
| Llama 3.2 3B · Ollama | yes | no | local, free, loose with types |
| DeepSeek-R1 1.5B · Ollama | **no** | no | local; chats but cannot look anything up |

The picker only lists models whose credentials are present, hides the local ones
when Ollama is unreachable, and warns when the selected model cannot use tools.

**Writes** is `trusted_for_writes`: the models allowed to *finish* a turn that
mutates the database, and therefore the ones offered as an escalation target.
It is a declared policy, not a benchmark. Groq is false there not because it was
seen failing a booking but because it has not been measured on the modification
case, and the default for unmeasured has to be the cautious one.

---

## Running it

```bash
git clone <this repo> && cd llm-tool-calling-lab
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

The routing bench, also against real providers, a few cents a run:

```bash
.venv/bin/python scripts/bench.py --markdown
```

## Tests

```bash
.venv/bin/python -m pytest
```

Around 170 tests, **no API key, no network, no cost**. The booking rules run
against a temporary SQLite database; the tool loop runs against a scripted
backend that replays canned streaming chunks, which is what makes it possible to
test fragmented arguments, parallel tool calls, chained rounds, runaway loops,
rate limits, the approval protocol and every escalation trigger without spending
anything — including the one that matters most, that a discarded round leaves no
orphaned `tool_calls` in the transcript. The bench is tested against scripted
models that are caricatures on purpose: one always double-books, one always
cancels first. If it cannot tell those two apart it is not measuring anything. The media tests
stub the OpenAI client, including one that asserts an off-menu dish **never
reaches the image API**.

## Cost

Image generation is the only per-call cost the *model* can trigger on its own,
and it is bounded three ways: only
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
visitors generating images on your key, and `ARNIE_BENCH=off` unless you want
them replaying three conversations against every model on it.

---

Written while working through Ed Donner's
[LLM Engineering](https://github.com/ed-donner/llm_engineering) course — the
ideas come from weeks 1 and 2, the code is my own.
