*This project has been created as part of the 42 curriculum by laviles*

---

## 📝 Description

**call_me_maybe** is a constrained-decoding pipeline that turns free-form natural
language into a single, structurally-valid JSON function call. It wraps a local
LLM (`Qwen/Qwen3-0.6B` through the provided `llm_sdk`) and replaces unconstrained
sampling with a state-machine-driven filter: at every token the decoder computes
the exact set of token ids that keep the partially-built output syntactically and
semantically valid, then picks the best one by logit rank.

**Goal.** Given a natural-language prompt and a set of callable function
definitions, produce:

```json
{"name": "<function>", "parameters": {<typed arguments>}}
}

```

...where the JSON is guaranteed well-formed, the function name exists in the
schema, the argument keys exist in that function, and every value respects its
declared type — **even though the model was never taught the format**. All of
that guarantee is enforced by the decoder, not requested from the model in a
prompt.

---

## 🚀 Instructions

Requires **Python 3.10+** and [`uv`](https://docs.astral.sh/uv/). The only
runtime dependencies are `numpy` and `pydantic`; `llm_sdk` is vendored in this
repository (permitted by the subject) and used as-is.  


```bash
# 1. install
make install          # == uv sync

# 2. run the pipeline on the bundled prompts
make run              # == uv run python -m src

# custom input/output files
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input               data/input/function_calling_tests.json \
    --output              data/output/answer.json

# tests + static checks (what the reviewer runs)
make test             # pytest
make lint             # flake8 + mypy
```

The pipeline is **non-interactive**: it reads the input JSON, runs every prompt
through the decoder, writes one JSON answer per prompt to the output file, and
returns a non-zero exit code on failure. No API keys, no network calls at run
time.

---

## 🧠 Algorithm explanation

The decoder is a **finite-state machine + schema validator + trie**, driven by
the model's logits. Generation is a loop; each step narrows the model's choice
from the whole vocabulary down to the tokens that keep the output valid.

### 1. The automaton (`src/decoder/state.py`)

A 16-phase character-level FSM drives the JSON structure. It starts at `ROOT`
and walks `OBJECT_OPEN → IN_OBJECT → KEY_START → IN_KEY → KEY_END → COLON →
IN_*_VALUE → VALUE_END → PARAMS_OBJECT → COMPLETE`, tracking `depth`,
`current_key`, and `keys_enclosed`. It is the single source of truth for *what
the output looks like right now*. Every candidate token is fed through it
**character by character**, so a token is accepted or rejected as a whole based
on the path it would trace.

### 2. Schema validation (`src/decoder/schema_validator.py`)

`SchemaContext.allows_token()` is the semantic layer. Five AND-ed clauses check
the token against the selected function's signature:  


| # | Clause | Gates | Blocks |
|---|--------|-------|--------|
| 1 | `_allows_name_value` | value of `"name"` must be a function in the schema | a name that matches nothing |
| 2 | `_allows_param_key` | keys of `"parameters"` exist, no duplicates | unknown / duplicate keys |
| 3 | `_allows_value_type` | a value matches its declared type | a string where a number is expected, etc. |
| 4 | `_allows_params_close` | `}` only after all required keys | closing with a required missing |
| 5 | `_allows_integer_form` | `"integer"` rejects `.`/`e`/`E` | `4.0` / `1e3` for an int parameter |

Clauses 3 and 5 read `token_text`; the rest work from phases and buffers, so the
schema never re-parses the output.

### 3. The trie (`src/decoder/trie.py`)

Function names live in a character trie. The filter walks it character by
character to check whether a partially-typed name is still a valid prefix of a
real function — this is what lets the model emit `"fn_greet"` one token at a
time without ever producing an invalid function name.

### 4. Token filter (`src/decoder/token_filter.py`)

`compute_allowed_ids()` narrows the ~151k vocabulary to the ids whose decoded
text keeps the output valid. It runs in phases: bucket the pre-indexed candidates
by expected first character, drop any token whose decoded text breaks UTF-8 or
the FSM, then run `allows_token()` on the survivors. When logits are supplied it
only validates the model's top-k (default 2000) instead of the full vocabulary.

It has two branches with different contracts, and they behave differently on
purpose:

- **M1 — fast path.** The model's raw argmax, validated on its own. At most one
  candidate; if it survives, nothing else was computed.
- **M2 — top-k masking.** The model's top-k, validated as a set. It returns the
  *whole valid tier* (up to 2000 candidates) so the fine pass below has real
  alternatives to fall back on.

### 5. Fine pass (post-argmax)

Choosing the top allowed token can still be wrong when a single BPE token
crosses a structural boundary (enters and exits a construct at once), because
the per-token FSM view can't see the intermediate character. So the winning token
is re-simulated **character by character against a fresh `SchemaContext`**
before it is committed. If it fails, the next-best allowed token is tried; with
M2 there are real alternatives to choose from.

### 6. The oracle (static injection)

Not everything needs the model. The wrapper JSON punctuation, indentation, and
the `"parameters": {` header are fully determined once the function is chosen.
An **oracle keyed on the current FSM state** injects these deterministic spans
pre-tokenized, skipping the forward pass entirely. This is where most of the
speedup lives.

### The commit invariant

Every committed token follows the exact same 4-step sequence — encapsulated once
in `_commit_token()`, not repeated per code path:

```
input_ids.append(id) → state.update_from_text(text) → schema.update(state)
→ emitted_parts.append(text)
```

`emitted_parts` is what the fine pass and the oracle read to see what's already
been emitted; a commit that skipped its last row would leave them working on an
incomplete view of the output.

### The decode loop

The loop repeats one idea: **before paying for a forward pass, ask whether the
model is needed at all.** Each iteration walks the stages above and stops at the
first question whose answer removes the model from the critical path.

#### 0 · Inject the static header

The output always opens the same way (`\n\n{\n  "name": "`), and the model has
not been asked for it yet. It is encoded once and injected directly.

> **Question:** *what must every valid output begin with?*
> **Action:** encode the fixed prefix and commit it. No forward pass.

#### 1 · Ask whether the state already determines the next text

Once the function is chosen, large parts of the remaining output are forced: the
indentation, the `"parameters": {` header, the argument keys in their declared
order. The oracle holds a table of such spans, each keyed on the current FSM
state.

> **Question:** *does the current state match one of the deterministic spans?*
> **Action:** if yes, inject that span pre-tokenized and go back to the loop —
> no forward, no filtering. If the injection fails to advance the state, it falls
> through to step 2 rather than looping forever on the same span.

#### 2 · Ask whether only one token is legal

For structural phases the valid candidate set is small (~10–100). Running the
full filter on it is cheap, so the code does — and then checks the size.

> **Question:** *is exactly one token valid here?*
> **Action:** if yes, commit it. A unique valid token carries no information the
> model could contradict, so the forward pass is pure waste.

#### 3 · Consult the model, and narrow the vocabulary

This is the only stage that calls `forward()`. The logits come back over the full
~151k vocabulary, and the filter reduces them to the ids that keep the output
valid (M1 validates the raw argmax alone; M2 validates the model's top-2000 as a
set).

> **Question:** *among the tokens that keep the output valid, which does the
> model prefer?*
> **Action:** pick the highest-logit id inside the allowed set.

#### 4 · Re-simulate the winner before trusting it

The per-token view of the FSM cannot see *inside* a token: a single BPE token can
enter and leave a construct at once, and the intermediate character would go
unread. So the winner is replayed character by character against a fresh
`SchemaContext` before it is committed.

> **Question:** *does the best candidate survive a character-by-character
> replay?*
> **Action:** if yes, commit. If no, **drop it and ask the same question again**
> with the next-best candidate. With M2 there are up to 2000 alternatives to fall
> back on; with M1 there is only one, so a failure there ends generation — a
> deliberate asymmetry, since M1 exists to avoid validating anything extra once
> the decision is already forced.

#### 5 · Commit atomically

Whichever token won, it is committed through the single 4-step sequence above, so
every code path keeps the FSM, the schema and `emitted_parts` in agreement. A
`"number"` about to close as an integer is completed to `2.0` here, before the
commit. The loop ends when the FSM reaches `COMPLETE`.

> **Question:** *has the FSM reached `COMPLETE`?*
> **Action:** if yes, decode the generated ids and return the text. If no, return
> to step 1.

Of these six steps, **only step 3 calls the model.** Steps 1 and 2 remove whole
forwards from the run, which is where most of the speedup comes from.

---

## 💡 Design decisions

- **Constrain structure, not the model.** The plan is to make invalid output
  *impossible to select*, not to ask nicely in a prompt. Grammar lives in the
  FSM; semantics live in the validator.
- **Deterministic work never goes through the model.** The oracle removes the
  computable part of the output from the forward-pass budget (see Performance).
- **Fast path + top-k as a pair.** M1 keeps the common case free of extra
  validation; M2 guarantees the fine pass always has alternatives. The two are
  a deliberate trade of compute for guaranteed recoverability.
- **Char-by-char FSM validation.** A BPE token can straddle a structural
  boundary, so every candidate is simulated as a character sequence, not
  validated as an opaque string.
- **Separate encoding from generated ids** (`prompt_length` slice) so the
  returned output contains only generated tokens, never the prompt.
- **`slots=True`** on the hot dataclasses (`DecoderState`, `TrieNode`) — the
  filter touches these per candidate, and dropping `__dict__` measurably speeds
  attribute access.
- **Only public `llm_sdk` surface**; no private methods or attributes are used.

---

## 📊 Performance analysis

All numbers are from a clean run (no other load on the machine) on the 11-prompt
input suite, model `Qwen/Qwen3-0.6B`, CPU-only build.

### ⏱️ Latency — the optimization stages

The KPI was *11-prompt suite under 5 minutes* on real CPU. Starting from a naive
"ask the model for everything", the strategy was to move deterministic work off
the model's critical path:

```
  Stage                            | Wall time | Forwards
  ---------------------------------+-----------+----------
  Original (all from the model)    |  34.9 min |     ~638
  Pre-index refactor               |  27.6 min |     ~638
  Static header (Opt2)             |   15.1 min|     314
  State-keyed oracle (N1)          |    7.6 min|     133
  VM tuned (4 vCPU, exec cap 80)   |    4:53   |     133  <-- KPI met
  ------------------------------------------------------------------

  ── optimization achieved (bar length = speedup vs the original) ──

  VM tuned (4 vCPU, exec cap 80)   ████████████████████  7.2x   4:53    🚀
  State-keyed oracle (N1)          █████████████        4.6x   7.6 min
  Static header (Opt2)             ██████                2.3x   15.1 min
  Pre-index refactor               ████                  1.3x   27.6 min
  Original (all from the model)    ███                   1.0x   34.9 min
```

**~7.2x faster and ~4.8x fewer forwards** than the original. The decisive lever
was the oracle: it cut the structural forwards to zero, because by the time the
function is known the punctuation is fully determined.

> ⚠️ The last row was measured at 133 forwards. The accuracy fix that reaches
> 10/11 adds ~4 forwards (137), so that stage is **pending a clean
> re-measurement**; the optimization series above is otherwise unchanged.

### 🎯 Accuracy

| Metric | Result |
|--------|--------|
| 🎯 Function name accuracy | 11/11 (100%) |
| ✅ Full argument accuracy (exact match on every argument) | 10/11 (90.9%) |
| 🔒 JSON well-formedness | 100% (guaranteed by construction) |

The one miss is a **model-capability limit, not a structural one**. For
*"replace vowels with asterisks"* the model must map the English word
"asterisks" to the punctuation character `*`. Measured on the exact prefix, the
decoder emitted `'****'` with logit **19.13** while the expected `'*'` scored
**5.36** — a gap of **13.77** (rank ~6.300). The decoder had no way to prefer
`*`: schema validation declares the *type* (`string`), never a content pattern,
so the string's interior is unconstrained by design. Every output is valid JSON
by construction — validity is enforced, not hoped for; and the regex arguments
themselves are produced correctly in all three substitution cases.

### Why CPU-only, and what it forced

This ran entirely on CPU, inside a virtual machine:

- The GPU on the host is not passed through to the VM, and the subject forbids
  pulling in `torch`/`CUDA`/`transformers`. The solution was a **pure-CPU,
  `numpy`+`pydantic` build with the vendored `llm_sdk`** — no heavyweight
  accelerator stack to download.
- The VM was originally configured with 6 vCPUs on 4 physical cores, which
  caused oversubscription. After diagnosing it, the VM was set to 4 vCPU (1:1
  with physical cores) plus an 80% execution cap so the host OS keeps breathing.
  This alone moved the clean run to 4:53.
- The SDK has **no KV-cache** — each forward re-feeds the whole sequence — so
  the workload is compute-bound, not candidate-bound. That is precisely *why*
  cutting forwards (the oracle) beats cutting candidates per forward.

### Reliability

- Deterministic decode: the same input produces byte-identical output across
  runs and across thread counts.
- Invalid outputs cannot be produced by construction.
- Errors are handled gracefully with clear messages; the pipeline never crashes
  unexpectedly.

---

## 🧗 Challenges faced

- **A single BPE token can cross a structural boundary.** Validating tokens as
  opaque strings let tokens that entered *and* exited a construct slip through.
  Solved with the char-by-char fine pass.
- **The schema couldn't see mid-token states.** Early on, a key of a fused token
  slipped past validation ("contraf-band"). Diagnosed by probing the phase
  machine; solved by making the key validator trigger on *change* rather than on
  a phase the token never officially "entered".
- **Escape sequences inside `"name"` weren't blocked by the trie** (BUG-011) —
  the guard was in the wrong layer, where the state never reached it. Moved into
  the FSM's string step, where the backslash is actually read.
- **Two genuinely different states read identically.** The static-injection
  trigger keyed on `(phase, depth, current_key)`, and `fn_greet`'s inner
  parameter is *also* called `name` — so the instant `parameters` closed, the
  state was indistinguishable from the instant the top-level `"name"` value
  closed. The oracle fired a second time mid-output, injecting a duplicate
  `"parameters"` key that silently dropped an argument while the run still
  reported success (BUG-013). The fix was not to clear the field but to add the
  observable that separates the cases — `keys_enclosed == ∅` — so each trigger
  owns a state domain no other can enter. Both *equivalent-looking* fixes were
  unsafe: clearing `current_key` erases the legitimate reading, and the sticky
  "already opened" flag it replaced is blind to a token carrying `{` inside it.
- **The filter returned a single candidate, silently turning the fine pass's
  "try the next one" recovery into a hard veto** (BUG-014). The M2 branch now
  returns the whole valid tier.
- **A measurement that looked green was not.** A raw-logit preview showed
  well-formed JSON that differed from the committed answer, and a
  micro-optimization that won in isolation lost under real load. Lesson applied
  afterwards: trust only end-to-end byte-level diffs of the actual output file,
  never a partial view of the pipeline.

---

## 🧪 Testing strategy

- **240 unit tests** (`make test`) covering the FSM, trie, validator, filter,
  oracle, and the full generator — including every documented bug as a
  regression case.
- **Static analysis** in the same gate as the test suite: `flake8` + `mypy`
  (including `--warn-return-any`, `--disallow-untyped-defs`) must be clean.
- **End-to-end suite**: the 11 input prompts are decoded through the real
  `llm_sdk` and checked on function-name and full-argument exact match.
- **Byte-level A/B**: behaviour-changing refactors are validated by diffing
  the full generated output before and after, so an "equivalent" refactor is
  *proven* identical rather than assumed.
- **Focused probes**: single-prompt scripts for isolating specific
  phases/behaviours during debugging.

---

## 🎬 Example usage

```bash
# install
make install

# run on the bundled 11 prompts
make run

# check static analysis and tests
make lint && make test
```

Input (`data/input/function_calling_tests.json`) is a list of prompts; the
output (`data/output/answer.json`) is one call per prompt:

```json
[
  { "name": "fn_get_weather", "parameters": { "city": "Paris", "unit": "celsius" } },
  { "name": "fn_add", "parameters": { "a": 3, "b": 5 } }
]
```

---

## 📚 Resources

### References
- Hugging Face, *Text Generation Inference* concepts — logits and constrained
  decoding. https://huggingface.co/docs
- Gopher / grammar-constrained decoding — the general idea of restricting
  sampling to grammar-valid continuations.
- The 42 subject brief: *Introduction to function calling in LLMs* (chapter VI,
  "Readme Requirements", and chapter VII for the function-calling concepts).
- `pydantic` documentation — schema-driven validation patterns.
  https://docs.pydantic.dev

### How AI was used
AI was used for three things: studying the general subject material, tutoring on
Python and architecture concepts, and proposing design options for the
constrained-decoding architecture (which were then evaluated, measured, and
either adopted or discarded by the project author). All implementation,
measurement, and validation decisions — and every line of the shipped code —
were made and verified by the project author.
