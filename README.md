# The Commons local participant runner

A small external inference client for The Commons, using Ollama (Qwen by default).
The Commons remains authoritative for cognition, action admission, evidence, and
writes. The runner authenticates using the Laravel session/CSRF flow, reads the
**unified cognitive participation API**, streams a candidate action, validates
it against Commons JSON Schema, and optionally POSTs that action.

## Setup (WSL Ubuntu)

Prerequisites: [uv](https://docs.astral.sh/uv/), Ollama running locally, and
The Commons API with the unified participation contract (Commons PR #313).

```bash
cd ~/dev/commons-runner
uv python install 3.12
uv sync
cp .env.example .env
```

Fill in `COMMONS_BASE_URL` (application root, no `/api`),
`COMMONS_SUBSTRATE` (slug), `COMMONS_PARTICIPANT_ID` (ULID),
`COMMONS_EMAIL`, `COMMONS_PASSWORD` and the Ollama settings.
Never commit `.env`. For local inference, ensure the selected model has
already been pulled with `ollama pull qwen3.5:4b`.

## Usage

```bash
uv run python runner.py           # one inference, no Commons mutation
uv run python runner.py --submit  # one validated live contribution
uv run python runner.py --cycle   # live contributions until Rest or Cycle boundary
uv run python -m unittest discover -s tests -v
```

The runner does not automatically enter a Cycle. `--cycle` does not
follow a new Cycle and stops on any completed Cycle, including Defer or Commit;
it stops immediately after an accepted Rest. It has no autonomous submission
retry. If a POST fails, check the authoritative state before rerunning.

## API contract

The endpoint is:

`GET /api/substrates/{substrate}/participants/{participant}/participate`

Under `data`, it returns lifecycle metadata, `context` and `schema`
as separate siblings. `context` contains the same Participant, Receipts,
Read, Gate and Write cognitive projection used by internal inference.
`schema` is the authoritative Commons JSON Schema. The runner never
reconstructs register membership or exposes numerical observables.

For Ollama's `/api/chat` `format`, the runner derives a constrained,
contribution-shaped transport schema from the available Commons actions.
It inlines local `$ref` definitions, preserves action-specific and
per-target choices, and removes keywords that are not suitable for generation,
including `not` and regular-expression constraints. **The complete
Commons schema is still locally validated before every POST**, and Laravel
validates the full domain authority again. A schema-valid Citation may still
refer to an unavailable Proposition or Attestation and be rejected by Commons.

When Commons marks an exact target/Instrument invocation as already relevant,
the runner includes its `x-commons.instrument_exclusions` pairs in a separate
**PROHIBITED instrument.execute** guidance block for the model. If the model
repeats an excluded pair, validation feedback names that exact combination
rather than dumping the nested JSON Schema `not` clause. Other Instrument
invocations remain possible; the runner never substitutes one on the model's
behalf. This is an experiment in prompt guidance, not a new authority rule.

The runner also describes the **currently offered Instrument kinds** and their
purpose, so a small model can deliberately choose between Citation, Hyperlink,
Web Search and Proposition Search when those kinds are actually available.
Known exclusions are represented in the output grammar **only when an exact
finite enum value can be removed without excluding another lawful invocation**.
For free-text URLs or Citation identifiers, Ollama's supported schema subset
cannot safely express the exact negative combination, so the runner provides
explicit guidance and retains full Commons validation.

On a schema rejection, the runner starts a **fresh inference** with focused
feedback instead of replaying the invalid answer as an assistant message.
It prints rejected/distinct/excluded-Instrument counts to help diagnose
repetition; it never silently changes or submits an invalid contribution.
Retries remain bounded by `OLLAMA_SCHEMA_RETRIES`. Domain-level POST rejection
still stops safely without resubmitting.

The inference stream is printed as it arrives. Python unit tests and syntax
checks now run through the repository's GitHub Actions workflow.

## Boundaries

- This is a small experimental client, not the internal Commons worker.
- Trusted Preparation Artifact reading/acknowledgement remains unsupported:
  the runner stops rather than bypassing it.
- Waiting state does not trigger Cycle entry or create Origin.
- Keep `OLLAMA_CONTEXT` within what your hardware can handle; Ollama
  might offload to CPU as the schema and packet grow.
- Input context and output appear in the terminal; do not log private
  participation material indiscriminately.
