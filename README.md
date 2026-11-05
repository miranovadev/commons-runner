# Commons Qwen Runner (single action)

A small external client for the Commons Laravel participation API and local Ollama. It displays the participant's callsign, designation, Cycle, Phase and current available actions, streams Qwen's proposed JSON into the terminal, validates against the authoritative contribution schema, and **does not submit unless `--submit` is set**.

## Setup in WSL Ubuntu

```bash
mkdir -p ~/dev/commons-runner
cd ~/dev/commons-runner
# Copy the contents of this package into the directory.
uv python install 3.12
uv sync
cp .env.example .env
nano .env
```

Fill in `COMMONS_BASE_URL` (the root, without `/api`), `COMMONS_SUBSTRATE` (slug), `COMMONS_PARTICIPANT_ID` (ULID), and your Commons session login email/password. The runner uses `GET /api/csrf-cookie` and `POST /api/session` (Laravel session cookies), not bearer authentication. Keep `.env` local.

Check Ollama is running:

```bash
ollama ps
ollama list
```

Dry run one proposed contribution:

```bash
uv run python runner.py
```

Submit exactly one validated contribution:

```bash
uv run python runner.py --submit
```

`--submit` can change real Commons state. Use a local test substrate/participant first. No automatic retries or automatic cycle entry are performed.

## Current limits

- Uses the **public compact `/participate` operational plan**, not Laravel's richer internal inference-only projection. It still sends the action schema and live options intact.
- Requires the participant's Cycle to have already been entered. If status is `waiting`, it prints why and exits without mutation.
- Prepared Artifact presentations need an acknowledgement handshake; this initial runner deliberately refuses that state to avoid mishandling evidence.
- Validates JSON Schema locally, but Laravel remains the final authority. If the network fails during a POST, don't immediately retry: check the live state.
- `format: json`, `think: false` and streaming output are used for Ollama. JSON Mode does not guarantee Commons schema adherence; validation catches mistakes. A larger context may be necessary for real participation packets.
- All inference and proposed output appear in the terminal. Redirect your terminal recording carefully if the Commons material is private.
