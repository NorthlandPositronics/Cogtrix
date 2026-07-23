# Comprehensive-test configuration

A single, dedicated, **secret-free** Cogtrix config for the heavy
non-deterministic harnesses — the agent-complexity fleet, the PM role test, and
the Gate-2 eval (see `docs/optional/testing/`).

## Files

| File | Tracked? | Purpose |
|---|---|---|
| `cogtrix.comprehensive.yaml` | ✅ committed | The dedicated config. **No secrets** — every key is injected from the environment. |
| `cogtrix.comprehensive.env.example` | ✅ committed | Template for the secrets. |
| `.env` | 🚫 gitignored | Your real keys/tokens. Created from the template. |
| `env_loader.py` | ✅ committed | Loads `.env` + pins `COGTRIX_CONFIG_FILE` at test start. |
| `conftest.py` | ✅ committed | Auto-loads the above for any test under this directory. |

## Setup (once)

```bash
cd tests/comprehensive
cp cogtrix.comprehensive.env.example .env
# edit .env and fill in the real values
```

The keys:

| Env var | Used by |
|---|---|
| `COGTRIX_PROVIDER_SPARK_API_KEY` | the local `spark` vLLM endpoint (agent-complexity, PM) — generic per-provider override (#2222) |
| `DEEPSEEK_API_KEY` | the Gate-2 eval cloud model |
| `TAVILY_API_KEY` | `web_search` tool |
| `OPENWEATHER_API_KEY` | weather tool |

## How "loaded upon tests start" works

`env_loader.load_comprehensive_env()`:

1. `python-dotenv`-loads `.env` into `os.environ` (so Cogtrix's `_apply_env_vars`
   resolves the keys), and
2. sets `COGTRIX_CONFIG_FILE` to `cogtrix.comprehensive.yaml` so `load_config` /
   `find_config_file` resolve **this** config deterministically — no dependence
   on `~/.config/cogtrix` vs `~/.cogtrix/config` (the ambiguity that produced a
   stale config + dead key in the 2026-06-24 cycle, finding F-01).

Existing environment variables win over `.env` (so CI-injected secrets are not
clobbered); pass `override=True` to flip that.

## Using it from each harness

**Anything under `tests/comprehensive/`** — automatic via `conftest.py`.

**PM role test / Gate-2 (in-process pytest, elsewhere)** — at session start:

```python
from tests.comprehensive.env_loader import load_comprehensive_env
load_comprehensive_env()                 # keys + COGTRIX_CONFIG_FILE
```

**Agent-complexity fleet (Docker):**

```bash
python -m tests.agent_complexity.runner \
    --config-path tests/comprehensive/cogtrix.comprehensive.yaml ...
```

The runner injects the secrets into each container via `docker --env-file`
(#2219). It **auto-detects** the `.env` sibling of `--config-path` — so the
command above picks up `tests/comprehensive/.env` with no extra flag. Override
with `--env-file <path>`, or pass a missing path to skip. If no secrets file is
found the runner warns (keyed providers/tools would be unauthenticated).

## Security

`.env` is gitignored and must never be committed. Config-authoritative secret
env vars are read once and unset from the environment after load (#2223 — every
LLM provider key + the search-tool keys); a few env-direct consumers
(weather / whatsapp / telegram / slack) are deferred to #2223 phase 2.
