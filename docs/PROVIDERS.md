# Cogtrix Provider Setup

Step-by-step guides for configuring LLM providers. If you're new to Cogtrix, start with the [Quick Start](../README.md#quick-start) first — you can come back here when you want to add or switch providers.

## Table of Contents

- [Which Provider Should I Choose?](#which-provider-should-i-choose)
- [Overview](#overview)
- [OpenAI](#openai)
- [Ollama](#ollama)
- [Groq](#groq)
- [Together AI](#together-ai)
- [xAI (Grok)](#xai-grok)
- [DeepSeek](#deepseek)
- [Local vLLM](#local-vllm)
- [Anthropic Claude](#anthropic-claude)
- [Google Gemini](#google-gemini)
- [Multiple Providers](#multiple-providers)
- [Troubleshooting](#troubleshooting)

---

## Which Provider Should I Choose?

Not sure where to start? Use this table:

| I want... | Best choice | Setup time |
|-----------|-------------|------------|
| **Free, private, runs on my machine** | **Ollama** (default) | 5 minutes |
| **Best quality, don't mind paying** | **OpenAI** (GPT-4.1) | 2 minutes (need API key) |
| **Fast inference, free tier available** | **Groq** | 3 minutes (need API key) |
| **Wide model selection, competitive pricing** | **Together AI** | 3 minutes (need API key) |
| **Full control, own GPU server** | **vLLM** | 15 minutes |
| **Claude models (reasoning, long context)** | **Anthropic** | 2 minutes (need API key) |
| **Gemini models (multimodal, fast)** | **Google** | 2 minutes (need API key) |
| **DeepSeek V3 / R1 reasoning, low cost** | **DeepSeek** | 2 minutes (need API key) |

Cogtrix defaults to Ollama on `localhost:11434`. If you already have Ollama running, you don't need to configure anything — just run `uv run python cogtrix.py`.

You can configure multiple providers and switch between models at runtime with `/model <alias>`.

---

## Overview

Cogtrix supports four provider types:

| Type | Protocol | Use For |
|------|----------|---------|
| `openai` | OpenAI API | OpenAI, Groq, Together, vLLM, LocalAI, xAI, DeepSeek |
| `ollama` | Ollama API | Ollama servers |
| `anthropic` | Anthropic API | Anthropic Claude (requires `cogtrix[anthropic]`) |
| `google` | Google Generative AI | Google Gemini (requires `cogtrix[google]`) |

Provider type values are case-insensitive (`"OpenAI"`, `"OLLAMA"`, etc. all work).

---

## OpenAI

### Setup

1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys)

2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run:
   ```bash
   uv run python cogtrix.py --model openai
   ```

### Configuration

**Environment variable only:**
```bash
export OPENAI_API_KEY="sk-..."
uv run python cogtrix.py --model gpt4
```

**Config file:**
```yaml
providers:
  openai:
    type: openai
    api_key: "sk-..."

models:
  default: gpt4
  gpt4:
    provider: openai
    model: gpt-4.1
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `gpt-4.1` | 1M | Complex tasks, coding |
| `gpt-4.1-mini` | 1M | Fast, cost-effective (default) |
| `gpt-4.1-nano` | 1M | Fastest, cheapest |
| `o3` | 200K | Reasoning tasks |
| `o3-mini` | 200K | Fast reasoning |

---

## Ollama

### Setup

1. Install Ollama from [ollama.com](https://ollama.com/download)

2. Start the server:
   ```bash
   ollama serve
   ```

3. Pull a model:
   ```bash
   ollama pull qwen3:8b       # or any model you prefer
   ```

4. Run:
   ```bash
   uv run python cogtrix.py           # Ollama is the default provider
   uv run python cogtrix.py -m qwen3:8b   # use a different model
   ```

No configuration file is needed for local Ollama — Cogtrix connects to `localhost:11434` automatically.

### Remote Ollama Server

Set the `COGTRIX_OLLAMA` environment variable to point at a remote server:

```bash
export COGTRIX_OLLAMA="192.168.1.100"          # default port 11434
export COGTRIX_OLLAMA="192.168.1.100:8080"     # custom port
```

Or use a config file:

```yaml
providers:
  ollama:
    type: ollama
    base_url: "http://192.168.1.100:11434"

models:
  default: local
  local:
    provider: ollama
    model: qwen3:8b
```

### Popular Models

| Model | Size | Best For |
|-------|------|----------|
| `qwen3:8b` | 8B | General purpose (default) |
| `qwen3:30b-a3b` | 30B (3B active) | General purpose, MoE — fast on low VRAM |
| `gemma3:12b` | 12B | Multimodal, 128K context |
| `llama4:scout` | 109B (17B active) | Multimodal, MoE |
| `deepseek-r1:14b` | 14B | Reasoning, math |
| `hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | 30B (3B active) | Code generation, MoE |
| `phi4-reasoning:14b` | 14B | Reasoning, math olympiad |
| `mistral-small3.2` | 24B | Agentic, tool use |

### Multiple Ollama Servers

```yaml
providers:
  gpu-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
  cpu-server:
    type: ollama
    base_url: "http://192.168.1.101:11434"

models:
  default: gpu
  gpu:
    provider: gpu-server
    model: qwen3:8b
  cpu:
    provider: cpu-server
    model: qwen3:8b
```

---

## Groq

Fast inference with open-source models.

### Setup

1. Get an API key from [console.groq.com](https://console.groq.com/keys)

2. Configure (`.cogtrix.yaml`) with the key in the `api_key` field:

   > **Note:** Cogtrix does not read `GROQ_API_KEY` from the environment. The key must be set in the config file's `api_key` field (or provided via `--setup`). Only `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, and `DEEPSEEK_API_KEY` are read automatically.
   ```yaml
   providers:
     groq:
       type: openai
       base_url: "https://api.groq.com/openai/v1"
       api_key: "gsk-..."  # your Groq API key

   models:
     default: groq
     groq:
       provider: groq
       model: llama-3.3-70b-versatile
   ```

3. Run:
   ```bash
   uv run python cogtrix.py --model groq
   ```

### Available Models

| Model | Context | Speed |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 128K | Fast |
| `llama-3.1-8b-instant` | 128K | Very fast |
| `mixtral-8x7b-32768` | 32K | Fast |
| `gemma2-9b-it` | 8K | Fast |

---

## Together AI

Wide model selection with competitive pricing.

### Setup

1. Get an API key from [api.together.xyz](https://api.together.xyz/settings/api-keys)

2. Configure (`.cogtrix.yaml`):
   ```yaml
   providers:
     together:
       type: openai
       base_url: "https://api.together.xyz/v1"
       api_key: "..."

   models:
     default: together
     together:
       provider: together
       model: meta-llama/Llama-3-70b-chat-hf
   ```

### Popular Models

| Model | Size |
|-------|------|
| `meta-llama/Llama-3-70b-chat-hf` | 70B |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 8x7B |
| `Qwen/Qwen2-72B-Instruct` | 72B |
| `deepseek-ai/deepseek-coder-33b-instruct` | 33B |

---

## xAI (Grok)

xAI uses the OpenAI-compatible API.

### Setup

Export your API key:

```bash
export XAI_API_KEY=xai-...
```

### Configuration

```yaml
providers:
  xai:
    type: openai
    base_url: https://api.x.ai/v1
    api_key: "xai-..."  # your xAI API key

models:
  default: grok
  grok:
    provider: xai
    model: grok-4.1-fast
```

---

## DeepSeek

DeepSeek offers a hosted API with an OpenAI-compatible endpoint. Both
`deepseek-chat` (V3, general purpose) and `deepseek-reasoner` (R1,
chain-of-thought) are supported.

### Setup

Export your API key:

```bash
export DEEPSEEK_API_KEY=sk-...
```

When `DEEPSEEK_API_KEY` is set, Cogtrix automatically creates a `deepseek`
provider pointing to `https://api.deepseek.com/v1` with `deepseek-chat` as the
default model.

### Configuration

```yaml
providers:
  deepseek:
    type: openai
    base_url: https://api.deepseek.com/v1
    api_key: "sk-..."  # your DeepSeek API key

models:
  default: deepseek
  deepseek:
    provider: deepseek
    model: deepseek-chat     # or deepseek-reasoner for R1
```

> **Note:** `deepseek-reasoner` (R1) returns a `reasoning_content` field in
> every assistant message. DeepSeek's API requires this field to be echoed back
> in subsequent calls — LangChain's standard `ChatOpenAI` wrapper silently drops
> it. Cogtrix handles this transparently via an internal subclass
> (`_DeepSeekChatModel` in `src/providers/openai.py`) that captures and
> re-injects the field on every round-trip, including streaming. No extra
> configuration is needed.

---

## Local vLLM

Run models locally with vLLM server.

### Setup

1. Install vLLM:
   ```bash
   pip install vllm
   ```

2. Start the server:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-3-8b-chat-hf \
     --port 8000
   ```

3. Configure (`.cogtrix.yaml`):
   ```yaml
   providers:
     vllm:
       type: openai
       base_url: "http://localhost:8000/v1"
       # api_key: omit or leave blank — vLLM does not require authentication by default

   models:
     default: vllm
     vllm:
       provider: vllm
       model: meta-llama/Llama-3-8b-chat-hf
   ```

> **No API key needed:** vLLM (and LM Studio) run unauthenticated by default. Leave `api_key` out of the provider config entirely. Cogtrix will connect without a key.

---

## Anthropic Claude

> **Optional dependency:** `pip install "cogtrix[anthropic]"` or `uv pip install "cogtrix[anthropic]"`

### Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)

2. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. Run (the API key auto-creates the provider; the first available provider is used by default):
   ```bash
   uv run python cogtrix.py
   ```

### Configuration

**Environment variable only:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python cogtrix.py --model claude
```

**Config file:**
```yaml
providers:
  anthropic:
    type: anthropic
    api_key: "sk-ant-..."

models:
  default: claude
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `claude-sonnet-4-5` | 200K | Complex reasoning (default) |
| `claude-opus-4-5` | 200K | Highest capability |
| `claude-haiku-4-5` | 200K | Fast, cost-effective |

> **Note:** Anthropic does not provide an embedding API. Use a different provider (OpenAI or Ollama) for RAG embeddings.

---

## Google Gemini

> **Optional dependency:** `pip install "cogtrix[google]"` or `uv pip install "cogtrix[google]"`

### Setup

1. Get an API key from [aistudio.google.com](https://aistudio.google.com/apikey)

2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="..."
   ```

3. Run (the API key auto-creates the provider; the first available provider is used by default):
   ```bash
   uv run python cogtrix.py
   ```

### Configuration

**Environment variable only:**
```bash
export GEMINI_API_KEY="..."
uv run python cogtrix.py --model gemini
```

**Config file:**
```yaml
providers:
  google:
    type: google
    api_key: "..."

models:
  default: gemini
  gemini:
    provider: google
    model: gemini-2.5-flash
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `gemini-2.5-flash` | 1M | Fast, cost-effective (default) |
| `gemini-2.5-pro` | 1M | Highest capability |

---

## Multiple Providers

Configure multiple providers for different use cases:

```yaml
providers:
  ollama-local:
    type: ollama
  ollama-gpu:
    type: ollama
    base_url: "http://gpu-server:11434"
  openai:
    type: openai
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."

models:
  default: local
  local:
    provider: ollama-local
    model: qwen3:8b
  fast: groq/llama-3.3-70b-versatile
  smart: openai/gpt-4.1
  coder:
    provider: ollama-gpu
    model: qwen3-coder:30b-a3b

delegate:
  enabled: true
  allowed_models: [fast, smart, coder]
```

### Switching Models

**At startup:**

```bash
# Use local model
uv run python cogtrix.py --model local

# Use GPU coder model
uv run python cogtrix.py --model coder

# Use OpenAI smart model
uv run python cogtrix.py --model smart

# Use Groq fast model
uv run python cogtrix.py --model fast
```

**At runtime** (during an interactive session):

```
You: /model smart
Switched to model smart (openai / gpt-4.1)

You: /m fast
Switched to model fast (groq / llama-3.3-70b-versatile)

You: /m local
Switched to model local (ollama-local / qwen3:8b)
```

The `/model` (or `/m`) command rebuilds the LLM and agent immediately. If the switch fails (e.g., invalid alias), the previous configuration is automatically restored. The `/provider` command is read-only — it lists configured providers and their connection details.

---

## Troubleshooting

### OpenAI

**"Invalid API key"**
```
Check:
- OPENAI_API_KEY environment variable is set
- Key starts with "sk-"
- Key is not expired
```

**"Model not found"**
```
Check:
- Model name is correct (e.g., "gpt-4.1" not "gpt4o")
- Your API key has access to the model
```

**"Rate limit exceeded"**
```
Solutions:
- Wait and retry
- Use a different model
- Upgrade your API plan
```

### Local vLLM / LM Studio

**"The api_key client option must be set"**
```
vLLM and LM Studio do not require an API key. Remove the api_key field
from the provider config (or leave it blank). Cogtrix passes a placeholder
automatically so the OpenAI SDK does not reject the connection.
```

**"Connection refused"**
```
Check:
- vLLM server is running (python -m vllm.entrypoints.openai.api_server ...)
- base_url matches the server port (default 8000 for vLLM)
```

### Ollama

**"Connection refused"**
```
Check:
- Ollama is running: ollama serve
- Port 11434 is accessible
- Firewall allows connection
```

**"Model not found"**
```
Pull the model first:
ollama pull qwen3:8b
```

**"Out of memory"**
```
Solutions:
- Use a smaller/MoE model (e.g., qwen3:30b-a3b instead of qwen3:32b)
- Close other applications
- Use quantized models (e.g., llama4:scout-q4_K_M)
```

### Groq / Together

**"Invalid API key"**
```
Check:
- API key is correct
- api_key is in the provider config, not environment
```

**"Model not available"**
```
Check provider documentation for current model names
```

### Anthropic

**"Invalid API key"**
```
Check:
- ANTHROPIC_API_KEY environment variable is set
- Key starts with "sk-ant-"
- Key is not expired
```

**"Module not found: langchain_anthropic"**
```
Install the optional dependency:
pip install "cogtrix[anthropic]"
```

### Google

**"Invalid API key"**
```
Check:
- GEMINI_API_KEY environment variable is set
- Key is valid (test at aistudio.google.com)
```

**"Module not found: langchain_google_genai"**
```
Install the optional dependency:
pip install "cogtrix[google]"
```

### General

**"Timeout"**
```
Solutions:
- Check network connection
- Increase timeout in delegate config
- Use a faster model or provider
```

**"Empty response"**
```
Check:
- Model is loaded correctly
- Input is not empty
- Try a simpler prompt
```

---

## See Also

- [Configuration Reference](CONFIGURATION.md) — provider YAML format and all config keys
- [Architecture Overview](ARCHITECTURE.md) — provider registry internals
- [Tools Reference](TOOLS_REFERENCE.md) — tools that require provider API keys
