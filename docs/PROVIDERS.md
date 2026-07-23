# Cogtrix Provider Setup

Step-by-step guides for configuring LLM providers. If you're new to Cogtrix, start with the [Quick Start](../README.md#quick-start) first — you can come back here when you want to add or switch providers.

## Table of Contents

- [Which Provider Should I Choose?](#which-provider-should-i-choose)
- [Overview](#overview)
- [OpenAI](#openai)
- [Ollama](#ollama)
- [Groq](#groq)
- [Together AI](#together-ai)
- [Local vLLM](#local-vllm)
- [Multiple Providers](#multiple-providers)
- [Troubleshooting](#troubleshooting)

---

## Which Provider Should I Choose?

Not sure where to start? Use this table:

| I want... | Best choice | Setup time |
|-----------|-------------|------------|
| **Free, private, runs on my machine** | **Ollama** (default) | 5 minutes |
| **Best quality, don't mind paying** | **OpenAI** (GPT-4o) | 2 minutes (need API key) |
| **Fast inference, free tier available** | **Groq** | 3 minutes (need API key) |
| **Wide model selection, competitive pricing** | **Together AI** | 3 minutes (need API key) |
| **Full control, own GPU server** | **vLLM** | 15 minutes |

Cogtrix defaults to Ollama on `localhost:11434`. If you already have Ollama running, you don't need to configure anything — just run `python cogtrix.py`.

You can configure multiple providers and switch between them at runtime with `/provider <name>`.

---

## Overview

Cogtrix supports four provider types:

| Type | Protocol | Use For |
|------|----------|---------|
| `openai` | OpenAI API | OpenAI, Groq, Together, vLLM, LocalAI |
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
   python cogtrix.py -p openai
   ```

### Configuration

**Environment variable only:**
```bash
export OPENAI_API_KEY="sk-..."
python cogtrix.py -p openai -m gpt-4.1
```

**Config file:**
```yaml
provider: openai
providers:
  openai:
    type: openai
    model: gpt-4.1
    api_key: "sk-..."
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
   python cogtrix.py           # Ollama is the default provider
   python cogtrix.py -m qwen3:8b   # use a different model
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
provider: ollama
providers:
  ollama:
    type: ollama
    base_url: "http://192.168.1.100:11434"
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
provider: gpu-server
providers:
  gpu-server:
    type: ollama
    base_url: "http://192.168.1.100:11434"
    model: qwen3:8b
  cpu-server:
    type: ollama
    base_url: "http://192.168.1.101:11434"
    model: qwen3:8b
```

---

## Groq

Fast inference with open-source models.

### Setup

1. Get an API key from [console.groq.com](https://console.groq.com/keys)

2. Configure (`.cogtrix.yaml`):
   ```yaml
   provider: groq
   providers:
     groq:
       type: openai
       base_url: "https://api.groq.com/openai/v1"
       api_key: "gsk-..."
       model: llama-3.3-70b-versatile
   ```

3. Run:
   ```bash
   python cogtrix.py -p groq
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
   provider: together
   providers:
     together:
       type: openai
       base_url: "https://api.together.xyz/v1"
       api_key: "..."
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
   provider: vllm
   providers:
     vllm:
       type: openai
       base_url: "http://localhost:8000/v1"
       model: meta-llama/Llama-3-8b-chat-hf
   ```

---

## Multiple Providers

Configure multiple providers for different use cases:

```yaml
provider: ollama-local

providers:
  ollama-local:
    type: ollama
    model: qwen3:8b
  ollama-gpu:
    type: ollama
    base_url: "http://gpu-server:11434"
    model: qwen3:8b
  openai:
    type: openai
    model: gpt-4.1-mini
  groq:
    type: openai
    base_url: "https://api.groq.com/openai/v1"
    api_key: "gsk-..."
    model: llama-3.3-70b-versatile

models:
  fast: groq/llama-3.3-70b-versatile
  smart: openai/gpt-4.1
  local: ollama-local/qwen3:8b
  coder: ollama-gpu/qwen3-coder:30b-a3b

delegate:
  enabled: true
  allowed_models: [fast, smart, coder]
```

### Switching Providers

**At startup:**

```bash
# Use local Ollama
python cogtrix.py -p ollama-local

# Use GPU server
python cogtrix.py -p ollama-gpu

# Use OpenAI
python cogtrix.py -p openai

# Use Groq
python cogtrix.py -p groq
```

**At runtime** (during an interactive session):

```
You: /provider openai
Switched to provider openai (model: gpt-4.1)

You: /p groq
Switched to provider groq (model: llama-3.3-70b-versatile)

You: /model gpt-4.1-mini
Switched to model gpt-4.1-mini (openai)
```

The `/provider` (or `/p`) and `/model` (or `/m`) commands rebuild the LLM and agent immediately. If the switch fails (e.g., invalid model name), the previous configuration is automatically restored.

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
