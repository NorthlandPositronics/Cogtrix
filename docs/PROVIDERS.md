# Cogtrix Provider Setup

Step-by-step guides for configuring LLM providers.

## Table of Contents

- [Overview](#overview)
- [OpenAI](#openai)
- [Ollama](#ollama)
- [Groq](#groq)
- [Together AI](#together-ai)
- [Local vLLM](#local-vllm)
- [Multiple Providers](#multiple-providers)
- [Troubleshooting](#troubleshooting)

---

## Overview

Cogtrix supports two provider types:

| Type | Protocol | Use For |
|------|----------|---------|
| `openai` | OpenAI API | OpenAI, Groq, Together, vLLM, LocalAI |
| `ollama` | Ollama API | Ollama servers |

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
python cogtrix.py -p openai -m gpt-4o
```

**Config file:**
```json
{
  "provider": "openai",
  "providers": {
    "openai": {
      "type": "openai",
      "model": "gpt-4o",
      "api_key": "sk-..."
    }
  }
}
```

### Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `gpt-4o` | 128K | Complex tasks, coding |
| `gpt-4o-mini` | 128K | Fast, cost-effective |
| `gpt-4-turbo` | 128K | Previous flagship |
| `o1-preview` | 128K | Reasoning tasks |
| `o1-mini` | 128K | Fast reasoning |

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
   ollama pull llama3:70b
   ```

4. Run:
   ```bash
   python cogtrix.py -p ollama -m llama3:70b
   ```

### Configuration

**Local server (default):**
```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "type": "ollama",
      "model": "llama3:70b"
    }
  }
}
```

**Remote server:**
```json
{
  "provider": "ollama",
  "providers": {
    "ollama": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b"
    }
  }
}
```

### Popular Models

| Model | Size | Best For |
|-------|------|----------|
| `llama3:70b` | 70B | General purpose |
| `llama3:8b` | 8B | Fast, lightweight |
| `codellama:34b` | 34B | Code generation |
| `mistral:7b` | 7B | Fast, efficient |
| `mixtral:8x7b` | 8x7B | MoE, balanced |
| `qwen2:72b` | 72B | Multilingual |
| `deepseek-coder:33b` | 33B | Code specialized |

### Multiple Ollama Servers

```json
{
  "provider": "gpu-server",
  "providers": {
    "gpu-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b"
    },
    "cpu-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.101:11434",
      "model": "llama3:8b"
    }
  }
}
```

---

## Groq

Fast inference with open-source models.

### Setup

1. Get an API key from [console.groq.com](https://console.groq.com/keys)

2. Configure:
   ```json
   {
     "provider": "groq",
     "providers": {
       "groq": {
         "type": "openai",
         "base_url": "https://api.groq.com/openai/v1",
         "api_key": "gsk-...",
         "model": "llama-3.3-70b-versatile"
       }
     }
   }
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

2. Configure:
   ```json
   {
     "provider": "together",
     "providers": {
       "together": {
         "type": "openai",
         "base_url": "https://api.together.xyz/v1",
         "api_key": "...",
         "model": "meta-llama/Llama-3-70b-chat-hf"
       }
     }
   }
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

3. Configure:
   ```json
   {
     "provider": "vllm",
     "providers": {
       "vllm": {
         "type": "openai",
         "base_url": "http://localhost:8000/v1",
         "model": "meta-llama/Llama-3-8b-chat-hf"
       }
     }
   }
   ```

---

## Multiple Providers

Configure multiple providers for different use cases:

```json
{
  "provider": "ollama-local",

  "providers": {
    "ollama-local": {
      "type": "ollama",
      "model": "llama3:8b"
    },
    "ollama-gpu": {
      "type": "ollama",
      "base_url": "http://gpu-server:11434",
      "model": "llama3:70b"
    },
    "openai": {
      "type": "openai",
      "model": "gpt-4o"
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-...",
      "model": "llama-3.3-70b-versatile"
    }
  },

  "delegate": {
    "model_aliases": {
      "fast": "groq/llama-3.3-70b-versatile",
      "smart": "openai/gpt-4o",
      "local": "ollama-local/llama3:8b",
      "code": "ollama-gpu/codellama:34b"
    }
  }
}
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
Switched to provider openai (model: gpt-4o)

You: /p groq
Switched to provider groq (model: llama-3.3-70b-versatile)

You: /model gpt-4o-mini
Switched to model gpt-4o-mini (openai)
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
- Model name is correct (e.g., "gpt-4o" not "gpt4o")
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
ollama pull llama3:70b
```

**"Out of memory"**
```
Solutions:
- Use a smaller model (e.g., llama3:8b instead of llama3:70b)
- Close other applications
- Use quantized models (e.g., llama3:70b-q4_0)
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

- [CONFIGURATION.md](CONFIGURATION.md) — Full configuration reference
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — Delegation tools
