# Cogtrix Memory Modes

Cogtrix manages what the LLM "remembers" during a session. Different tasks benefit from different memory strategies — a quick Q&A session doesn't need error tracking, and a planning session benefits from decision logging. Memory modes let you pick the right strategy for the job.

**Not sure which mode to use?** Start with `conversation` (the default). Switch to `code` when you start writing or debugging code, and to `reasoning` when you need to plan, compare options, or make decisions.

## Table of Contents

- [Overview](#overview)
- [Hybrid Memory System](#hybrid-memory-system)
- [Message Timestamps](#message-timestamps)
- [Mode Comparison](#mode-comparison)
- [Conversation Mode](#conversation-mode)
- [Code Development Mode](#code-development-mode)
- [Reasoning Mode](#reasoning-mode)
- [Configuration](#configuration)
- [Switching Modes](#switching-modes)

---

## Overview

Cogtrix uses a pluggable memory system that optimizes context management for different use cases. Each mode manages:

- **Working Memory** — Recent messages sent to the LLM (sliding window)
- **Hybrid Memory** — Automatic summarization and optional semantic recall of older messages
- **Context Tracking** — Mode-specific information (files, decisions, etc.)
- **System Prompt Additions** — Mode-specific instructions for the LLM
- **Token-Aware Trimming** — Ensures the context always fits the model's context window

```
┌─────────────────────────────────────────────────────────────────┐
│                      Memory Factory                              │
│                   create(mode, store, ...)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Conversation  │     │     Code      │     │   Reasoning   │
│   (25 msgs)   │     │  (30 msgs)    │     │  (30 msgs)    │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Hybrid Memory    │
                    │  (all modes)      │
                    │  • Summary        │
                    │  • Vector Recall  │
                    └───────────────────┘
```

---

## Hybrid Memory System

All three memory modes share a **hybrid memory layer** that prevents long-term context loss. When messages fall outside the sliding window, they are not simply discarded — they are processed in two ways:

1. **Incremental summarization** — An LLM generates a concise rolling summary of older messages, preserving key facts, decisions, and user preferences.
2. **Vector recall** (optional) — Older message pairs are embedded and stored in a per-session FAISS index. On each turn, the user's input is used to retrieve the most semantically relevant past exchanges.

Both layers are injected at the top of the context, giving the LLM a sense of the full conversation history without consuming the entire context window.

### How It Works

```
Full conversation (e.g. 80 messages over time)
│
├─ Messages 1–44  → Covered by rolling summary (compressed text)
│                   + stored in vector index for semantic recall
│
├─ Messages 45–55  → Pending batch (will be summarized when ≥ 10 accumulate)
│
└─ Messages 56–80  → Sliding window (sent verbatim to the LLM)


What the LLM actually sees on each turn:
┌─────────────────────────────────────────────────────────┐
│  System Prompt + Mode-specific additions                │
├─────────────────────────────────────────────────────────┤
│  Conversation summary (older context):                  │
│  • The user asked about Python web frameworks...        │
│  • They decided to use FastAPI with PostgreSQL...       │
├─────────────────────────────────────────────────────────┤
│  Related past exchanges (vector recall):                │
│  User: How should I structure the database schema?      │
│  Assistant: For your e-commerce project, I recommend... │
├─────────────────────────────────────────────────────────┤
│  [Sliding window: last 25/30 messages verbatim]      │
│  [2026-02-14 15:23:05 UTC] Human: "..."                 │
│  [2026-02-14 15:23:12 UTC] AI: "..."                    │
│  Human: "..." ← Current input                           │
└─────────────────────────────────────────────────────────┘
```

### Summarization

Summarization is triggered **after each response**, not during the user's wait for a reply. Specifically:

1. After the agent replies, the memory manager checks how many messages have fallen outside the sliding window since the last summary was generated.
2. If unsummarized messages have accumulated, a meaningful-content gate runs before sending anything to the LLM:
   - At least **4 meaningful messages** (2 full human+assistant turns) must be present (`_MIN_MEANINGFUL_MSGS_FOR_SUMMARY = 4`)
   - At least **5,000 characters** of meaningful content must exist (`_MIN_MEANINGFUL_CHARS_FOR_SUMMARY = 5000`)
   - Both thresholds must be missed simultaneously to skip summarization — if either is met, the batch proceeds. This prevents summarization from firing on short or tool-heavy exchanges that contain no real conversational substance.
3. The LLM produces an updated rolling summary that merges the new batch into the existing summary.
4. The summary index is advanced so those messages aren't re-summarized.

The summarization prompt instructs the LLM to:
- Preserve key facts, data, decisions, user preferences, and action items
- Drop small-talk, greetings, and verbose tool-call details
- Write in third person present tense
- Keep the summary under 400 words
- Use bullet points for clarity

**Graceful degradation:** If the LLM call fails or returns an empty result, the previous summary is retained unchanged. Summarization never blocks or crashes the conversation.

### Vector Recall

When an embedding provider is available (Ollama with `nomic-embed-text`, OpenAI, etc.), Cogtrix automatically:

1. Embeds older conversation exchanges (human + AI pairs) into a per-session FAISS index.
2. On each new user input, queries the index for the top-k most similar past exchanges.
3. Injects the recalled exchanges into the context as "Related past exchanges."

This allows the agent to recall specific details from much earlier in the conversation — even details that the rolling summary may have compressed away.

**Graceful degradation:** If no embedding provider is available, vector recall is simply skipped. The sliding window and rolling summary still function normally.

**Configuring embeddings:** Cogtrix auto-detects an embedding provider at startup (tries Ollama first, then OpenAI). To explicitly control which embedding model is used for hybrid memory, define a model entry in the [`models` registry](CONFIGURATION.md#models) and configure `rag.model` to reference it — the same model is used for both RAG ingestion and memory vector recall.

**Embedding model tracking:** The embedding model name is stored alongside the FAISS index. If you switch embedding models between sessions, the stale index is automatically discarded and rebuilt from scratch.

### Configuration

Hybrid memory is enabled by default. You can tune it per mode:

```yaml
memory:
  modes:
    conversation:
      working_memory_size: 25
      summarization: true        # Enable/disable LLM summarization (default: true)
      vector_recall_k: 3         # Number of past exchanges to recall (default: 3)
    code:
      working_memory_size: 30
      summarization: true
      vector_recall_k: 3
    reasoning:
      working_memory_size: 30
      summarization: true
      vector_recall_k: 3
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `summarization` | bool | `true` | Enable incremental LLM summarization of older messages |
| `vector_recall_k` | int | `3` | Number of semantically similar past exchanges to retrieve |

Setting `summarization: false` disables the rolling summary (useful if you want to save LLM calls on a metered API). Setting `vector_recall_k: 0` effectively disables vector recall.

### Persistence

Hybrid memory state is persisted alongside session history:

- **Summary text + coverage index** → `data/history/{session_id}_hybrid.json`
- **Vector index** → `data/vectordb/sessions/{session_id}/` (FAISS files + metadata)

When you resume a session, both the summary and vector index are restored. If the session history was sanitized (e.g., corrupted messages removed), the summary index is automatically clamped to stay within bounds.

---

## Message Timestamps

Every message in the conversation history is automatically stamped with a **UTC timestamp** at the moment it is created:

- **User messages** are stamped when the input is submitted (at `prepare_context()` time).
- **AI responses** are stamped when the LLM finishes generating its reply (at `update()` time).

When messages are sent to the LLM, each one is prefixed with a human-readable timestamp:

```
[2026-02-14 15:23:05 UTC] What are the top news affecting the stock market?
[2026-02-14 15:23:47 UTC] Here are the top stories...
```

This gives the model a sense of time: it can see how long a response took, how much time passed between turns, and whether a session spans minutes or days. Timestamps are stored in UTC for unambiguous cross-timezone comparison.

**Persistence:** Timestamps are saved alongside each message in the session JSON file (as an ISO 8601 string, e.g. `"2026-02-14T15:23:05Z"`). Old session files without timestamps load normally — those messages simply appear without a time prefix.

---

## Mode Comparison

| Aspect | Conversation | Code | Reasoning |
|--------|--------------|------|-----------|
| **Working Memory** | 25 messages | 30 messages | 30 messages |
| **Best For** | General chat, Q&A | Programming, debugging | Planning, decisions |
| **Tracks** | Topics, entities | Files, errors, changes | Goals, decisions, constraints |
| **Context Focus** | Conversation flow | Current code + task | Problem + objectives |
| **Hybrid Memory** | Summary + vector recall | Summary + vector recall | Summary + vector recall |

**About tools:** All tools are on-demand regardless of memory mode — the agent requests only the tools it needs for the current task. See [Tool Loading](CONFIGURATION.md#tool-loading) for details.

---

## Conversation Mode

**CLI:** `python cogtrix.py -M conversation` (default)

**Best for:** General chat, Q&A, research, information lookup

### How It Works

Maintains a sliding window of recent messages with entity tracking:

```
┌────────────────────────────────────────────────────────────────┐
│                    CONVERSATION MEMORY                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Hybrid Prefix (injected when available)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Summary: "The user discussed Python frameworks..."      │  │
│  │  Related: [vector-recalled past exchanges]               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Working Memory (Last 25 messages, timestamped)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [2026-02-14 15:23:05 UTC] Human: "What is Python?"     │  │
│  │  [2026-02-14 15:23:12 UTC] AI: "Python is a ..."        │  │
│  │  [2026-02-14 15:23:40 UTC] Human: "How do I install?"   │  │
│  │  [2026-02-14 15:23:48 UTC] AI: "You can download..."    │  │
│  │  ... (up to 25 messages)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Entity Tracking                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Topics: [Python, installation, programming]             │  │
│  │  Key Facts: [user wants to learn Python]                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Context Composition

What gets sent to the LLM:

```
┌──────────────────────────────────────────────────┐
│ System Prompt                                    │
│ "You are a helpful AI assistant..."              │
├──────────────────────────────────────────────────┤
│ Hybrid Prefix (summary + recalled exchanges)     │
├──────────────────────────────────────────────────┤
│ Working Memory (Last 25 messages, timestamped)   │
│   [2026-02-14 15:23:05 UTC] Human: "..."         │
│   [2026-02-14 15:23:12 UTC] AI: "..."            │
│   Human: "..." ← Current input                   │
└──────────────────────────────────────────────────┘
```

### Configuration

```yaml
memory:
  mode: conversation
  modes:
    conversation:
      working_memory_size: 25
      summarization: true
      vector_recall_k: 3
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 25 | Number of messages to keep in context |
| `summarization` | `true` | Enable rolling summary of older messages |
| `vector_recall_k` | 3 | Semantically similar past exchanges to retrieve |

---

## Code Development Mode

**CLI:** `python cogtrix.py -M code`

**Best for:** Programming, debugging, code review, software development

### How It Works

Optimized for coding with task and file tracking:

```
┌────────────────────────────────────────────────────────────────┐
│                   CODE DEVELOPMENT MEMORY                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Hybrid Prefix (summary + vector recall)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Summary: "Working on auth module refactor..."           │  │
│  │  Related: [past exchanges about auth.py]                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Working Memory (Last 30 messages, timestamped)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [10:05:30 UTC] Human: "Fix the bug in auth.py"         │  │
│  │  [10:05:47 UTC] AI: "I see the issue..."                │  │
│  │  ... (up to 30 messages)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Task Context                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Current Task: "Fix authentication bug"                  │  │
│  │  Progress: ["Identified issue", "Modified auth.py"]      │  │
│  │  Files Touched: [auth.py, tests/test_auth.py]            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Error Tracking                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Recent Errors:                                          │  │
│  │  - TypeError at auth.py:45                               │  │
│  │  - ImportError in test_auth.py                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Context Composition

What gets sent to the LLM:

```
┌────────────────────────────────────────┐
│ System Prompt                          │
│ "You are an expert programmer..."      │
├────────────────────────────────────────┤
│ Hybrid Prefix (summary + recall)       │
├────────────────────────────────────────┤
│ Task Context                           │
│ "Current task: Fix authentication bug  │
│  Files: auth.py, test_auth.py          │
│  Recent errors: TypeError at line 45"  │
├────────────────────────────────────────┤
│ Working Memory (Last 30 messages)      │
│   [10:05:30 UTC] Human: "..."          │
│   [10:05:47 UTC] AI: "..."             │
│   Human: "..." ← Current input         │
└────────────────────────────────────────┘
```

### Special Features

1. **File Tracking** — Automatically tracks mentioned files
2. **Error Memory** — Retains error messages for debugging context
3. **Task Progress** — Tracks what's been accomplished
4. **Structured Context** — Task, files, and errors injected alongside messages

### Configuration

```yaml
memory:
  mode: code
  modes:
    code:
      working_memory_size: 30
      max_files: 20
      max_errors: 5
      summarization: true
      vector_recall_k: 3
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 30 | Number of messages to keep |
| `max_files` | 20 | Maximum files to track |
| `max_errors` | 5 | Maximum errors to remember |
| `summarization` | `true` | Enable rolling summary of older messages |
| `vector_recall_k` | 3 | Semantically similar past exchanges to retrieve |

---

## Reasoning Mode

**CLI:** `python cogtrix.py -M reasoning`

**Best for:** Strategic planning, architecture decisions, complex problem-solving

### How It Works

Designed for deep thinking with goal and decision tracking:

```
┌────────────────────────────────────────────────────────────────┐
│                     REASONING MEMORY                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Hybrid Prefix (summary + vector recall)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Summary: "Evaluating microservices architecture..."     │  │
│  │  Related: [recalled constraint discussion]               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Working Memory (Last 30 messages, timestamped)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [09:00:15 UTC] Human: "Should we use microservices?"   │  │
│  │  [09:01:03 UTC] AI: "Let me analyze the trade-offs..."  │  │
│  │  ... (up to 30 messages)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Goal Hierarchy                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Primary Objective: "Design scalable architecture"       │  │
│  │  Sub-goals:                                              │  │
│  │    ├── Evaluate service patterns                         │  │
│  │    ├── Consider team capabilities                        │  │
│  │    └── Plan migration strategy                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Decision Log                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  #1: Use event-driven architecture                       │  │
│  │      Rationale: Better decoupling, async processing      │  │
│  │      Alternatives rejected: Direct API calls             │  │
│  │                                                          │  │
│  │  #2: Start with monolith, extract services later         │  │
│  │      Rationale: Team size, time constraints              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Constraints                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Budget: $50k                                          │  │
│  │  - Timeline: 3 months                                    │  │
│  │  - Team: 4 developers                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Context Composition

What gets sent to the LLM:

```
┌────────────────────────────────────────┐
│ System Prompt                          │
│ "You are a strategic advisor..."       │
├────────────────────────────────────────┤
│ Hybrid Prefix (summary + recall)       │
├────────────────────────────────────────┤
│ Goal Hierarchy                         │
│ "Objective: Design scalable arch       │
│  Sub-goals: [list]                     │
│  Current phase: Evaluation"            │
├────────────────────────────────────────┤
│ Constraints                            │
│ "Budget: $50k, Timeline: 3 months..."  │
├────────────────────────────────────────┤
│ Recent Decisions                       │
│ "#1: Use event-driven - Rationale:..." │
├────────────────────────────────────────┤
│ Working Memory (Last 30 messages)      │
│   [09:00:15 UTC] Human: "..."          │
│   [09:01:03 UTC] AI: "..."             │
└────────────────────────────────────────┘
```

### Special Features

1. **Goal Tracking** — Maintains objective hierarchy
2. **Decision Audit** — Logs decisions with rationale
3. **Constraint Awareness** — Keeps boundaries visible
4. **Alternative Tracking** — Records rejected options
5. **Assumption Logging** — Explicit assumption tracking

### Configuration

```yaml
memory:
  mode: reasoning
  modes:
    reasoning:
      working_memory_size: 30
      max_decisions: 20
      max_alternatives: 10
      summarization: true
      vector_recall_k: 3
      prefix_max_stale_turns: 3  # Turns before a stale section is omitted from prefix
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 30 | Number of messages to keep |
| `max_decisions` | 20 | Maximum decisions to track |
| `max_alternatives` | 10 | Maximum alternatives to track |
| `summarization` | `true` | Enable rolling summary of older messages |
| `vector_recall_k` | 3 | Semantically similar past exchanges to retrieve |
| `prefix_max_stale_turns` | 3 | Turns a prefix section can go unmodified before being omitted from the context prefix (section-freshness gating) |

---

## Configuration

### Via Config File

```yaml
memory:
  mode: code
  modes:
    conversation:
      working_memory_size: 25
      summarization: true
      vector_recall_k: 3
    code:
      working_memory_size: 30
      summarization: true
      vector_recall_k: 3
    reasoning:
      working_memory_size: 30
      summarization: true
      vector_recall_k: 3
```

### Via Environment Variable

```bash
export COGTRIX_MEMORY_MODE=code
python cogtrix.py
```

### Via Command Line

```bash
python cogtrix.py -M code
python cogtrix.py --memory-mode reasoning
```

---

## Switching Modes

### At Runtime (Live Switching)

Switch modes during an interactive session using the `/mode` or `/M` command:

```
You: /mode code
Switched to code mode

You: /M reasoning
Switched to reasoning mode
```

Switching preserves the current session but rebuilds the system prompt, memory context, and tool presets for the new mode. The agent is re-initialized immediately.

### At Startup

Specify a mode when starting:

```bash
# Morning: Planning session
python cogtrix.py -M reasoning -s project-planning

# Afternoon: Coding session
python cogtrix.py -M code -s project-dev

# Evening: Research session
python cogtrix.py -M conversation -s research
```

### Mode Selection Guide

| If you're doing... | Use mode | Why |
|--------------------|----------|-----|
| General questions, research | `conversation` | Lightweight, fast — no extra overhead |
| Summarizing articles, brainstorming | `conversation` | Focus on the flow of ideas |
| Writing or reviewing code | `code` | Tracks files you mention and errors you hit |
| Debugging errors | `code` | Error memory prevents the LLM from losing context on the bug |
| Refactoring a codebase | `code` | Larger working memory (30 msgs) keeps more context visible |
| Architecture decisions | `reasoning` | Decision log records choices and rationale |
| Project planning | `reasoning` | Goal hierarchy keeps objectives structured |
| Comparing options with trade-offs | `reasoning` | Constraint tracking + deep think integration |

**Rule of thumb:** conversation < code < reasoning in terms of working memory size and tracking overhead. Pick the lightest mode that fits your task.

---

## Memory Persistence

All modes save to the same JSON format:

```
data/history/{session_id}.json              ← Message history + session metadata
data/history/{session_id}_hybrid.json       ← Summary text + coverage index
data/history/{session_id}_mode_state.json   ← Mode-specific state (goals, decisions, etc.)
data/vectordb/sessions/{session_id}/        ← FAISS vector index (if embeddings available)
```

The history file contains:
- Full message history (each message includes a UTC `timestamp` field)
- Session metadata

The mode state file contains mode-specific tracking data (goals, decisions, reasoning chains, code tasks, conversation entities, turn counters, and section timestamps) persisted via `_save_mode_meta()` and restored on session restart via `_restore_mode_state()`.

Memory is automatically loaded when resuming a session:

```bash
# First session
python cogtrix.py -M code -s my-project
# ... work on code ...
# Exit

# Resume later (memory restored — including summary and vector index)
python cogtrix.py -M code -s my-project
```

---

## Token-Aware Context Management

Regardless of the memory mode, Cogtrix ensures the prepared context never exceeds the model's context window. Before messages are sent to the LLM:

1. The total token count is estimated using a character-based heuristic (~4 characters per token).
2. If the total exceeds the available budget, the oldest history messages are dropped first.
3. If individual messages are still too large after trimming, they are truncated with a `[…truncated…]` marker.
4. The system prompt and the current user input are never removed.

The `max_tokens` parameter sent to the LLM is also dynamically calculated to avoid requesting more tokens than the remaining context window allows, preventing "max_tokens must be at least 1" errors from the API.

---

## See Also

- [Configuration Reference](CONFIGURATION.md) — memory and summarization settings
- [Architecture Overview](ARCHITECTURE.md) — how memory fits in the execution pipeline
