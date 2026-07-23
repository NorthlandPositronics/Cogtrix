# Cogtrix Memory Modes

Detailed documentation of the modular memory management system.

## Table of Contents

- [Overview](#overview)
- [Mode Comparison](#mode-comparison)
- [Conversation Mode](#conversation-mode)
- [Code Development Mode](#code-development-mode)
- [Reasoning Mode](#reasoning-mode)
- [Configuration](#configuration)
- [Switching Modes](#switching-modes)

---

## Overview

Cogtrix uses a pluggable memory system that optimizes context management for different use cases. Each mode manages:

- **Working Memory** — Recent messages sent to the LLM
- **Context Tracking** — Mode-specific information (files, decisions, etc.)
- **System Prompt Additions** — Mode-specific instructions for the LLM

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
│    (20 msgs)  │     │   (8 msgs)    │     │   (6 msgs)    │
└───────────────┘     └───────────────┘     └───────────────┘
```

---

## Mode Comparison

| Aspect | Conversation | Code | Reasoning |
|--------|--------------|------|-----------|
| **Working Memory** | 20 messages | 8 messages | 6 messages |
| **Best For** | General chat, Q&A | Programming, debugging | Planning, decisions |
| **Tracks** | Topics, entities | Files, errors, changes | Goals, decisions, constraints |
| **Context Focus** | Conversation flow | Current code + task | Problem + objectives |

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
│  Working Memory (Last 20 messages)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Human: "What is Python?"                                │  │
│  │  AI: "Python is a programming language..."               │  │
│  │  Human: "How do I install it?"                           │  │
│  │  AI: "You can download Python from..."                   │  │
│  │  ... (up to 20 messages)                                 │  │
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
┌────────────────────────────────────────┐
│ System Prompt                          │
│ "You are a helpful AI assistant..."    │
├────────────────────────────────────────┤
│ Working Memory (Last 20 messages)      │
│   Human: "..."                         │
│   AI: "..."                            │
│   Human: "..." ← Current input         │
└────────────────────────────────────────┘
```

### Configuration

```json
{
  "memory": {
    "mode": "conversation",
    "modes": {
      "conversation": {
        "working_memory_size": 20
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 20 | Number of messages to keep in context |

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
│  Working Memory (Last 8 messages)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Human: "Fix the bug in auth.py"                         │  │
│  │  AI: "I see the issue. The token validation..."          │  │
│  │  ... (up to 8 messages)                                  │  │
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
│ Task Context                           │
│ "Current task: Fix authentication bug  │
│  Files: auth.py, test_auth.py          │
│  Recent errors: TypeError at line 45"  │
├────────────────────────────────────────┤
│ Working Memory (Last 8 messages)       │
│   Human: "..."                         │
│   AI: "..."                            │
│   Human: "..." ← Current input         │
└────────────────────────────────────────┘
```

### Special Features

1. **File Tracking** — Automatically tracks mentioned files
2. **Error Memory** — Retains error messages for debugging context
3. **Task Progress** — Tracks what's been accomplished
4. **Concise Context** — Smaller window to leave room for code

### Configuration

```json
{
  "memory": {
    "mode": "code",
    "modes": {
      "code": {
        "working_memory_size": 8,
        "max_files": 20,
        "max_errors": 10
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 8 | Number of messages to keep |
| `max_files` | 20 | Maximum files to track |
| `max_errors` | 10 | Maximum errors to remember |

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
│  Working Memory (Last 6 messages)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Human: "Should we use microservices?"                   │  │
│  │  AI: "Let me analyze the trade-offs..."                  │  │
│  │  ... (up to 6 messages)                                  │  │
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
│ Goal Hierarchy                         │
│ "🎯 OBJECTIVE: Design scalable arch    │
│  Sub-goals: [list]                     │
│  Current phase: Evaluation"            │
├────────────────────────────────────────┤
│ Constraints                            │
│ "Budget: $50k, Timeline: 3 months..."  │
├────────────────────────────────────────┤
│ Recent Decisions                       │
│ "#1: Use event-driven - Rationale:..." │
├────────────────────────────────────────┤
│ Working Memory (Last 6 messages)       │
│   Human: "..."                         │
│   AI: "..."                            │
└────────────────────────────────────────┘
```

### Special Features

1. **Goal Tracking** — Maintains objective hierarchy
2. **Decision Audit** — Logs decisions with rationale
3. **Constraint Awareness** — Keeps boundaries visible
4. **Alternative Tracking** — Records rejected options
5. **Assumption Logging** — Explicit assumption tracking

### Configuration

```json
{
  "memory": {
    "mode": "reasoning",
    "modes": {
      "reasoning": {
        "working_memory_size": 6,
        "max_decisions": 20,
        "max_goals": 10
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `working_memory_size` | 6 | Number of messages to keep |
| `max_decisions` | 20 | Maximum decisions to track |
| `max_goals` | 10 | Maximum goals to track |

---

## Configuration

### Via Config File

```json
{
  "memory": {
    "mode": "code",
    "modes": {
      "conversation": { "working_memory_size": 20 },
      "code": { "working_memory_size": 8 },
      "reasoning": { "working_memory_size": 6 }
    }
  }
}
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

| If you're doing... | Use mode |
|--------------------|----------|
| General questions, research | `conversation` |
| Writing or reviewing code | `code` |
| Debugging errors | `code` |
| Architecture decisions | `reasoning` |
| Project planning | `reasoning` |
| Analyzing trade-offs | `reasoning` |

---

## Memory Persistence

All modes save to the same JSON format:

```
data/history/{session_id}.json
```

The file contains:
- Full message history
- Mode-specific tracking data
- Session metadata

Memory is automatically loaded when resuming a session:

```bash
# First session
python cogtrix.py -M code -s my-project
# ... work on code ...
# Exit

# Resume later (memory restored)
python cogtrix.py -M code -s my-project
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) — Full configuration reference
- [ARCHITECTURE.md](ARCHITECTURE.md) — System internals
