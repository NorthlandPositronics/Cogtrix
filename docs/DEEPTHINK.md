# Deep Think — Tree-of-Thought Reasoning Engine

Cogtrix includes a built-in reasoning engine for problems that deserve more than a single-pass answer. Instead of generating one response, Deep Think explores multiple approaches in parallel, evaluates them, and iteratively refines the best elements into a superior solution. This page covers how it works under the hood.

## Table of Contents

- [Overview](#overview)
- [Invocation Methods](#invocation-methods)
- [Architecture](#architecture)
- [The Three Phases](#the-three-phases)
- [Iteration and Beam Search](#iteration-and-beam-search)
- [Parameters](#parameters)
- [LLM Interaction](#llm-interaction)
- [Output Format](#output-format)
- [When to Use](#when-to-use)
- [How the Agent Discovers It](#how-the-agent-discovers-it)
- [Research Delegate Pre-Processing](#research-delegate-pre-processing)
- [Configuration](#configuration)
- [Internal Data Structures](#internal-data-structures)
- [Error Handling and Robustness](#error-handling-and-robustness)
- [Performance Characteristics](#performance-characteristics)

---

## Overview

Deep Think is a reasoning engine that solves complex problems by exploring multiple solution paths in parallel, evaluating each through structured reflection, and iteratively refining the best elements. It implements a combination of two established reasoning frameworks:

- **Tree-of-Thought (ToT)** — Generates and evaluates diverse approaches simultaneously, selecting the most promising branches for further exploration.
- **Chain-of-Thought with Reflection (CoT-R)** — Each approach is developed through a structured cycle: **Plan → Execute → Observe → Reflect → Revise → Retry**.

The engine is a standalone tool (`src/tools/deep_think.py`) that makes multiple LLM calls behind the scenes and returns a structured analysis report.

---

## Invocation Methods

### 1. Slash Command (direct, recommended for users)

```
/think Design a caching strategy for 50 microservices with mixed read/write workloads
```

This runs a two-stage pipeline. First, the task is classified into one of 23 categories via
`classify_think_task()`, which selects a domain-specific gather prompt. The agent then runs a Stage
1 pass using its tools to gather relevant information. Stage 2 feeds the gathered context into
`deep_think()` for structured analysis. The result is displayed in a rich panel and saved to session
memory so subsequent conversation has context.

### 2. Agent (automatic)

When the agent encounters a complex problem — one with multiple valid approaches, significant trade-offs, or a request for thorough analysis — it may invoke `deep_think` autonomously. The system prompt, tool description, and memory mode all guide this decision (see [How the Agent Discovers It](#how-the-agent-discovers-it)).

### 3. Agent (explicit)

```
You: Use deep_think to compare REST vs GraphQL vs gRPC for our API layer
```

Naming the tool explicitly guarantees the agent will use it.

---

## Architecture

The engine runs an iteration loop. Each iteration consists of three phases:

```
┌─────────────────────────────────────────────────────────┐
│                   ITERATION LOOP                         │
│                                                          │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐       │
│  │  BRANCH  │──→│  DEVELOP  │──→│   CONVERGE   │       │
│  │ (1 call) │   │ (N calls) │   │   (1 call)   │       │
│  │ Generate │   │ Plan+Exec │   │ Eval+Reflect │       │
│  │ N ideas  │   │ +Observe  │   │ +Synthesize  │       │
│  └──────────┘   └───────────┘   └──────┬───────┘       │
│                                         │               │
│       ┌─────────────────────────────────┘               │
│       │  reflection feeds next iteration                │
│       ▼                                                 │
│  Converged? ──No──→ Next iteration                      │
│       │                                                 │
│      Yes                                                │
│       │                                                 │
│       ▼                                                 │
│  FINAL SOLUTION                                         │
└─────────────────────────────────────────────────────────┘
```

**Data flow between iterations:** The CONVERGE phase produces a reflection summary (patterns observed, mistakes found, what was missed, insights gained) and a "next focus" directive. These feed into the next BRANCH phase as `prior_reflection`, steering the LLM to generate improved approaches that address gaps from previous rounds.

---

## The Three Phases

### Phase 1: BRANCH

**Goal:** Generate N fundamentally different approaches to the problem.

**LLM calls:** 1

**How it works:**

1. A structured prompt instructs the LLM to produce N diverse strategies.
2. The prompt demands diversity across methodology, perspective, and abstraction level — not just surface variation.
3. If this is iteration 2+, the reflection from the previous cycle is injected into the prompt, guiding the LLM to explore directions that were missed or underperformed.
4. The LLM returns a JSON array of approach objects.

**Each approach contains:**

| Field | Description |
|-------|-------------|
| `name` | Short descriptive title |
| `strategy` | Detailed description of the approach |
| `rationale` | Why this approach could work |
| `risks` | Potential pitfalls |

**Fallback:** If JSON parsing fails, a single "Direct approach" is created using the raw LLM output as the strategy.

### Phase 2: DEVELOP

**Goal:** Fully develop each approach through a structured Chain-of-Thought process.

**LLM calls:** N (executed in parallel via `ThreadPoolExecutor`)

**How it works:**

Each branch receives its own LLM call with instructions to work through four steps in order:

1. **Plan** — Break the approach into concrete, actionable steps.
2. **Execute** — Work through each step with full reasoning. Produce the complete solution.
3. **Observe** — Critically examine the solution: Does it fully address the task? Are there gaps or hidden assumptions?
4. **Reflect** — What went well? What was difficult? What could be improved? Rate confidence 0-10.

**Each developed branch produces:**

| Field | Description |
|-------|-------------|
| `plan` | Concrete steps |
| `execution` | Full working-out |
| `solution` | The complete solution |
| `observation` | Critical self-examination |
| `reflection` | What worked, what didn't |
| `confidence` | Self-assessed score 0-10 |
| `strengths` | List of strong points |
| `weaknesses` | List of weak points |

**Fallback:** If JSON parsing fails for a branch, the raw text is used as the solution with confidence 3.0.

### Phase 3: CONVERGE

**Goal:** Evaluate all solutions, cross-pollinate the best ideas, and synthesize an improved solution.

**LLM calls:** 1

**How it works:**

A meta-analyst prompt receives all developed solutions (with their strategies, solutions, confidence scores, strengths, weaknesses, and reflections) and performs four tasks:

1. **Evaluate** — Score each solution 0-10 on correctness, completeness, elegance, and practicality. Give a one-line verdict.
2. **Reflect** — What patterns emerge from the best solutions? What common mistakes appeared? What was missed by ALL approaches? What surprising insights emerged?
3. **Synthesize** — Combine the best elements from all solutions into a single superior solution. Address every weakness identified.
4. **Decide** — Rate confidence 0-10. Should we iterate further? If yes, what should the next iteration focus on?

**Output:**

| Field | Description |
|-------|-------------|
| `evaluations` | Per-branch scores and verdicts |
| `reflection` | Patterns, mistakes, missed elements, insights |
| `synthesis` | The combined superior solution with reasoning |
| `confidence` | Overall confidence in the synthesized solution |
| `should_continue` | Whether more iterations would help |
| `next_focus` | What the next iteration should focus on |

---

## Iteration and Beam Search

### Iteration Loop

After each CONVERGE phase, the engine decides whether to iterate:

1. **Converged** — The LLM's `should_continue` is `false`, or confidence >= 9.5 (from iteration 2 onwards). Stop.
2. **Budget exceeded** — Maximum iterations reached. Stop.
3. **Continue** — Build a reflection context from the current results and start a new BRANCH phase.

The reflection context injected into the next iteration includes:
- Previous best confidence score
- Full reflection summary
- Explicit "focus for improvement" directive

### Beam Search

After convergence, branches are ranked by their evaluation scores. Only the top `beam_width` branches are highlighted in progress output. The reflection summary carries forward the lessons from all branches, ensuring insights from lower-ranked approaches aren't lost — they influence the next iteration's branching.

---

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `task` | string | — | — | Problem to solve (required) |
| `context` | string | `""` | — | Additional context, constraints, or data |
| `max_iterations` | int | `3` | 1-5 | Maximum reflection-revision cycles |
| `num_branches` | int | `3` | 2-5 | Parallel approaches per iteration |
| `beam_width` | int | `2` | 1-3 | Best paths to keep between iterations |

**Clamping:** All numeric parameters are clamped to their valid ranges at the start of execution. `beam_width` is additionally clamped to not exceed `num_branches`.

### Parameter Tuning Guide

| Scenario | Recommended Settings |
|----------|---------------------|
| Quick exploration | `max_iterations=1, num_branches=3` |
| Balanced (default) | `max_iterations=3, num_branches=3, beam_width=2` |
| Thorough analysis | `max_iterations=5, num_branches=4, beam_width=2` |
| Maximum depth | `max_iterations=5, num_branches=5, beam_width=3` |

---

## LLM Interaction

### Prompt Structure

All three phases use structured prompts that:

1. Define the LLM's role (problem-solver, CoT executor, or meta-analyst)
2. Provide the task and optional context
3. Give explicit step-by-step instructions
4. Request JSON-only output with a specific schema

### JSON Parsing

LLM responses are parsed with a three-tier fallback strategy:

1. **Direct parse** — Try `json.loads()` on the raw response.
2. **Code fence extraction** — Search for ```` ```json ... ``` ```` or ```` ``` ... ``` ```` blocks and parse their content.
3. **Bracket matching** — Find the first balanced `{ ... }` or `[ ... ]` and parse it.

This makes the engine resilient to LLMs that wrap JSON in markdown, add explanatory text, or mix formats.

### Parallel Execution

The DEVELOP phase fires all N branch prompts simultaneously using `ThreadPoolExecutor` (effective concurrency capped at 4 by a module-level semaphore). Each call has an individual timeout, and the executor itself has a 2x timeout. Failed calls return empty strings — the engine continues with whatever branches succeeded.

### LLM Creation

Deep Think creates its own LLM instance using the same provider configuration as the main agent. It supports all four provider types (`openai`, `ollama`, `anthropic`, `google`) via the provider registry, including:
- Custom `base_url` for remote/self-hosted servers
- `api_key` for API-authenticated providers
- `context_window` for context window sizing (forwarded to Ollama as `num_ctx`)
- Configurable `temperature` (default: 0.7 for creative diversity)

---

## Output Format

The tool returns a markdown report with this structure:

```markdown
# Deep Think — Tree-of-Thought Analysis

**Task:** <the original task>

## Iteration 1

Approaches explored (3):

- **[8.5/10]** Approach A ★ — Verdict or strategy excerpt
- **[7.0/10]** Approach B — Verdict or strategy excerpt
- **[6.0/10]** Approach C — Verdict or strategy excerpt

**Reflection:** Patterns, mistakes, missed elements...

**Next focus:** What the next iteration should address

## Iteration 2
...

---
## Final Solution (confidence: 8.5/10)

The synthesized solution text, combining the best elements
from all iterations.

**Reasoning:** Why this synthesis is superior.

**Key insights:**

- Insight from improvements made
- Cross-pollinated idea
- Unexpected finding

---
*3 iterations, 9 branches explored, 142.3s elapsed*
```

The `★` marker indicates the highest-scored approach in each iteration. Branches are listed in descending score order.

---

## When to Use

### Good Candidates

- **Architecture decisions** — Microservices vs monolith, database selection, API design
- **Strategy planning** — Migration paths, technology adoption, scaling strategies
- **Complex debugging** — Intermittent failures, performance issues with unclear root cause
- **Technology comparison** — Evaluating frameworks, languages, or platforms
- **Design problems** — System design, algorithm selection, trade-off analysis
- **User requests** — "Think deeply about...", "Analyze thoroughly", "Consider all angles"

### Not Suitable For

- Simple factual questions ("What is the capital of France?")
- Quick lookups or calculations
- Straightforward tasks with one obvious approach
- Tasks that require real-time tool access (web search, file reading)

---

## How the Agent Discovers It

The agent's decision to use `deep_think` is guided by three layers:

### 1. System Prompt Guidance

The base system prompt (`src/agent/core.py`) includes a "Deep Reasoning" section that instructs the agent:

> When the user asks for deep or thorough analysis, invoke the `deep_think` tool. It explores multiple solution paths in parallel using Tree-of-Thought reasoning. Use it for architecture decisions, strategy, complex debugging, or multi-angle analysis.

### 2. Tool Description

The tool's description (visible to the LLM alongside all other tool descriptions) includes explicit trigger scenarios:

- Problem has multiple valid approaches
- User asks for thorough analysis or deep research
- Architecture or design decisions with trade-offs
- Complex debugging with unclear root cause
- Strategy or planning tasks
- User says "think step by step", "analyze thoroughly", or "consider all angles"

The description also includes a guard: "DO NOT use for simple factual questions."

### 3. Memory Mode Enhancement

In **reasoning mode** (`-M reasoning`), the system prompt receives an additional nudge:

> Use `deep_think` for decisions with significant trade-offs, complex strategy questions, or problems that benefit from exploring multiple approaches.

This makes the agent more likely to reach for deep reasoning in the mode designed for strategic planning.

---

## Research Delegate Pre-Processing

When deep thinking is triggered on a task that involved web research, the orchestrator can optionally run a **research delegate** before invoking the Deep Think engine. This addresses a key limitation: the main agent's web tool outputs are often truncated to fit the normal context budget, losing critical details like exact schemas, field names, and code examples.

### How It Works

1. The main agent performs initial web research (searches, page fetches) with the standard output cap.
2. The orchestrator detects that web tools were used (`agent_used_web_tools()`) and extracts the URLs the agent visited (`extract_fetched_urls()`).
3. A **research delegate** sub-agent is spawned with the same provider/model configuration. Its web tools are temporarily patched to allow output up to **85% of the model's context window** (configurable via `research_delegate.cap_ratio`).
4. The delegate re-fetches the URLs and is instructed to extract **verbatim specifications** — exact schemas, field names, code examples, file paths — without summarizing.
5. The delegate's structured output is passed to `force_deep_think()` as `research_context`, where it takes priority over the raw tool outputs.

### Why This Matters

Without the research delegate, Deep Think receives the same truncated web content that the main agent saw. This often leads to:
- Hallucinated field names and configuration syntax
- Generic advice instead of project-specific recommendations
- Missing code examples that were present on the original pages

With the delegate, Deep Think operates on high-fidelity data extracted directly from the source pages, producing more accurate and actionable analysis.

### Configuration

The research delegate is enabled by default and configurable via the [`research_delegate` section](CONFIGURATION.md#research-delegate-section) in your config file. Set `enabled: false` to disable it if you don't use web research with deep thinking.

---

## Configuration

Deep Think uses the active provider configuration from your config file (`.cogtrix.yaml` or `.cogtrix.json`). It creates its own LLM instance using the same provider type, base URL, model, API key, and context window settings as the main agent.

No additional configuration is required. The tool is auto-registered by the tool registry on startup and initialized via `configure_deep_think_tool()` in the main entry point.

### Provider Configuration Used

| Setting | Source | Fallback |
|---------|--------|----------|
| Provider type | `providers.<name>.type` | `"ollama"` |
| Model | Active model alias (`models.default` or CLI `-m`) | `gpt-4.1-mini` (OpenAI) / `qwen3:8b` (Ollama) |
| Base URL | `providers.<name>.base_url` | Provider defaults |
| API key | `providers.<name>.api_key` or env var (e.g. `OPENAI_API_KEY`) | — |
| Context window | `models.<alias>.context_window` | Provider default |
| Temperature | Hardcoded | `0.7` |

---

## Internal Data Structures

### ThoughtBranch

Represents a single reasoning path through all phases:

```python
@dataclass
class ThoughtBranch:
    id: str                     # e.g. "b0", "b1"
    name: str                   # Short title
    strategy: str               # Approach description
    rationale: str              # Why it could work
    risks: str                  # Potential pitfalls

    # Populated during DEVELOP (CoT)
    plan: str                   # Concrete steps
    execution: str              # Full reasoning
    solution: str               # Complete solution
    observation: str            # Critical self-examination
    reflection: str             # What worked/didn't
    confidence: float           # Self-assessed 0-10
    strengths: List[str]
    weaknesses: List[str]

    # Populated during CONVERGE
    score: float                # Evaluator score 0-10
    verdict: str                # One-line evaluation
```

### IterationResult

Captures the complete outcome of one iteration cycle:

```python
@dataclass
class IterationResult:
    iteration: int
    branches: List[ThoughtBranch]
    best_solution: str          # Synthesized solution
    synthesis_reasoning: str    # Why this synthesis is best
    confidence: float           # Overall 0-10
    reflection_summary: str     # Patterns, mistakes, insights
    insights: List[str]         # Key takeaways
    should_continue: bool       # LLM's recommendation
    next_focus: str             # What to improve next
```

---

## Error Handling and Robustness

The engine is designed to degrade gracefully rather than fail:

| Failure | Recovery |
|---------|----------|
| LLM call timeout | Returns empty string; branch skipped |
| JSON parse failure (BRANCH) | Creates single fallback "Direct approach" branch |
| JSON parse failure (DEVELOP) | Uses raw LLM output as solution with confidence 3.0 |
| JSON parse failure (CONVERGE) | Uses raw text as synthesis with confidence 3.0 |
| Parallel call failure | Only failed branches return empty; rest proceed |
| LLM creation failure | Returns error message immediately |
| Missing configuration | Returns error message immediately |

All LLM call failures are logged at WARNING level via the `cogtrix` logger.

---

## Performance Characteristics

### LLM Call Budget

| Metric | Formula | Defaults (3 iter, 3 branches) |
|--------|---------|-------------------------------|
| Calls per iteration | N + 2 | 5 |
| Max total calls | iterations × (N + 2) | 15 |
| Min total calls | 1 × (N + 2) | 5 (if converges on first iteration) |

### Timing

| Model Type | Per-iteration | Total (3 iterations) |
|------------|---------------|----------------------|
| Cloud API (GPT-4o) | 15-30s | 45-90s |
| Fast cloud (GPT-4o-mini) | 5-15s | 15-45s |
| Local Ollama (70B) | 30-90s | 90-270s |
| Local Ollama (8B) | 10-30s | 30-90s |

### Concurrency

The DEVELOP phase runs up to 4 parallel LLM calls (limited by a module-level semaphore shared across all sessions). For Ollama providers, actual parallelism depends on the server's `OLLAMA_NUM_PARALLEL` setting (default: 1 for most models). Cloud APIs generally handle parallel requests well.

---

## See Also

- [Configuration Reference](CONFIGURATION.md) — `research_delegate` and deep thinking settings
- [Tools Reference](TOOLS_REFERENCE.md) — `deep_think` and `delegate_task` tool parameters
- [Architecture Overview](ARCHITECTURE.md) — research delegate pipeline
