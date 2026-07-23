# Cogtrix Tools Reference

Complete documentation of all 72 built-in tools. You don't need to memorize these — the agent picks the right tool automatically based on your request. This page is a reference for when you want to know exactly what's available, what parameters a tool accepts, or how to configure optional providers.

**Quick orientation:**

- **DuckDuckGo search** works immediately — no setup, no API key.
- **File, shell, Python, text, JSON, date/time, HTTP, NLP, delegation, and deep reasoning tools** are always available.
- **Premium search** (Tavily, Exa, Brave, Google, SerpAPI), **weather**, **WhatsApp**, and **Telegram** tools appear automatically when you provide their API key or token. If the key is missing, the tool is silently hidden — no errors.
- Tools marked with a warning sign require user confirmation before running.

## Table of Contents

- [Overview](#overview)
- [System Tools](#system-tools)
- [File Operations](#file-operations)
- [Git Operations](#git-operations)
- [Math & Calculation](#math--calculation)
- [Date & Time](#date--time)
- [Text Processing](#text-processing)
- [JSON Processing](#json-processing)
- [Search](#search)
- [Web & HTTP](#web--http)
- [Weather](#weather)
- [NLP Tools](#nlp-tools)
- [WhatsApp Messaging](#whatsapp-messaging)
- [Telegram Messaging](#telegram-messaging)
- [Scheduling](#scheduling)
- [Knowledge Base](#knowledge-base)
- [Delegation](#delegation)
- [Deep Reasoning](#deep-reasoning)

---

## Overview

### Safety Categories

| Category | Confirmation | Examples |
|----------|--------------|----------|
| **Safe** | No | `read_file`, `calculate`, `search_web` |
| **Sensitive** | Yes | `execute_shell_command`, `write_file`, `execute_python` |

### Confirmation Responses

When prompted for confirmation:
- `y` — Yes, allow this execution once
- `n` — No, deny execution
- `a` — Allow all (approve tool for entire session)
- `d` — Disable tool (block for this session)
- `f` — Forbid all further tool requests (resets on next prompt)
- `c` — Cancel the current agent workflow

### On-Demand Loading

All tools start in an on-demand pool. The agent requests only the tools it needs for the current task through an internal `request_tools` meta-tool. When the agent calls `request_tools`, the requested tools are activated before the next turn. The agent can also release tools it no longer needs.

Agent-loaded tools are automatically unloaded at the start of each new prompt cycle so the LLM starts fresh. Tools loaded manually via `/tools load`, `--activate-tools`, or the API are **pinned** and persist across prompt cycles until explicitly unloaded with `/tools unload`.

You don't need to manage this yourself — the agent decides which tools to load based on your request. See [Tool Loading](CONFIGURATION.md#tool-loading) for details.

---

## System Tools

### execute_shell_command ⚠️

Execute shell commands with timeout protection.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cmd` | string | Yes | Shell command to execute |
| `working_directory` | string | No | Working directory (default: current) |
| `timeout` | int | No | Timeout in seconds (default: 30) |

**Example:**
```
Execute: ls -la /home/user
```

**Returns:** Command output (stdout + stderr) and exit code

---

### execute_python ⚠️

Execute Python code in a restricted environment with persistent state.

**Requires Confirmation:** Yes

**Features:**
- **Persistent variables** — Variables preserved between calls within a session
- **REPL-style output** — Last expression value automatically displayed
- **True timeout** — Enforced via subprocess isolation
- **Execution history** — Track past executions with `%history`
- **Optional NumPy/Pandas** — Automatically enabled if installed
- **Special commands** — `%vars`, `%clear`, `%history`, `%modules`, `%help`

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | — | Python code to execute |
| `timeout` | int | No | 30 | Timeout in seconds (max: 60) |
| `persistent` | bool | No | true | Persist variables between calls |
| `session_id` | string | No | `"default"` | Session identifier for state isolation |

**Special Commands:**

| Command | Description |
|---------|-------------|
| `%vars` | List all stored variables with types |
| `%clear` | Clear all variables |
| `%reset` | Same as `%clear` |
| `%history N` | Show last N executions (default: 10) |
| `%modules` | Show available modules |
| `%help` | Show available commands |

**Core Modules:**

`math`, `random`, `string`, `re`, `json`, `datetime`, `collections`, `itertools`, `functools`, `operator`, `statistics`, `decimal`, `fractions`, `csv`, `dataclasses`, `enum`, `uuid`, `typing`, `base64`, `hashlib`, `textwrap`, `time`, `cmath`, `bisect`, `heapq`

**Optional Modules (if installed):**

`numpy`, `pandas`, `scipy` — Automatically available if installed on the system.

**Security Limits:**

| Limit | Value | Description |
|-------|-------|-------------|
| Max output | 10,000 chars | Output truncated with warning |
| Max result | 2,000 chars | Result repr truncated |
| Max loop iterations | 100,000 | Prevents infinite loops |
| Max recursion depth | 100 | Prevents stack overflow |
| Max range size | 100,000 | Large ranges blocked |
| Max collection size | 10,000 | Large lists/dicts/sets limited |
| History entries | 50 | Per session |

**Security Features:**
- **AST Analysis** — Deep inspection blocks dangerous attribute access
- **Loop Limiting** — Automatic iteration counter injection
- **Recursion Control** — Custom depth limit per subprocess
- **Size Guards** — Prevents memory exhaustion attacks

**Restrictions:**
- No file system access (`open`, `pathlib`)
- No network access (`socket`, `urllib`, `requests`)
- No system commands (`os`, `sys`, `subprocess`)
- No dangerous builtins (`eval`, `exec`, `compile`)
- No dangerous attributes (`__class__`, `__bases__`, `__subclasses__`, `__globals__`)

**Examples:**

*Multi-step computation with persistent state:*
```
Call 1: data = [1, 2, 3, 4, 5]
→ [Variables: data]

Call 2: avg = sum(data) / len(data)
→ Result: 3.0
  [Variables: avg, data]

Call 3: print(f"Average: {avg}")
→ Average: 3.0
```

*REPL-style expression evaluation:*
```python
2 ** 10
```
→ `Result: 1024`

*Using allowed modules:*
```python
import math
math.factorial(20)
```
→ `Result: 2432902008176640000`

*View variables with types:*
```
%vars
```
→
```
Variables:
  avg: float = 3.0
  data: list = [1, 2, 3, 4, 5]
```

*View execution history:*
```
%history 5
```
→
```
Last 5 execution(s):
  1. [10:15:01] ✓ data = [1, 2, 3, 4, 5]
  2. [10:15:05] ✓ avg = sum(data) / len(data)
  3. [10:15:10] ✓ 2 ** 10
  4. [10:15:15] ✗ undefined_var
  5. [10:15:20] ✓ %vars
```

---

## File Operations

All file tools enforce path safety: paths must resolve within the current working directory. Read-only tools (`read_file`, `list_directory`, `file_info`) also allow access to the application install directory — this matters in Docker containers where the working directory may differ from the install path (e.g., `-w /tmp` while the app lives at `/app`). Write tools are restricted to the working directory by default, but additional write directories can be allowed via `--allow-write-path`, `COGTRIX_ALLOWED_WRITE_PATHS`, or the `allowed_write_paths` config option (see [Configuration](CONFIGURATION.md#allowed-write-paths)).

### read_file

Read contents of a file.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File path to read |
| `encoding` | string | No | File encoding (default: "utf-8") |
| `start_line` | int | No | Line number to start reading from (0-based) |
| `max_lines` | int | No | Maximum number of lines to read from start_line |

**Example:**
```
Read file: /path/to/config.json
```

---

### write_file ⚠️

Write content to a file (creates if not exists, overwrites if it does).

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File path to write (must be within working directory) |
| `content` | string | Yes | Content to write |
| `encoding` | string | No | File encoding (default: "utf-8") |

---

### patch_file ⚠️

Surgically replace an exact string in a file without rewriting the whole file.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File path to edit (must be within an allowed write root) |
| `old_str` | string | Yes | Exact string to find (must appear exactly once in the file) |
| `new_str` | string | Yes | Replacement string (may be empty to delete the match) |

**Notes:**

- Fails with a clear error if `old_str` is not found or appears more than once — add more surrounding context to make the match unique.
- Writes the result atomically; the file is never left in a partial state.
- Prefer this over `write_file` for targeted edits to existing files.

---

### append_file ⚠️

Append content to an existing file.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File path to append to |
| `content` | string | Yes | Content to append |
| `encoding` | string | No | File encoding (default: "utf-8") |

---

### list_directory

List contents of a directory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Directory path |
| `pattern` | string | No | Glob pattern to filter files (default: "*") |
| `show_hidden` | bool | No | Include hidden files (default: false) |

**Returns:** List of files/directories with sizes

---

### file_info

Get detailed information about a file or directory.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | File or directory path |

**Returns:** Size, creation date, modification date, permissions

---

## Git Operations

Run Git commands from within Cogtrix to inspect and manage a repository. Read-only tools (`git_status`, `git_diff`, `git_log`) run without confirmation. Write tools (`git_add`, `git_commit`, `git_create_branch`, `git_checkout`) require confirmation. All commands use `os.getcwd()` as the working directory. Arguments are passed as list tokens — no shell interpolation occurs.

### git_status

Show the working tree status: staged changes, unstaged changes, and untracked files.

**Parameters:** None

**Returns:** Short-format status output with branch name.

---

### git_diff

Show a unified diff of changes in the working tree (or staged changes).

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | No | `""` (all) | Specific file or directory to diff. Leave empty to diff all changes. |
| `staged` | bool | No | `false` | Show staged (cached) diff instead of unstaged working-tree diff. |

**Returns:** `git diff --stat --patch` output.

---

### git_log

Show recent commit history with hash, date, author, and subject.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_count` | int | No | `10` | Number of commits to show (1–100). |
| `branch` | string | No | `""` (current) | Branch or ref to show history for. |

**Returns:** Formatted one-line-per-commit log.

---

### git_add ⚠️

Stage one or more files for the next commit.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paths` | array of strings | Yes | File paths to stage. Pass `["."]` to stage all changes. |

---

### git_commit ⚠️

Create a commit from all currently staged changes.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | Yes | Commit message (max 4096 chars). |

---

### git_create_branch ⚠️

Create a new branch and switch to it.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | — | New branch name (max 256 chars). |
| `base` | string | No | `"HEAD"` | Starting point: branch name, tag, or commit hash. |

---

### git_checkout ⚠️

Switch to a branch or restore a file to its last committed state.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ref` | string | Yes | Branch name to switch to, or file path to restore to HEAD. |

---

## Math & Calculation

### calculate

Evaluate mathematical expressions safely.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `expression` | string | Yes | Math expression to evaluate |

**Supported Functions:**
- Basic operators: `+`, `-`, `*`, `/`, `**`, `%`
- Built-in: `abs`, `round`, `min`, `max`, `sum`
- Powers & logarithms: `sqrt`, `pow`, `exp`, `log`, `log10`, `log2`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Angular conversion: `degrees`, `radians`
- Rounding: `ceil`, `floor`, `trunc`
- Other: `factorial`, `gcd`
- Constants: `pi`, `e`, `tau`, `inf`

**Safety limits:** factorial capped at 1,558; exponents capped at magnitude 1,000.

**Examples:**
```
sqrt(16) + 2**3        → 12.0
sin(pi/2)              → 1.0
log(100, 10)           → 2.0
factorial(5)           → 120
ceil(3.2)              → 4
```

---

## Date & Time

### get_current_datetime

Get current date and time in any timezone.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `timezone` | string | No | Timezone name (default: UTC) |
| `output_format` | string | No | Output format (default: %Y-%m-%d %H:%M:%S %Z) |

**Examples:**
```
Timezone: America/New_York
Timezone: Europe/London
Timezone: Asia/Tokyo
```

---

### convert_timezone

Convert datetime between timezones.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datetime_str` | string | Yes | Datetime to convert |
| `from_timezone` | string | Yes | Source timezone |
| `to_timezone` | string | Yes | Target timezone |
| `output_format` | string | No | Output format (strftime format string, default: "%Y-%m-%d %H:%M:%S %Z") |

---

### parse_date

Parse date strings in various formats.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `date_str` | string | Yes | Date string to parse |
| `output_format` | string | No | Output format (strftime format string, default: "%Y-%m-%d %H:%M:%S") |

**Supported Formats:**
- `2024-12-25`
- `December 25, 2024`
- `25/12/2024`
- `12-25-2024`

---

## Text Processing

### word_count

Count words, characters, lines, and sentences.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze |

**Returns:**
```json
{
  "words": 150,
  "characters": 823,
  "lines": 12,
  "sentences": 8
}
```

---

### find_replace

Find and replace text with regex support.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Input text |
| `find` | string | Yes | Pattern to find |
| `replace` | string | Yes | Replacement text |
| `use_regex` | bool | No | Use regex (default: false) |
| `case_sensitive` | bool | No | Case sensitive (default: true) |

---

### extract_urls

Extract all URLs from text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to extract URLs from |

**Returns:** List of URLs found

---

### extract_emails

Extract all email addresses from text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to extract emails from |

**Returns:** List of email addresses found

---

### text_compare

Compare two texts and show differences.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text1` | string | Yes | First text |
| `text2` | string | Yes | Second text |

**Returns:** Diff output showing additions and deletions

---

### split_text

Split text by delimiter.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to split |
| `delimiter` | string | No | Delimiter (default: newline) |
| `max_parts` | int | No | Maximum number of parts to split into |

---

### trim_text

Trim text to maximum length.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to trim |
| `max_length` | int | Yes | Maximum length |
| `add_ellipsis` | bool | No | Whether to add '...' when truncated (default: true) |

---

## JSON Processing

### parse_json

Parse and validate JSON strings.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `json_str` | string | Yes | JSON string to parse |

**Returns:** Parsed object or validation error

---

### format_json

Pretty-print JSON with indentation.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `json_str` | string | Yes | JSON to format |
| `indent` | int | No | Indentation level (default: 2) |

---

### query_json

Query JSON using path expressions.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `json_str` | string | Yes | JSON to query |
| `path` | string | Yes | Query path (e.g., `data.users[0].name`) |

**Path Syntax:**
```
data.users         → Access 'users' in 'data'
data.users[0]      → First element of array
data.users[-1]     → Last element of array
data.users[*].name → All 'name' fields in array
```

---

### extract_json

Extract JSON from mixed text content.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text containing JSON |

**Returns:** Extracted JSON object(s)

---

### json_to_text

Convert JSON to human-readable text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `json_str` | string | Yes | JSON to convert |

---

## Search

Cogtrix includes 11 search tools across 6 providers. All API-key-gated tools (search, weather, WhatsApp) are automatically hidden from the agent when not configured — they simply don't appear in the tool list. DuckDuckGo is always available (no API key, provides both `search_web` and `search_news`). Other providers are automatically enabled when their API key is configured, and hidden from the agent otherwise.

### search_web

Search the web using DuckDuckGo (no API key needed).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `num_results` | int | No | Maximum results (default: 5) |
| `region` | string | No | Region for results (default: `"wt-wt"` worldwide). Examples: `"us-en"`, `"uk-en"` |

**Returns:** List of results with title, URL, and snippet

---

### search_news

Search recent news using DuckDuckGo.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `num_results` | int | No | Maximum results (default: 5) |
| `timelimit` | string | No | Time filter (default: `"w"` for past week). Values: `"d"` (day), `"w"` (week), `"m"` (month) |

**Returns:** List of news articles with title, URL, date, and source

---

### tavily_search

AI-optimised web search that crawls pages and extracts their full text content.

**Requires:** `TAVILY_API_KEY` environment variable or `services.tavily.api_key` in config. Also requires `tavily-python` package.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Search query |
| `search_depth` | string | No | `"advanced"` | `"basic"` (fast, snippets) or `"advanced"` (deep crawl, full content) |
| `max_results` | int | No | `5` | Number of results (1-10) |
| `include_answer` | bool | No | `true` | Include AI-generated answer summary |
| `topic` | string | No | `"general"` | `"general"` or `"news"` |

**Returns:** AI summary + results with title, URL, relevance score, and extracted page content

---

### tavily_extract

Extract clean text content from specific URLs using Tavily. Handles JavaScript-rendered pages.

**Requires:** Same as `tavily_search`.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `urls` | array | Yes | List of URLs to extract content from (max 20) |

**Returns:** Extracted text content per URL

---

### exa_search

AI-native semantic web search using neural embeddings. Understands the *meaning* of queries, not just keywords.

**Requires:** `EXA_API_KEY` environment variable or `services.exa.api_key` in config. Also requires `exa-py` package.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Natural-language search query |
| `num_results` | int | No | `5` | Number of results (1-10) |
| `include_text` | bool | No | `true` | Include extracted page text |
| `search_type` | string | No | `"auto"` | `"auto"`, `"neural"` (semantic), `"fast"` (low-latency), `"deep"` (thorough), or `"instant"` (cached) |

**Returns:** Results with title, URL, relevance score, and extracted page text

---

### exa_find_similar

Find web pages similar to a given URL using Exa's neural embeddings.

**Requires:** Same as `exa_search`.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | — | URL of the reference page |
| `num_results` | int | No | `5` | Number of similar results (1-10) |
| `include_text` | bool | No | `true` | Include extracted page text |

**Returns:** List of similar pages with content

---

### exa_get_contents

Extract clean text content from web pages using Exa.

**Requires:** Same as `exa_search`.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `urls` | array | Yes | List of URLs to extract content from |

**Returns:** Extracted text content per URL (truncated at 8,000 chars each)

---

### brave_search

Search the web using Brave Search — a privacy-focused search engine with its own independent index.

**Requires:** `BRAVE_API_KEY` environment variable or `services.brave.api_key` in config. No extra package needed.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Search query |
| `count` | int | No | `5` | Number of results (1-20) |
| `search_type` | string | No | `"web"` | `"web"` or `"news"` |
| `freshness` | string | No | `""` | Time filter: `"pd"` (day), `"pw"` (week), `"pm"` (month), `"py"` (year) |

**Returns:** Results with titles, URLs, descriptions, age, extra snippets, FAQ answers, and infoboxes

---

### google_search

Search using the official Google Custom Search JSON API — real Google Search results.

**Requires:** `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` environment variables (or `services.google.api_key` / `services.google.cse_id` in config). No extra package needed. Free tier: 100 queries/day.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Search query |
| `num_results` | int | No | `10` | Number of results (1-10) |
| `date_restrict` | string | No | `""` | Date filter: `"d7"` (7 days), `"w2"` (2 weeks), `"m1"` (month), `"y1"` (year) |
| `language` | string | No | `""` | Language restriction (e.g., `"lang_en"`, `"lang_de"`) |
| `safe_search` | string | No | `"off"` | `"off"` or `"active"` |

**Returns:** Organic results with titles, URLs, snippets, spelling suggestions, published dates, and meta descriptions

---

### serpapi_search

Search using SerpAPI — structured proxy for Google and Bing with the richest structured output.

**Requires:** `SERPAPI_API_KEY` environment variable or `services.serpapi.api_key` in config. Also requires `google-search-results` package.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | Search query |
| `engine` | string | No | `"google"` | `"google"` or `"bing"` |
| `num_results` | int | No | `10` | Number of results (1-20) |
| `search_type` | string | No | `""` | `""` (web), `"nws"` (news), `"isch"` (images), `"shop"` (shopping) |
| `time_period` | string | No | `""` | `"qdr:d"` (day), `"qdr:w"` (week), `"qdr:m"` (month), `"qdr:y"` (year) |

**Returns:** Answer boxes, knowledge graph, People Also Ask, rich snippets, and organic results

---

## Web & HTTP

### http_get

Make HTTP GET requests.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | URL to request |
| `headers` | string | No | Request headers as JSON string |
| `timeout` | int | No | Timeout in seconds (default: 30) |

**Returns:** Response body and status code

---

### http_post ⚠️

Make HTTP POST requests with JSON data.

**Requires Confirmation:** Yes

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | Yes | URL to request |
| `data` | string | Yes | Request body as JSON string |
| `headers` | string | No | Request headers as JSON string |
| `timeout` | int | No | Timeout in seconds (default: 30) |

---

## Weather

### get_weather

Get current weather for any location.

**Requires:** OpenWeather API key (set in config or `OPENWEATHER_API_KEY`)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `location` | string | Yes | City name or coordinates |
| `units` | string | No | Units: `metric`, `imperial` (default: metric) |

**Returns:**
```json
{
  "temperature": 22,
  "feels_like": 24,
  "humidity": 65,
  "description": "partly cloudy",
  "wind_speed": 12
}
```

---

## NLP Tools

### analyze_sentiment

Analyze text sentiment.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze |

**Returns:**
```json
{
  "sentiment": "positive",
  "polarity": 0.75,
  "subjectivity": 0.6
}
```

- **polarity:** -1.0 (negative) to 1.0 (positive)
- **subjectivity:** 0.0 (objective) to 1.0 (subjective)

---

### summarize_text

Summarize long text by extracting important sentences.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to summarize |
| `num_sentences` | int | No | Number of sentences (default: 3) |

---

### extract_keywords

Extract the most important keywords from text.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze |
| `num_keywords` | int | No | Number of keywords (default: 10) |

---

## WhatsApp Messaging

Send and receive WhatsApp messages via a self-hosted [Waha](https://waha.devlike.pro/) container. All four tools are automatically hidden when WhatsApp is not configured.

**Requires:** A running Waha Docker container. See [WhatsApp Setup](CONFIGURATION.md#whatsapp-messaging) for configuration.

### whatsapp_send ⚠️

Send a text message via WhatsApp.

**Requires Confirmation:** Yes (configurable)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Recipient — phone number in E.164 format (e.g. `+14155551234`) or a phonebook nickname (e.g. `alice`) |
| `message` | string | Yes | Text message body (max 4096 chars) |

**Returns:** Confirmation with message ID, or error/block reason

**Example:**
```
whatsapp_send(to="alice", message="Meeting moved to 3pm")
```

---

### whatsapp_send_image ⚠️

Send an image via WhatsApp given a public URL.

**Requires Confirmation:** Yes (configurable)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Recipient — phone number or phonebook nickname |
| `image_url` | string | Yes | Public URL of the image (JPEG preferred) |
| `caption` | string | No | Optional caption text |

**Returns:** Confirmation with message ID, or error/block reason

---

### whatsapp_check

Retrieve recent WhatsApp messages. Optionally filter by a specific contact.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `contact` | string | No | All chats | Phone number or phonebook nickname to filter by |
| `limit` | int | No | `10` | Number of recent messages (max 50) |

**Returns:** Formatted list of recent messages with sender, timestamp, and body

---

### whatsapp_contacts

List the configured phonebook contacts and active filter rules.

**Parameters:** None

**Returns:** Phonebook listing with nicknames, phone numbers, and filter mode

---

### Contact Filtering

WhatsApp tools enforce contact restrictions before any message is sent or received:

| Filter Mode | Behavior |
|-------------|----------|
| `none` (default) | All contacts allowed |
| `allow` | Only contacts in the list can send/receive (legacy value `whitelist` is auto-mapped to `allow`) |
| `ignore` | Skip listed contacts — no response sent, message kept on the platform |
| `blacklist` | Contacts in the list are blocked (message deleted, chat archived) |

Phonebook nicknames (e.g. `"alice"`) are resolved to E.164 numbers automatically and are case-insensitive.

### Rate Limiting

Outbound messages are rate-limited to prevent abuse. Default: 30 messages/hour (configurable, 0 = unlimited). The limit uses an in-memory sliding window that resets on process restart.

---

## Telegram Messaging

Send and receive Telegram messages via a bot created with [@BotFather](https://t.me/BotFather). All four tools are automatically hidden when the bot token is not configured.

**Requires:** A bot token (`COGTRIX_TELEGRAM_TOKEN` environment variable or `services.telegram.bot_token` in config). See [Telegram Setup](CONFIGURATION.md#telegram-messaging) for configuration.

### telegram_send

Send a text message via Telegram.

**Requires Confirmation:** Yes (configurable)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Recipient — a chat ID (numeric), @username, or a phonebook nickname (e.g. `alice`) |
| `message` | string | Yes | Text message body (max 4096 chars) |

**Returns:** Confirmation with message ID, or error/block reason

**Example:**
```
telegram_send(to="alice", message="Meeting moved to 3pm")
```

---

### telegram_send_photo

Send a photo via Telegram given a public URL.

**Requires Confirmation:** Yes (configurable)

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `to` | string | Yes | Recipient — a chat ID, @username, or phonebook nickname |
| `photo_url` | string | Yes | Public URL of the photo |
| `caption` | string | No | Optional caption text |

**Returns:** Confirmation with message ID, or error/block reason

---

### telegram_check

Retrieve recent Telegram messages sent to the bot.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `10` | Number of recent messages (max 50) |

**Returns:** Formatted list of recent messages with sender, timestamp, chat, and body

---

### telegram_contacts

List the configured phonebook contacts and active filter rules.

**Parameters:** None

**Returns:** Phonebook listing with nicknames, chat IDs, and filter mode

---

### Contact Filtering

Telegram tools enforce contact restrictions before any message is sent or received:

| Filter Mode | Behavior |
|-------------|----------|
| `none` (default) | All contacts allowed |
| `allow` | Only contacts in the list can send/receive (legacy value `whitelist` is auto-mapped to `allow`) |
| `ignore` | Skip listed contacts — no response sent, message kept on the platform |
| `blacklist` | Contacts in the list are blocked (message deleted) |

Phonebook nicknames are resolved to chat IDs automatically and are case-insensitive.

### Rate Limiting

Outbound messages are rate-limited to prevent abuse. Default: 30 messages/hour (configurable, 0 = unlimited). The limit uses an in-memory sliding window that resets on process restart.

---

## Scheduling

### schedule_reply

Schedule a reply for delayed delivery instead of sending immediately.

**Availability:** Only available in assistant mode (`--assistant`). Injected per-call by `MessageHandler` when a `MessageScheduler` is configured.

**When Used:** The agent calls this tool when the system prompt includes timing or scheduling instructions (e.g., "reply in 3 hours", "respond after the meeting"). The agent's response is NOT sent immediately — it is queued and delivered by a background thread.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | The reply message to send later |
| `delay_minutes` | int | Yes | Minutes to wait before sending (1–1440) |

**Notes:**

- Message text is sanitized through output guardrails before being queued
- Delivery retries up to 3 times with exponential backoff (30 s → 120 s → 600 s) on send failure
- A new incoming message from the same chat cancels any pending scheduled replies for that chat
- Quiet hours are enforced at dispatch time — delivery defers to the end of the quiet window rather than being dropped
- Queue is persisted to `data/assistant/schedule.json` and survives restarts
- Configurable via `services.assistant.response_timing` (quiet hours, per-contact overrides)

**Returns:** Confirmation string telling the agent not to repeat the message.

> **Note:** In assistant mode, additional message management tools are available:
> `queue_reply`, `edit_last_reply`, `list_scheduled_messages`,
> `edit_scheduled_message`, `cancel_scheduled_message`, `defer_processing`,
> and `suppress_reply`. These are injected per-call and documented in the
> [Architecture Guide](ARCHITECTURE.md).

---

## Knowledge Base

### query_knowledge_base

Search the knowledge base for information from uploaded documents. Searches all available FAISS indexes — both the global CLI index (`data/vectordb/faiss_index/`) and per-document API indexes (`data/api/uploads/{doc_id}/vectordb/faiss_index/`).

**Requires:** Vector store built with `python cogtrix.py --ingest` or documents uploaded via the API

**Auto-activation:** When a knowledge base exists, this tool is automatically pinned as active at startup (no need to load it via `request_tools`). Its description dynamically shows the number of indexes and total size.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question or topic to search for |
| `k` | int | No | Number of results to return (default: 4, max: 10) |

**Returns:** Relevant document chunks with sources

See [RAG_GUIDE.md](RAG_GUIDE.md) for setup instructions.

---

## Delegation

### delegate_task

Delegate a single task to another LLM model.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | — | Task description |
| `context` | string | No | `""` | Relevant context/data for the task |
| `response_format` | string | No | `"text"` | Expected format: `text`, `json`, `code`, `markdown` |
| `json_schema` | string | No | — | Expected JSON structure (if `response_format="json"`) |
| `provider` | string | No | — | Provider name or alias |
| `model` | string | No | — | Model alias or model name |
| `timeout` | int | No | `60` | Timeout in seconds (10-600) |
| `temperature` | float | No | `0.7` | Model temperature (0.0-2.0) |
| `use_tools` | bool | No | `true` | Give the delegate access to tools. Set `false` for LLM-only tasks. |

**Model Resolution:**
```
model: "fast"                    → Uses string alias from config
model: "deep"                    → Uses object alias (with context_window, temperature)
model: "ollama/qwen3:8b"         → Direct provider/model
model: "openai/gpt-4.1"           → Direct provider/model
```

Object entries can override `context_window`, `temperature`, and `timeout` per model. Note: `context_window` is forwarded to Ollama as `num_ctx` and silently ignored for other providers. See [CONFIGURATION.md](CONFIGURATION.md#models) for model entry format details and [Delegate Section](CONFIGURATION.md#delegate-section) for `allowed_models` restrictions.

---

### delegate_parallel

Run multiple tasks in parallel across LLM models.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tasks` | array | Yes | List of task objects |
| `timeout` | int | No | Timeout per task |

**Task Object:**
```json
{
  "task": "Summarize this article",
  "model": "fast",
  "context": "Article text here...",
  "provider": "ollama",
  "temperature": 0.5,
  "response_format": "text",
  "use_tools": true
}
```

Only `task` is required; other fields are optional.

**Returns:** List of results from all tasks

---

## Deep Reasoning

### deep_think

Tree-of-Thought with Chain-of-Thought Reflection engine for complex problems.

**Also available as:** `/think <task>` slash command (invokes deep_think directly, bypassing agent tool selection).

**How It Works:**

The engine runs multiple iterations, each with three phases:

1. **Branch** — Generate N fundamentally different approaches (1 LLM call)
2. **Develop** — Full Chain-of-Thought for each approach in parallel: Plan → Execute → Observe → Reflect (N parallel LLM calls)
3. **Converge** — Evaluate all solutions, cross-pollinate best ideas, synthesize an improved solution (1 LLM call)

Between iterations, the reflection output feeds into the next branching phase, progressively refining the solution. Stops when confidence is high or max iterations reached.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | — | Problem to solve through deep reasoning |
| `context` | string | No | `""` | Additional context or constraints |
| `max_iterations` | int | No | `3` | Reflection-revision cycles (1-5) |
| `num_branches` | int | No | `3` | Parallel approaches per iteration (2-5) |
| `beam_width` | int | No | `2` | Best paths to keep between iterations (1-3) |

**LLM calls per iteration:** N + 2 (where N = `num_branches`)

**Typical duration:** 1-5 minutes depending on model speed and parameters.

**Returns:** Structured analysis report with:
- Scored approaches from each iteration
- Reflection insights
- Final synthesized solution with confidence rating

**When the LLM uses this tool automatically:**

The agent is guided to use `deep_think` when it encounters:
- Problems with multiple valid approaches or significant trade-offs
- Requests for thorough analysis, deep research, or "think step by step"
- Architecture/design decisions, strategy planning
- Complex debugging where the root cause is unclear
- Comparing or evaluating multiple options systematically

In **reasoning mode** (`-M reasoning`), the agent receives extra encouragement to use this tool for decisions with trade-offs.

**Example:**
```
deep_think(
  task="Design a caching strategy for a microservices architecture
        with 50 services and mixed read/write workloads",
  context="Budget: moderate. Must handle 10K req/s. Latency < 50ms.",
  max_iterations=3,
  num_branches=3
)
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) — Tool configuration
- [DEVELOPMENT.md](DEVELOPMENT.md) — Adding custom tools
- [DEEPTHINK.md](DEEPTHINK.md) — Deep Think reasoning guide
- [RAG_GUIDE.md](RAG_GUIDE.md) — Knowledge base setup
- [WHATSAPP_GUIDE.md](WHATSAPP_GUIDE.md) — WhatsApp assistant setup
- [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md) — Telegram assistant setup
