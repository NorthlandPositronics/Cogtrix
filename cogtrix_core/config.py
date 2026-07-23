"""
Cogtrix Agent - Configuration Management

Configuration priority (highest to lowest):
1. Command line arguments (``--config-file`` overrides search)
2. Environment variables
3. Configuration file (first found wins)
4. Built-in defaults

Supported formats:
    JSON (``.json``) and YAML (``.yml`` / ``.yaml``).

Config file search order (first found wins):
    1. ``--config-file <path>``  (explicit, skips all search)
    2. ``./.cogtrix.json``
    3. ``./.cogtrix.yml``  or ``./.cogtrix.yaml``
    4. ``~/.cogtrix.json``
    5. ``~/.cogtrix.yml``  or ``~/.cogtrix.yaml``
    6. ``~/.config/cogtrix/cogtrix.json``
    7. ``~/.config/cogtrix/cogtrix.yml``  or ``~/.config/cogtrix/cogtrix.yaml``

YAML example::

    provider: ollama
    model: fast

    providers:
      ollama:
        type: ollama
        base_url: http://localhost:11434
        model: qwen3:8b
      openai:
        type: openai
        api_key: sk-...
        model: gpt-4.1-mini

    models:
      fast:
        provider: ollama
        model: qwen3:8b
        context_window: 32768
      reasoning:
        provider: openai
        model: gpt-4.1
        temperature: 0.2
        # Optional sampling passthrough forwarded to the chat model (#2122);
        # use it to tune out model repetition loops on OpenAI-compatible
        # endpoints (LiteLLM/vLLM/qwen3, etc.):
        model_kwargs:
          frequency_penalty: 0.3
          top_p: 0.9
          extra_body: { repetition_penalty: 1.1 }
      embed-local:
        provider: ollama
        model: nomic-embed-text

    rag:
      model: embed-local
      chunk_size: 2000
      chunk_overlap: 200
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cogtrix_core.orchestration.run_config import ExecutionSettings

_log = logging.getLogger("cogtrix")


def _safe_int(value: Any, field_name: str, default: int | None = None) -> int | None:
    """Safely coerce a config value to int, logging a warning on failure."""
    if isinstance(value, bool):
        _log.warning(
            "Invalid integer for %s: %r (bool is not a valid integer), skipping", field_name, value
        )
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        if default is not None:
            _log.warning("Invalid integer for %s: %r, using default %d", field_name, value, default)
            return default
        _log.warning("Invalid integer for %s: %r, skipping", field_name, value)
        return None


def _safe_float(value: Any, field_name: str, default: float | None = None) -> float | None:
    """Safely coerce a config value to float, logging a warning on failure."""
    import math

    if isinstance(value, bool):
        _log.warning(
            "Invalid float for %s: %r (bool is not a valid float), skipping", field_name, value
        )
        return default
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            _log.warning("Invalid float for %s: %r is NaN or Inf, skipping", field_name, result)
            return default
        return result
    except (ValueError, TypeError):
        if default is not None:
            _log.warning("Invalid float for %s: %r, using default %g", field_name, value, default)
            return default
        _log.warning("Invalid float for %s: %r, skipping", field_name, value)
        return None


class ConfigError(Exception):
    """Raised when configuration is invalid."""

    pass


CONFIG_BASENAME = ".cogtrix"
# All supported config file names, checked in this order within each directory
_CONFIG_NAMES = [f"{CONFIG_BASENAME}.json", f"{CONFIG_BASENAME}.yml", f"{CONFIG_BASENAME}.yaml"]
# XDG-style directory uses a simpler name (no leading dot)
_XDG_CONFIG_NAMES = ["cogtrix.json", "cogtrix.yml", "cogtrix.yaml"]


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider (connection info only)."""

    name: str
    type: str  # "openai", "ollama", "anthropic", or "google"
    base_url: str | None = None
    api_key: str | None = None
    tool_instructions: str | None = None

    def __post_init__(self) -> None:
        if self.type:
            from cogtrix_core.providers.defaults import PROVIDER_TYPES

            if self.type not in PROVIDER_TYPES:
                raise ConfigError(
                    f"providers.{self.name}.type '{self.type}' is not a recognized provider type. "
                    f"Supported: {', '.join(sorted(PROVIDER_TYPES))}"
                )

    def get_base_url(self) -> str | None:
        """Get base URL with defaults for known types."""
        if self.base_url:
            return self.base_url
        from cogtrix_core.providers import get_default_base_url

        return get_default_base_url(self.type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        api_key: str | None = self.api_key
        if api_key:
            if len(api_key) < 10:
                api_key = "***"
            else:
                api_key = api_key[:3] + "***" + api_key[-4:]
        return {
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "api_key": api_key,
            "tool_instructions": self.tool_instructions,
        }


@dataclass
class ModelConfig:
    """Configuration for a named model in the models registry."""

    provider: str  # references a key in Config.providers
    model: str  # actual model name at the provider
    context_window: int | None = None  # None → DEFAULT_CONTEXT_WINDOW (32768)
    temperature: float | None = None  # None → DEFAULT_TEMPERATURE (0.5)
    max_tokens: int | None = None  # Max output tokens per LLM call
    timeout: int = 180  # LLM request timeout in seconds (per call, not total)
    #: Extra keyword arguments forwarded verbatim to the underlying chat-model
    #: constructor (e.g. ``frequency_penalty``, ``presence_penalty``, ``top_p``,
    #: or ``extra_body={"repetition_penalty": 1.1}`` for OpenAI-compatible
    #: endpoints).  Lets operators tune anti-repetition sampling per model from
    #: ``cogtrix.yaml`` (#2122).  Reserved keys already set explicitly elsewhere
    #: (model, temperature, max_tokens, api_key, base_url, streaming, …) are
    #: dropped to avoid duplicate-keyword errors.
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Declare whether this model accepts image content (vision). ``True`` =
    #: vision-capable; ``False`` = text-only; ``None`` (default) = unknown.
    #: Used by the assistant vision delegation path (#2262): when ``None`` and
    #: no ``vision_model`` is configured, images are forwarded as-is.
    supports_vision: bool | None = None

    DEFAULT_TEMPERATURE: float = 0.5
    DEFAULT_CONTEXT_WINDOW: int = 32_768

    #: Default context window applied when no explicit value is set in config.
    DEFAULT_CONTEXT_WINDOW: int = 32_768
    #: Default temperature applied when no explicit value is set in config.
    DEFAULT_TEMPERATURE: float = 0.5

    def __post_init__(self) -> None:
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.context_window is not None and self.context_window < 256:
            raise ConfigError(f"context_window must be >= 256, got {self.context_window}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.timeout < 10:
            raise ConfigError(f"timeout must be >= 10, got {self.timeout}")
        if not isinstance(self.model_kwargs, dict):
            raise ConfigError(
                f"model_kwargs must be a mapping, got {type(self.model_kwargs).__name__}"
            )


@dataclass
class RAGConfig:
    """Configuration for RAG document ingestion."""

    docs_dir: str = "docs"
    vectordb_dir: str = "vectordb"
    # #1952 Option C: lowered from 2000/200 → 800/100.  See
    # ``cogtrix_core/rag/ingest.py:IngestConfig`` docstring for the rationale.
    # Operators with explicit ``rag.chunk_size`` / ``rag.chunk_overlap``
    # in ``~/.cogtrix.yaml`` keep their override values; operators on
    # defaults get the new behaviour after re-running ``--ingest``.
    chunk_size: int = 800
    chunk_overlap: int = 100
    model: str | None = None  # references a key in Config.models for embedding
    score_threshold: float = 0.0  # minimum similarity score for RAG retrieval (M4.3)
    # #1981: opt-in BM25 hybrid retrieval.  Both flags default OFF —
    # existing pure-vector pipelines see zero behavioural change until
    # the operator enables them.
    #
    # ``build_bm25_sidecar`` is honoured at ingest time: when True,
    # ``python cogtrix.py --ingest`` writes a ``bm25.pkl`` next to
    # ``index.faiss``.  Inexpensive (~5% of ingest wall-clock for
    # typical corpora) so flipping this on is low-cost even if the
    # query side hasn't enabled hybrid yet.
    #
    # ``use_bm25_hybrid`` is honoured at query time: when True AND a
    # sidecar exists, the query path fuses vector + BM25 ranks via
    # Reciprocal Rank Fusion.  Operators with the regime-B (monetary
    # tokens) or regime-C (one document dominates) problems from
    # #1952 should set BOTH flags and re-ingest.
    build_bm25_sidecar: bool = False
    use_bm25_hybrid: bool = False
    # RRF tuning constant — Cormack et al. 2009's standard value is 60.
    # Operators normally should not need to tune this.
    bm25_rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"rag.chunk_overlap ({self.chunk_overlap}) must be less than "
                f"rag.chunk_size ({self.chunk_size})"
            )
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ConfigError(
                f"rag.score_threshold must be in [0.0, 1.0], got {self.score_threshold}"
            )
        if self.bm25_rrf_k <= 0:
            raise ConfigError(f"rag.bm25_rrf_k must be positive, got {self.bm25_rrf_k}")


# SlowAPI-style rate-limit spec: ``<N>/<window>`` where window is one of
# ``second`` / ``minute`` / ``hour`` / ``day`` (or the single-letter alias).
# Accepts an optional trailing ``s`` (``5/minutes``) and surrounding
# whitespace. Used by :class:`APIConfig` and by the API rate-limit
# resolver in ``cogtrix_core/api/rate_limit.py``.
_RATE_LIMIT_SPEC_RE = re.compile(
    r"^\s*(\d+)\s*/\s*(second|minute|hour|day|s|m|h|d)s?\s*$",
    re.IGNORECASE,
)


@dataclass
class APIConfig:
    """Configuration for the Cogtrix HTTP API server (#1879).

    Currently exposes per-route rate-limit overrides and the trusted-
    reverse-proxy CIDR allowlist that ``_client_key`` consults to
    recover the real client IP behind a load balancer.

    Defaults preserve the previously-hardcoded values so existing
    deployments behave identically when no ``api:`` block is present
    in ``.cogtrix.yaml``.
    """

    # Per-route rate limits, expressed as SlowAPI-style ``"<N>/<window>"``
    # strings. The ``default`` key is the catch-all applied when a route
    # name is not explicitly listed. Adding a new key requires no code
    # change beyond a matching ``per_route_rate_limit_for("<name>")``
    # call at the route definition.
    rate_limits: dict[str, str] = field(
        default_factory=lambda: {
            "default": "120/minute",
            "auth_register": "3/hour",
            "auth_login": "5/minute",
            "auth_refresh": "5/minute",
            "saml_acs": "5/minute",
        }
    )

    # IPv4 / IPv6 CIDR networks for trusted reverse proxies. When
    # non-empty, ``_client_key`` walks ``X-Forwarded-For`` right-to-left
    # honouring this allowlist so the rate limiter buckets by the real
    # client IP rather than the load-balancer's pod address.
    trusted_proxy_cidrs: list[str] = field(default_factory=list)

    # Optional Redis URL for a shared rate-limit counter (#1879 Slice B).
    # When unset, the rate limiter falls back to a per-process in-memory
    # sliding window — correct for single-node deployments but jitters
    # under horizontal scaling. Set to a ``redis://...`` (or
    # ``rediss://...`` / ``redis+sentinel://...``) URL to share the
    # counter across replicas. Requires the ``cogtrix[redis]`` install
    # extra. ``COGTRIX_REDIS_URL`` env var takes precedence over this
    # value at runtime.
    redis_url: str | None = None

    # Allowed CORS origins for the browser-facing API (#2059). Exact-match
    # origins the API answers cross-origin requests for. The default is
    # localhost-only on purpose: a production deployment MUST set its real
    # origin(s) (via ``COGTRIX_CORS_ORIGINS``, the ``api.cors_origins`` config
    # key, or a Helm value) so a misconfigured prod fails loudly rather than
    # silently half-allowing a placeholder host. ``COGTRIX_CORS_ORIGINS``
    # (comma-separated) overrides this at runtime per the config hierarchy.
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",  # Vite React dev server
            "http://localhost:3000",  # Create-React-App dev server (fallback)
        ]
    )

    # Content guardrails for the API chat path (#2056). The assistant/messaging
    # mode runs a GuardrailPipeline (banned strings, PII redaction, URL blocking,
    # encoding/injection detection, optional LLM-judge, rate-limit/blacklist
    # reactions); the API chat path historically had none. This dict mirrors the
    # assistant ``services.assistant.guardrails`` schema and is wired into the API
    # turn runner. Empty (the default) = OFF: the pipeline is only constructed
    # when ``api.guardrails.enabled`` is true, so existing deployments are
    # unchanged. See docs/CONFIGURATION.md.
    guardrails: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, spec in self.rate_limits.items():
            if not _RATE_LIMIT_SPEC_RE.match(spec):
                raise ConfigError(
                    f"api.rate_limits.{name!r} must be '<N>/<window>' where "
                    f"window is one of second/minute/hour/day (or s/m/h/d); "
                    f"got {spec!r}"
                )
        if "default" not in self.rate_limits:
            raise ConfigError(
                "api.rate_limits must define a 'default' key (used as the "
                "catch-all when a route is not explicitly listed)"
            )
        import ipaddress

        for cidr in self.trusted_proxy_cidrs:
            try:
                ipaddress.ip_network(cidr, strict=False)
            except ValueError as exc:
                raise ConfigError(f"api.trusted_proxy_cidrs: invalid CIDR {cidr!r}: {exc}") from exc

        if not isinstance(self.cors_origins, list) or not all(
            isinstance(o, str) and o.strip() for o in self.cors_origins
        ):
            raise ConfigError(
                "api.cors_origins must be a list of non-empty origin strings "
                "(e.g. ['https://cogtrix.ai'])"
            )

        if not isinstance(self.guardrails, dict):
            raise ConfigError(
                f"api.guardrails must be a mapping (got {type(self.guardrails).__name__}); "
                "see docs/CONFIGURATION.md for the schema"
            )
        # The GuardrailPipeline raises on `enabled: false` (it would bypass every
        # check). On the API path we instead treat falsey/absent as "off" — only a
        # truthy `enabled` constructs the pipeline — so a config that explicitly
        # sets `api.guardrails.enabled: false` means "disabled", not "construct a
        # bypassed pipeline". No further validation here; the pipeline validates
        # its own sub-keys at construction.


@dataclass
class Config:
    """Application configuration with defaults."""

    # General settings
    session: str = "default"
    data_dir: str = "data"

    # Inference providers (LLM backends)
    # Populated from "providers" (preferred) or "inference" (alias)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Models registry — named model configurations
    models: dict[str, ModelConfig] = field(default_factory=dict)

    # External services — flat dict of {service_name: {config...}}
    services: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cron jobs defined in config — list of {name, schedule, prompt, context}
    cron: list[dict[str, Any]] = field(default_factory=list)

    # Memory settings
    memory_mode: str = "conversation"
    memory_config: dict[str, Any] | None = field(default=None)
    memory_modes: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Delegate tool settings
    delegate_enabled: bool = True
    delegate_default_timeout: int = 60
    delegate_allowed_providers: list | None = field(default=None)
    delegate_allowed_models: list[str] | None = field(default=None)

    # Prompt optimizer — rewrite complex prompts before agent execution
    prompt_optimizer: bool = True

    # Adaptive memory — auto-select and switch memory mode based on prompt heuristics
    adaptive_memory: bool = True

    # Auto model routing — use a fast model for simple queries
    auto_route: bool = False
    auto_route_fast_model: str | None = None  # model alias in models registry

    # Quick mode — skip optimizer, memory, and compression for one-off queries
    quick_mode: bool = False

    # Git-native mode — auto stage+commit after each file write
    git_native: bool = False

    # Banner display mode: "compact" (default), "full", or "off"
    banner: str = "compact"

    theme: str = "default"
    """UI colour theme. Built-in values: default, minimal, dracula."""

    # Per-tool trust overrides: tool_name -> "always" | "ask" | "deny"
    tool_trust: dict[str, str] = field(default_factory=dict)

    # Allow shell/bash/python_exec in API sessions (disabled by default for safety)
    api_dangerous_tools: bool = False

    # Enable data science modules (numpy, pandas, scipy) in python_exec
    # WARNING: These modules bypass the AST sandbox via C-extensions.
    # Set to true ONLY if you need numpy/pandas/scipy and understand the risks.
    enable_datascience_modules: bool = False

    # Enable organization scoping for admin enumeration endpoints.
    # Phase 1 (default false): regular admins receive 403 on scoped endpoints.
    # Phase 2 (future): per-org filtering with JWT org_id claims.
    enable_org_scoping: bool = False

    # Named flag profiles: profile_name -> {config_key: value}
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Parallel tool execution — run independent tool calls concurrently
    parallel_tool_execution: bool = True

    # File operations — extra directories allowed for write operations
    allowed_write_paths: list[str] = field(default_factory=list)

    # File operations — extra directories allowed for read operations
    allowed_read_paths: list[str] = field(default_factory=list)

    # Shell tool — allowed domains for curl/wget URL targets.
    # When non-empty, curl and wget invocations must target these domains only.
    # Mitigates data exfiltration via URL param injection (issue #1604).
    shell_curl_wget_allowed_domains: list[str] = field(default_factory=list)

    # Shell tool — operator-controlled, opt-in policy extensions (#2392).
    # Both default empty → the shell=True policy is exactly as locked-down as the
    # built-in blocklist/allowlist; nothing relaxes unless the operator sets these.
    #   * shell_extra_safe_commands: extra command names allowed on the shell=True
    #     allowlist (e.g. ["sync", "umount"]). Does NOT override the blocklist.
    #   * shell_allow_patterns: regexes that fully exempt a matching command from
    #     the shell=True restrictions (blocklist + download-then-exec + allowlist) —
    #     the operator's explicit "this command shape is safe on my infra" hatch
    #     (e.g. sysadmin over SSH to a host they control). Audited on use.
    shell_extra_safe_commands: list[str] = field(default_factory=list)
    shell_allow_patterns: list[str] = field(default_factory=list)

    # Plugin tools — extra directories to scan for file-drop tool modules
    tool_dirs: list[str] = field(default_factory=list)

    # Context compression — summarize old ToolMessages during agent loop
    context_compression: bool = True
    context_compression_min_age: int = 6
    context_compression_min_chars: int = 2000
    context_compression_emergency_threshold: float = 0.85
    """Context ratio (0–1) above which min_age_cycles drops to 1 for emergency compression."""
    context_compression_human_msg_max_chars: int = 20_000
    """Maximum HumanMessage content length before middle-truncation. 0 = disabled."""
    context_compression_model: str | None = None  # model name or "provider/model"
    context_max_messages: int = 200
    """Maximum retained message count before oldest-end truncation. 0 = disabled.

    A floor, not a ceiling: when the operator does not set this explicitly, the
    effective cap scales up with the resolved token budget via
    :meth:`resolve_context_max_messages` (#2397) — a flat 200 is the *binding*
    constraint on big-window models (it evicts context every turn even though the
    token budget has ample room), so the token cap should govern and the message
    cap stay a proportional backstop."""
    #: Internal (not a config key): flips to False once the operator sets
    #: ``context_max_messages`` explicitly, so :meth:`resolve_context_max_messages`
    #: only auto-scales an unset cap and never overrides an operator choice.
    context_max_messages_is_default: bool = True
    context_max_tokens: int = 40_000
    """Maximum retained context token budget for oldest-end truncation. 0 = disabled.

    A floor, not a ceiling: when the operator does not set this explicitly, the
    effective budget scales up with the active model's context window via
    :meth:`resolve_context_max_tokens` (#2360) — big-window models were
    over-compressed by a flat 40k every turn."""
    #: Internal (not a config key): flips to False once the operator sets
    #: ``context_max_tokens`` explicitly, so :meth:`resolve_context_max_tokens`
    #: only auto-scales an unset budget and never overrides an operator choice.
    context_max_tokens_is_default: bool = True

    # Search quality heuristic — substantive results discriminator (#1593, Option B)
    search_quality_min_url_count: int = 2
    """Minimum ``"URL: "`` lines in a search Web ToolMessage for it to be considered substantive."""
    search_quality_min_chars: int = 300
    """Minimum character length for a search Web ToolMessage to be considered substantive."""

    # Tiered Context Cache (TCC) — pre-compressed tier snapshots for accurate
    # context size tracking and O(1) context assembly.
    tier_cache_enabled: bool = True
    tier0_fraction: float = 0.60
    """Fraction of available tokens for the verbatim (Tier 0) window."""
    tier1_fraction: float = 0.30
    """Fraction of available tokens for lightly-compressed (Tier 1) history."""
    tier2_fraction: float = 0.08
    """Fraction of available tokens for heavily-compressed (Tier 2) history."""

    # Per-turn context budget guard — abort tool loop when this fraction of
    # max_context_tokens has been consumed (0–1, default 80%).
    tool_context_limit_pct: float = 0.80

    # MCP server configurations
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Named agent configurations (loaded lazily by AgentRegistry)
    agents: dict[str, Any] = field(default_factory=dict)

    # RAG settings
    rag: RAGConfig = field(default_factory=RAGConfig)

    # API server settings (#1879 — per-route rate limits, trusted-proxy CIDRs).
    api: APIConfig = field(default_factory=APIConfig)

    # Logging and debug settings
    debug: bool = False
    verbose: bool = False  # Log full message content without truncation
    log_file: str | None = None  # None = no logging, "" = default file
    verbosity: int = 0  # 0=normal, 1=debug, 2=verbose, 3=trace

    # Audit log (M5.4)
    audit_log_enabled: bool = True
    audit_log_path: str = "data/audit/audit.log"

    # Redis session presence (M5.2)
    redis_url: str = ""  # empty = disabled
    redis_session_ttl: int = 7200

    # OIDC/SSO (M5.3)
    oidc_enabled: bool = False
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_uri: str | None = None
    oidc_allow_insecure_oidc: bool = False
    oidc_role_claim: str = "roles"
    oidc_default_role: str = "user"

    # Per-user quotas (M5.5)
    quota_token_budget_per_day: int | None = None
    quota_requests_per_hour: int | None = None
    quota_max_concurrent_sessions: int | None = None

    # Self-improvement loop (M4.1)
    self_improve_auto_commit: bool = False

    # Semantic tool index (M4.4)
    semantic_tool_index: bool = True

    # Track where config was loaded from (for display)
    config_file_path: Path | None = None

    # The model alias that is currently active (from models.default or CLI --model)
    active_model_alias: str | None = field(default=None, repr=False)

    # Embedding overrides populated from env vars (read by cogtrix.py)
    embedding_provider_override: str | None = None
    embedding_model_override: str | None = None

    # Research delegate settings
    research_delegate_enabled: bool = True
    research_delegate_timeout: int = 300
    research_delegate_cap_ratio: float = 0.85
    research_delegate_auto: bool = False
    """When True, research queries are delegated pre-flight when context exceeds the threshold."""
    research_delegate_auto_threshold: float = 0.50
    """Session context ratio (0–1) above which research_delegate_auto activates."""

    # Research diversity settings (M6.1)
    research_diversity: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "min_diversity_threshold": 0.5,
            "max_dominant_origin_ratio": 0.7,
            "require_contradiction_search": True,
        }
    )
    """Source diversity tracking and contradiction detection settings.

    Controls source diversity tracking and contradiction detection for research workflows.
    Helps identify when multiple sources trace back to the same origin, preventing
    overconfidence in claims supported only by sources from a single origin.
    """

    # Decision accountability settings (ADR-0052 M2/M3)
    decision_accountability_enabled: bool = False
    """Inject the self-debate prompt and parse structured output (ADR-0052).
    Off by default — opt in via decision_accountability.enabled in .cogtrix.yaml."""
    decision_accountability_min_confidence: float = 7.0
    """Minimum adjusted confidence (0–10) to proceed; below this the agent appends
    an uncertainty note when decision_accountability_report_uncertainty is True."""
    decision_accountability_require_counter_plan: bool = True
    """When True, the accountability prompt requires a counter-plan before acting."""
    decision_accountability_report_uncertainty: bool = True
    """When True, append an uncertainty note to responses where should_proceed is False."""

    # ── Task ownership classifier ─────────────────────────────────────────
    task_ownership_classifier_enabled: bool = True
    """When True (default), classify task ownership before graph invocation."""
    task_ownership_classifier_llm_fallback: bool = False
    """When True, invoke LLM for Layer 2 classification (off by default — adds latency)."""
    task_ownership_ambiguous_action: str = "ask"
    """How to handle AMBIGUOUS ownership. Values: 'ask' | 'inform' | 'execute'."""

    # ── Pre-action confirmation gate ─────────────────────────────────────────
    pre_action_confirmation_enabled: bool = False
    """When True, inject the pre-action confirmation prompt so the agent asks
    for consent before irreversible operations. Off by default — opt in via
    pre_action_confirmation.enabled in .cogtrix.yaml."""

    # ── Metrics endpoint security ────────────────────────────────────────────
    metrics_auth_enabled: bool = True
    """When True (default), the Prometheus metrics endpoint requires
    authentication. Disable only for in-cluster scrapers that cannot
    present a bearer token."""

    def to_execution_settings(self) -> ExecutionSettings:
        """Project agent-facing runtime knobs into an execution settings bundle."""
        return ExecutionSettings(
            context_compression=self.context_compression,
            compression_min_age=self.context_compression_min_age,
            compression_min_chars=self.context_compression_min_chars,
            context_max_messages=self.resolve_context_max_messages(),
            tier_cache_enabled=self.tier_cache_enabled,
            tool_context_limit_pct=self.tool_context_limit_pct,
            parallel_tool_execution=self.parallel_tool_execution,
            git_native=self.git_native,
            decision_accountability_enabled=self.decision_accountability_enabled,
            decision_accountability_report_uncertainty=(
                self.decision_accountability_report_uncertainty
            ),
            decision_accountability_min_confidence=self.decision_accountability_min_confidence,
            task_ownership_classifier_enabled=self.task_ownership_classifier_enabled,
            task_ownership_classifier_llm_fallback=self.task_ownership_classifier_llm_fallback,
            task_ownership_ambiguous_action=self.task_ownership_ambiguous_action,
            pre_action_confirmation_enabled=self.pre_action_confirmation_enabled,
        )

    # ── Service key accessors ─────────────────────────────────────
    # These provide a clean API for tool configuration code, reading
    # from the consolidated ``services`` dict.

    @property
    def openweather_api_key(self) -> str | None:
        return self.services.get("openweather", {}).get("api_key")

    @property
    def tavily_api_key(self) -> str | None:
        return self.services.get("tavily", {}).get("api_key")

    @property
    def exa_api_key(self) -> str | None:
        return self.services.get("exa", {}).get("api_key")

    @property
    def brave_api_key(self) -> str | None:
        return self.services.get("brave", {}).get("api_key")

    @property
    def searxng_url(self) -> str | None:
        return self.services.get("searxng", {}).get("url") or os.getenv("SEARXNG_URL")

    @property
    def serpapi_api_key(self) -> str | None:
        return self.services.get("serpapi", {}).get("api_key")

    @property
    def google_api_key(self) -> str | None:
        return self.services.get("google", {}).get("api_key")

    @property
    def google_cse_id(self) -> str | None:
        return self.services.get("google", {}).get("cse_id")

    @property
    def whatsapp_config(self) -> dict[str, Any]:
        """Return the full WhatsApp service config dict (may be empty)."""
        return self.services.get("whatsapp", {})

    @property
    def telegram_config(self) -> dict[str, Any]:
        """Return the full Telegram service config dict (may be empty)."""
        return self.services.get("telegram", {})

    @property
    def assistant_config(self) -> dict[str, Any]:
        """Return the full assistant service config dict (may be empty)."""
        return self.services.get("assistant", {})

    def dump_debug(self, log: Any) -> None:
        """Log all resolved configuration parameters at DEBUG level.

        Masks API keys and secrets.  Called once at startup when debug
        mode is active so the operator can see the exact config the
        application is using after all sources (file, env, CLI) have
        been merged.
        """
        if not log.isEnabledFor(logging.DEBUG):
            return

        def _mask(val: str | None) -> str:
            if not val:
                return "(none)"
            if len(val) < 10:
                return "***"
            return val[:3] + "***" + val[-4:]

        lines = [
            "=== Resolved configuration ===",
            f"  config_file:          {self.config_file_path or '(none)'}",
            f"  session:              {self.session}",
            f"  data_dir:             {self.data_dir}",
            f"  active_model_alias:   {self.active_model_alias or '(none)'}",
            f"  memory_mode:          {self.memory_mode}",
            f"  prompt_optimizer:     {self.prompt_optimizer}",
            f"  parallel_tool_exec:   {self.parallel_tool_execution}",
            f"  context_compression:  {self.context_compression}",
            f"    min_age:            {self.context_compression_min_age}",
            f"    min_chars:          {self.context_compression_min_chars}",
            f"    max_messages:       {self.context_max_messages}",
            f"    max_tokens:         {self.context_max_tokens}",
            f"    model:              {self.context_compression_model or '(default)'}",
            f"  delegate_enabled:     {self.delegate_enabled}",
            f"    timeout:            {self.delegate_default_timeout}",
            f"    allowed_models:     {self.delegate_allowed_models}",
            f"  research_delegate:    {self.research_delegate_enabled}",
            f"  debug:                {self.debug}",
            f"  verbose:              {self.verbose}",
            f"  verbosity:            {self.verbosity}",
            f"  log_file:             {self.log_file}",
        ]
        # Providers
        lines.append("  providers:")
        for name, pc in self.providers.items():
            lines.append(
                f"    {name}: type={pc.type}, "
                f"base_url={pc.get_base_url() or '(default)'}, "
                f"api_key={_mask(pc.api_key)}"
            )
        # Models
        lines.append("  models:")
        for name, mc in self.models.items():
            lines.append(
                f"    {name}: provider={mc.provider}, model={mc.model}, "
                f"temp={mc.temperature}, context_window={mc.context_window}, "
                f"max_tokens={mc.max_tokens}"
            )
        # RAG
        lines.append(
            f"  rag: docs_dir={self.rag.docs_dir}, "
            f"vectordb_dir={self.rag.vectordb_dir}, "
            f"chunk_size={self.rag.chunk_size}, "
            f"model={self.rag.model or '(none)'}"
        )
        # Memory modes
        if self.memory_modes:
            lines.append("  memory_modes:")
            for mode_name, mode_cfg in self.memory_modes.items():
                lines.append(f"    {mode_name}: {mode_cfg}")
        # MCP servers
        if self.mcp_servers:
            lines.append(f"  mcp_servers: {list(self.mcp_servers.keys())}")
        # Services (names only, no secrets)
        if self.services:
            svc_names = [
                f"{k}(api_key={_mask(v.get('api_key'))})" if "api_key" in v else k
                for k, v in self.services.items()
            ]
            lines.append(f"  services: {', '.join(svc_names)}")
        if self.cron:
            lines.append(f"  cron: {len(self.cron)} job(s)")

        log.debug("\n".join(lines))

    def resolve_data_path(self, subpath: str) -> Path:
        """Resolve a data subpath against ``data_dir``.

        Absolute paths are returned as-is.  Relative paths starting with
        the legacy ``data/`` prefix (from configs written before the
        ``data_dir`` option existed) are normalized to avoid double-nesting.
        Path traversal sequences (``..``) that escape ``data_dir`` are rejected.
        """
        subpath = str(subpath)
        # ── Early traversal guard: reject ``..`` in the raw string ───────
        # Pathlib normalises ``data/../foo`` to ``foo``, which can hide the
        # ``..`` before the legacy ``data/`` prefix is stripped below.
        # Checking the raw string first catches traversal attempts before
        # any manipulation, closing the ``data/../../../etc`` bypass window.
        if ".." in subpath.replace("\\", "/").split("/"):
            raise ConfigError(f"Path traversal detected in data path: {subpath!r}")
        p = Path(subpath)
        if p.is_absolute():
            return p
        # Legacy migration: configs written before data_dir existed may
        # contain paths like "data/vectordb".  Strip the "data/" prefix
        # so they resolve correctly under any data_dir value.
        data_dir = Path(self.data_dir)
        if subpath.startswith("data/") or subpath == "data":
            stripped = subpath[5:]  # len("data/") == 5
            result = data_dir / stripped if stripped else data_dir
        else:
            result = data_dir / subpath
        # Defense-in-depth: resolve-and-compare as secondary check.
        base_resolved = data_dir.resolve()
        result_resolved = result.resolve()
        if not result_resolved.is_relative_to(base_resolved):
            raise ConfigError(f"Path traversal detected in data path: {subpath!r}")
        return result_resolved

    def resolve_rag_index_dir(self, override: str | None = None) -> Path:
        """Canonical FAISS index directory for RAG: ``<vectordb_dir>/faiss_index``.

        Single source of truth so the CLI ingest path (``cogtrix.run_ingest``)
        and the query-tool config (``src.tools.configure.configure_rag_tool``)
        cannot drift apart. They DID drift in #2216: post-#1951 ``run_ingest``
        wrote the index straight to ``vectordb_dir`` while ``configure_rag_tool``
        configured the query side to read ``vectordb_dir/faiss_index`` — so
        ``query_knowledge_base`` never found a CLI-ingested index. Both callers
        now derive the directory from here.

        ``override`` is the optional CLI ``--vectordb-dir`` value; when absent
        the configured ``rag.vectordb_dir`` is used.
        """
        return self.resolve_data_path(override or self.rag.vectordb_dir) / "faiss_index"

    def get_provider_config(self, name: str | None = None) -> ProviderConfig:
        """Get configuration for a provider by name.

        Args:
            name: Provider name (uses active model's provider if None)

        Returns:
            ProviderConfig for the requested provider

        Raises:
            ValueError: If provider is not configured
        """
        provider_name = name or self.get_active_model().provider
        if provider_name in self.providers:
            return self.providers[provider_name]
        raise ValueError(f"Unknown provider: '{provider_name}'. Available: {self.list_providers()}")

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        return sorted(self.providers.keys())

    def get_model_config(self, name: str | None = None) -> ModelConfig | None:
        """Look up a model in the models registry. Returns None if not found."""
        model_name = name or self.active_model_alias
        if model_name and model_name in self.models:
            return self.models[model_name]
        return None

    def resolve_embedding_config(self) -> tuple[str, str | None, str | None, str | None]:
        """Resolve RAG embedding model to (provider_type, model, base_url, api_key).

        Looks up rag.model in the models registry, then resolves provider
        connection details. Falls back to the active provider if rag.model
        is not set or not found.

        Raises:
            ConfigError: If no models are configured.
            ValueError: If the active provider is not configured.
        """
        import logging

        _log = logging.getLogger("cogtrix")
        model_name = self.rag.model
        if model_name and model_name in self.models:
            mc = self.models[model_name]
            pc = self.providers.get(mc.provider)
            if pc:
                self._warn_if_not_embedding_capable(
                    pc.type, source=f"rag.model '{model_name}' (provider '{mc.provider}')"
                )
                return pc.type, mc.model, pc.get_base_url(), pc.api_key
            _log.warning(
                "rag.model '%s' references provider '%s' which is not configured; "
                "falling back to active provider",
                model_name,
                mc.provider,
            )
        elif model_name:
            # #2066: rag.model is set but not defined in the models registry. The
            # silent fallback hides the typo, so surface it explicitly.
            _log.warning(
                "rag.model '%s' is not defined in the models registry; falling back to the "
                "active provider for embeddings. Define the embedding alias or fix the name.",
                model_name,
            )
        # Check if there's an active provider before calling get_active_provider()
        # to provide a clearer error message
        try:
            pc = self.get_active_provider()
        except (ConfigError, ValueError) as e:
            # Re-raise with more context
            raise type(e)(
                f"Cannot resolve embedding config: {e}. "
                "Ensure at least one model and provider are configured via /setup or the config file."
            ) from e
        self._warn_if_not_embedding_capable(pc.type, source="the active provider")
        return pc.type, None, pc.get_base_url(), pc.api_key

    def _warn_if_not_embedding_capable(self, provider_type: str, *, source: str) -> None:
        """#2066: warn clearly when the resolved provider cannot produce embeddings.

        Non-fatal — resolution still returns the config — but it turns the
        otherwise-opaque downstream failure (e.g. Anthropic's NotImplementedError,
        or a 404 from a chat-only endpoint) into an actionable startup/ingest log.
        """
        try:
            from cogtrix_core.providers import is_embeddings_available
        except Exception:  # pragma: no cover - defensive
            return
        if not is_embeddings_available(provider_type):
            import logging

            logging.getLogger("cogtrix").warning(
                "RAG embeddings: %s resolves to provider type '%s', which does not support "
                "embeddings (or its package is not installed); ingestion will fail. Point "
                "rag.model at an OpenAI/Ollama/Google embedding model.",
                source,
                provider_type,
            )

    def find_model_entry(self, target: str) -> "tuple[str | None, ModelConfig | None]":
        """Resolve *target* to a (canonical_alias, ModelConfig) pair.

        Resolution order:
        1. Exact alias key in self.models.
        2. Scan self.models values for .model == target (first match).

        Returns (alias_or_None, ModelConfig_or_None). Both are None when not found.
        """
        if target in self.models:
            return target, self.models[target]
        for alias, mc in self.models.items():
            if mc.model == target:
                return alias, mc
        return None, None

    def get_active_model(self) -> "ModelConfig":
        """Resolve the active model from the models registry.

        Uses ``active_model_alias`` if set. Falls back to the first model
        in the registry when no alias is configured.

        Raises:
            ConfigError: If the alias is set but not found, or no models exist.
        """
        if self.active_model_alias:
            mc = self.models.get(self.active_model_alias)
            if mc is None:
                cfg_hint = f" (config: {self.config_file_path})" if self.config_file_path else ""
                raise ConfigError(
                    f"Model '{self.active_model_alias}' not found in models registry{cfg_hint}. "
                    f"Available: {', '.join(sorted(self.models)) or '(none)'}. "
                    f"Run /setup or edit your config file to add this model."
                )
            return mc
        if self.models:
            first_alias = next(iter(self.models))
            return self.models[first_alias]
        cfg_hint = f" (config: {self.config_file_path})" if self.config_file_path else ""
        raise ConfigError(
            f"No models configured{cfg_hint}. Run /setup or cogtrix.py --setup to configure a model."
        )

    def get_active_provider(self) -> "ProviderConfig":
        """Get the provider config for the active model.

        Resolves through ``get_active_model().provider``.
        """
        mc = self.get_active_model()
        return self.get_provider_config(mc.provider)

    def resolve_llm_config(self) -> "tuple[ProviderConfig, ModelConfig]":
        """Resolve the active LLM configuration as a (provider, model) pair.

        Returns a **copy** of the provider config so the original in
        ``self.providers`` is never mutated. The ``ModelConfig`` is returned
        as-is (it has no mutable shared state).
        """
        from copy import copy

        mc = self.get_active_model()
        pc = copy(self.get_provider_config(mc.provider))
        return pc, mc

    def resolve_context_max_tokens(self) -> int:
        """Effective compression-budget cap, scaled to the active model (#2360).

        When the operator sets ``context_max_tokens`` explicitly (including 0 =
        disabled), that value wins. Otherwise the budget scales with the active
        model's context window — ``max(40_000, window // 2)`` — so small/medium
        windows keep today's 40k floor while big-window models (e.g. Kimi 262k)
        get a proportionate budget instead of being compressed to a flat 40k on
        every turn.

        The ~50% factor is grounded in the ``role_sysadmin`` context-heavy sweep:
        at 50% of a 262k window the task passed 3/3 with zero context eviction,
        vs task failures + heavy eviction (and ~2× wall-clock) at 15-20% (a flat
        40k). 76% removed only a rare heavy-tail compression at +50% tokens/turn —
        a poor trade — so 50% is the recommended point. ``max(40_000, …)`` ensures
        no regression for models whose window is <= 80k (they keep 40k).
        """
        # Operator override wins (set only when the config key was present).
        if not getattr(self, "context_max_tokens_is_default", True):
            return self.context_max_tokens
        try:
            _, mc = self.resolve_llm_config()
            window = mc.context_window or ModelConfig.DEFAULT_CONTEXT_WINDOW
        except Exception:  # noqa: BLE001 — never crash on config resolution; fall back to 40k
            window = ModelConfig.DEFAULT_CONTEXT_WINDOW
        return max(40_000, round(window / 2))

    #: Token budget the default 200-message cap was calibrated against; the cap
    #: scales up in proportion to how much larger the resolved token budget is.
    _CONTEXT_MAX_MESSAGES_BASELINE_TOKENS = 40_000

    def resolve_context_max_messages(self) -> int:
        """Effective message-count cap, scaled to the token budget (#2397).

        A flat 200 messages is right for a ~40k budget but far too tight for a
        large-window model: 200 messages fit comfortably under the token cap, so
        the message cap becomes the *binding* constraint and evicts recent context
        on every turn (the amnesia in #2397). Scale the default in proportion to
        the resolved token budget so the TOKEN cap is the governing gate and the
        message cap stays a proportional backstop against unbounded list growth.

        An explicit operator value wins (including 0 = disabled). Small/medium
        budgets keep exactly 200 (the ratio floors at 1.0), so no regression for
        models whose resolved budget is <= the 40k baseline.
        """
        # Operator override wins (set only when the config key was present).
        if not getattr(self, "context_max_messages_is_default", True):
            return self.context_max_messages
        # 0 = message cap disabled; honor it (the token cap alone governs trimming).
        if self.context_max_messages <= 0:
            return self.context_max_messages
        resolved_tokens = self.resolve_context_max_tokens()
        if resolved_tokens <= 0:
            return self.context_max_messages
        ratio = max(1.0, resolved_tokens / self._CONTEXT_MAX_MESSAGES_BASELINE_TOKENS)
        return max(self.context_max_messages, round(self.context_max_messages * ratio))

    def resolve_llm_config_for(self, alias: str) -> "tuple[ProviderConfig, ModelConfig]":
        """Resolve a named model alias to a (provider, model) pair.

        Used by compression, delegate, knowledge extraction, and other
        subsystems that need to create an LLM for a specific model alias.

        Supports both registry aliases and ``"provider/model"`` shorthand.

        Raises:
            ConfigError: If the alias is not found and not in shorthand format.
        """
        from copy import copy

        # Check models registry first
        mc = self.models.get(alias)
        if mc is not None:
            pc = copy(self.get_provider_config(mc.provider))
            return pc, mc

        # Try "provider/model" shorthand
        if "/" in alias:
            provider_name, model_name = alias.split("/", 1)
            pc = copy(self.get_provider_config(provider_name))
            mc = ModelConfig(provider=provider_name, model=model_name)
            return pc, mc

        raise ConfigError(
            f"Model alias '{alias}' not found in models registry and is not "
            f"in 'provider/model' format. Available aliases: "
            f"{', '.join(sorted(self.models)) or '(none)'}"
        )


def find_config_file() -> Path | None:
    """
    Search for a Cogtrix configuration file.

    Both JSON and YAML formats are supported.  Within each directory
    JSON is checked first, then ``.yml``, then ``.yaml``.

    Search order (first found wins):
        1. ``./.cogtrix.json``
        2. ``./.cogtrix.yml``  /  ``./.cogtrix.yaml``
        3. ``~/.cogtrix.json``
        4. ``~/.cogtrix.yml``  /  ``~/.cogtrix.yaml``
        5. ``~/.config/cogtrix/cogtrix.json``
        6. ``~/.config/cogtrix/cogtrix.yml``  /  ``~/.config/cogtrix/cogtrix.yaml``

    Returns:
        Path to config file if found, None otherwise.
        Absence of config file is not an error.
    """
    cwd = Path.cwd()
    home = Path.home()
    xdg_dir = home / ".config" / "cogtrix"

    # Directories paired with their candidate filenames
    search: list[tuple[Path, list[str]]] = [
        (cwd, _CONFIG_NAMES),
        (home, _CONFIG_NAMES),
        (xdg_dir, _XDG_CONFIG_NAMES),
    ]
    for directory, names in search:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


def load_config(cli_args=None, *, unset_secrets: bool = True) -> Config:
    """
    Load configuration with priority:
    CLI args > Environment variables > Config file > Defaults

    If ``cli_args`` has a ``config_file`` attribute, that path is used
    directly instead of the normal search.

    Args:
        cli_args: Parsed command line arguments (argparse namespace)
        unset_secrets: When True (default), config-authoritative secret env
            vars are removed from ``os.environ`` after being copied into the
            returned ``Config`` (#2223). Pass False in tests that must assert
            environment contents after a load.

    Returns:
        Config object with resolved settings

    Raises:
        ConfigError: If an explicit ``--config-file`` does not exist or
            has invalid syntax.
    """
    config = Config()

    # 1. Determine config file (explicit path has priority over search)
    explicit_path = getattr(cli_args, "config_file", None) if cli_args else None
    if not explicit_path:
        explicit_path = os.environ.get("COGTRIX_CONFIG_FILE")
    if explicit_path:
        config_file = Path(explicit_path)
        if not config_file.exists():
            raise ConfigError(f"Config file not found: {config_file}")
    else:
        config_file = find_config_file()

    # 2. Load from config file (lowest priority)
    if config_file:
        _apply_config_file(config, config_file)
        config.config_file_path = config_file

    # 3. Override with environment variables (medium priority)
    _apply_env_vars(config, unset_secrets=unset_secrets)

    # 4. Override with CLI arguments (highest priority)
    if cli_args:
        _apply_cli_args(config, cli_args)

    # 5. Resolve model alias against the models registry
    _resolve_model(config)

    # 6. If active_model_alias is still None, generate a default from first provider
    if config.active_model_alias is None and config.providers:
        first_prov = next(iter(config.providers.values()))
        from cogtrix_core.providers import get_default_model

        default_model = get_default_model(first_prov.type)
        _synth = f"{first_prov.name}/{default_model}"
        if _synth not in config.models:
            config.models[_synth] = ModelConfig(provider=first_prov.name, model=default_model)
        config.active_model_alias = _synth

    return config


# ── Process-wide cached configuration (#2101) ────────────────────────────────
# Environment variables must be read EXACTLY ONCE per process. ``load_config()``
# re-applies ``os.environ`` on every call, so runtime paths that re-invoke it
# (RAG ingest, the DB-engine default-URL resolver, the API CORS resolver, the
# weather / WhatsApp / Telegram tool loaders) would re-read the environment — and,
# after the #2102/#2223 unset, observe missing secrets. ``get_cached_config()``
# resolves config ONCE and returns that instance to every later caller, so the
# environment is read a single time and resolution no longer depends on the env
# still being present. The admin ``reload_config`` endpoint is the ONLY sanctioned
# re-read path, via ``reload_cached_config()``.
_CACHED_CONFIG: "Config | None" = None
_CACHED_CONFIG_LOCK = threading.Lock()


def get_cached_config() -> "Config":
    """Return the process-wide resolved :class:`Config`, reading os.environ once.

    Runtime/post-startup callers MUST use this instead of :func:`load_config` so
    environment variables are read exactly once (#2101). The first call resolves
    config (no CLI args — this is the post-startup accessor); every later call
    returns the same instance. Thread-safe (double-checked lock) because the API
    resolves config from worker threads.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        with _CACHED_CONFIG_LOCK:
            if _CACHED_CONFIG is None:
                _CACHED_CONFIG = load_config()
    return _CACHED_CONFIG


def reload_cached_config() -> "Config":
    """Force a fresh resolution and replace the process cache (#2101).

    The ONLY sanctioned re-read path — the admin ``reload_config`` endpoint. Every
    other consumer reuses :func:`get_cached_config`.
    """
    global _CACHED_CONFIG
    with _CACHED_CONFIG_LOCK:
        _CACHED_CONFIG = load_config()
    return _CACHED_CONFIG


def reset_cached_config() -> None:
    """Drop the cached Config so the next :func:`get_cached_config` re-resolves.

    For test isolation only — never call in app code (the cache is meant to hold
    the single process-lifetime resolution).
    """
    global _CACHED_CONFIG
    with _CACHED_CONFIG_LOCK:
        _CACHED_CONFIG = None


def _resolve_model(config: Config) -> None:
    """Resolve active_model_alias against the models registry.

    If the alias matches a key in config.models, the resolution succeeds.
    If not found but models exist, log a warning.
    """
    if not config.active_model_alias:
        return

    if config.active_model_alias in config.models:
        return

    # Alias not found — may be a literal model name; try to find it
    for alias, mc in config.models.items():
        if mc.model == config.active_model_alias:
            config.active_model_alias = alias
            return

    if config.models:
        _log.warning(
            "active_model_alias '%s' not found in models registry",
            config.active_model_alias,
        )


def _is_yaml_file(path: Path) -> bool:
    """Return True if the path has a YAML extension."""
    return path.suffix.lower() in (".yml", ".yaml")


def _parse_config_file(path: Path) -> dict[str, Any]:
    """
    Read and parse a config file (JSON or YAML).

    Returns:
        Parsed dict from the config file.

    Raises:
        ConfigError: On read errors or invalid syntax.
    """
    content = ""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        raise ConfigError(f"Cannot read config file {path}: {e}") from None

    if not content.strip():
        return {}

    if _is_yaml_file(path):
        return _parse_yaml(content, path)
    return _parse_json(content, path)


def _parse_json(content: str, path: Path) -> dict[str, Any]:
    """Parse JSON content with detailed error reporting."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        lines = content.split("\n") if content else []
        error_lines = [
            f"Invalid JSON in config file: {path}",
            f"Error: {e.msg} at line {e.lineno}, column {e.colno}",
        ]

        # Show context around the error
        if lines and 1 <= e.lineno <= len(lines):
            error_lines.append("")
            start = max(0, e.lineno - 3)
            for i in range(start, e.lineno):
                prefix = ">>>" if i == e.lineno - 1 else "   "
                error_lines.append(f"{prefix} {i + 1:3d} | {lines[i]}")
            pointer = " " * (e.colno + 6) + "^"
            error_lines.append(pointer)
            for i in range(e.lineno, min(len(lines), e.lineno + 2)):
                error_lines.append(f"    {i + 1:3d} | {lines[i]}")

        error_lines.append("")
        error_lines.append("Common JSON errors:")
        error_lines.append("  - Missing comma between object properties")
        error_lines.append("  - Trailing comma after last property")
        error_lines.append("  - Unquoted strings or keys")
        error_lines.append("  - Single quotes instead of double quotes")
        error_lines.append("")
        error_lines.append(f"Fix the JSON syntax in: {path}")

        raise ConfigError("\n".join(error_lines)) from None

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a JSON object (got {type(data).__name__}): {path}"
        )
    return data


def _parse_yaml(content: str, path: Path) -> dict[str, Any]:
    """Parse YAML content with detailed error reporting."""
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        error_lines = [f"Invalid YAML in config file: {path}"]
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark  # type: ignore[union-attr]
            error_lines.append(f"Error at line {mark.line + 1}, column {mark.column + 1}")
        if hasattr(e, "problem"):
            error_lines.append(f"  {e.problem}")  # type: ignore[union-attr]

        # Show context around the error
        lines = content.split("\n")
        if hasattr(e, "problem_mark") and lines:
            mark = e.problem_mark  # type: ignore[union-attr]
            error_lines.append("")
            start = max(0, mark.line - 2)
            end = min(len(lines), mark.line + 3)
            for i in range(start, end):
                prefix = ">>>" if i == mark.line else "   "
                error_lines.append(f"{prefix} {i + 1:3d} | {lines[i]}")
            if mark.column > 0:
                error_lines.append(" " * (mark.column + 7) + "^")

        error_lines.append("")
        error_lines.append("Common YAML errors:")
        error_lines.append("  - Incorrect indentation (use spaces, not tabs)")
        error_lines.append("  - Missing colon after key name")
        error_lines.append("  - Unquoted special characters (wrap in quotes)")
        error_lines.append("")
        error_lines.append(f"Fix the YAML syntax in: {path}")

        raise ConfigError("\n".join(error_lines)) from None

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a YAML mapping (got {type(data).__name__}): {path}"
        )
    return data


def _apply_config_file(config: Config, path: Path) -> None:
    """Apply settings from configuration file (JSON or YAML).

    Raises:
        ConfigError: If the config file has invalid syntax.
    """
    data = _parse_config_file(path)

    # ── General settings (top-level) ─────────────────────────────────
    # Capture legacy top-level provider/model for post-processing below
    _legacy_provider = data.get("provider")
    _legacy_model = data.get("model")
    if "session" in data:
        config.session = data["session"]
    if "data_dir" in data:
        config.data_dir = str(data["data_dir"])

    # ── Providers ──────────────────────────────────────────────────
    # Preferred key: "providers"; alias: "inference"
    if "providers" in data and "inference" in data:
        _log.warning("Config has both 'providers' and 'inference' keys; using 'providers'")
    providers_data = data.get("providers") if "providers" in data else data.get("inference")
    if isinstance(providers_data, dict):
        _parse_providers_section(config, providers_data)

    # ── Models registry ────────────────────────────────────────────
    # Preferred key: "models"; alias: "model_aliases"
    if "models" in data and "model_aliases" in data:
        _log.warning("Config has both 'models' and 'model_aliases' keys; using 'models'")
    models_data = data.get("models") if "models" in data else data.get("model_aliases")
    if isinstance(models_data, dict):
        _parse_models_section(config, models_data)

    # ── Services (external APIs) ──────────────────────────────────
    if "services" in data and isinstance(data["services"], dict):
        for svc_name, svc_cfg in data["services"].items():
            if isinstance(svc_cfg, dict):
                config.services[svc_name] = svc_cfg

    # Legacy format: top-level keys like "openweather", "tavily", etc.
    _LEGACY_SERVICE_KEYS = (
        "openweather",
        "tavily",
        "exa",
        "brave",
        "serpapi",
        "google",
    )
    for key in _LEGACY_SERVICE_KEYS:
        if key in data and isinstance(data[key], dict):
            if key not in config.services:
                config.services[key] = data[key]

    # ── Cron jobs ───────────────────────────────────────────────────
    if "cron" in data:
        cron_data = data["cron"]
        if isinstance(cron_data, list):
            parsed_cron: list[dict[str, Any]] = []
            for idx, item in enumerate(cron_data):
                if not isinstance(item, dict):
                    _log.warning(
                        "Skipping cron[%d]: expected mapping, got %s", idx, type(item).__name__
                    )
                    continue
                schedule = item.get("schedule")
                prompt = item.get("prompt")
                if not isinstance(schedule, str) or not schedule.strip():
                    _log.warning("Skipping cron[%d]: missing or invalid schedule", idx)
                    continue
                if not isinstance(prompt, str) or not prompt.strip():
                    _log.warning("Skipping cron[%d]: missing or invalid prompt", idx)
                    continue
                context = str(item.get("context", "fresh")).strip().lower()
                if context not in {"fresh", "inherit"}:
                    _log.warning(
                        "cron[%d].context must be 'fresh' or 'inherit'; using 'fresh'",
                        idx,
                    )
                    context = "fresh"
                parsed_cron.append(
                    {
                        "name": str(item.get("name", "")),
                        "schedule": schedule,
                        "prompt": prompt,
                        "context": context,
                    }
                )
            config.cron = parsed_cron
        else:
            _log.warning("cron must be a list of mappings, ignoring")

    # ── Memory settings ───────────────────────────────────────────
    if "memory" in data and isinstance(data["memory"], dict):
        memory_cfg = data["memory"]
        if "mode" in memory_cfg:
            config.memory_mode = memory_cfg["mode"]
        if "modes" in memory_cfg and isinstance(memory_cfg["modes"], dict):
            config.memory_modes = dict(memory_cfg["modes"])
            mode = config.memory_mode
            if mode in memory_cfg["modes"]:
                config.memory_config = memory_cfg["modes"][mode]

    # ── Delegate tool settings ────────────────────────────────────
    if "delegate" in data and isinstance(data["delegate"], dict):
        delegate_cfg = data["delegate"]
        if "enabled" in delegate_cfg:
            config.delegate_enabled = bool(delegate_cfg["enabled"])
        if "default_timeout" in delegate_cfg:
            val = _safe_int(delegate_cfg["default_timeout"], "delegate.default_timeout")
            if val is not None and val > 0:
                config.delegate_default_timeout = val
            elif val is not None:
                _log.warning(
                    "delegate.default_timeout must be > 0, using default %d",
                    config.delegate_default_timeout,
                )
        if "allowed_providers" in delegate_cfg:
            config.delegate_allowed_providers = delegate_cfg["allowed_providers"]
        if "allowed_models" in delegate_cfg:
            config.delegate_allowed_models = delegate_cfg["allowed_models"]
        # Backward compat: delegate.model_aliases → config.models (only if models not set yet)
        if "model_aliases" in delegate_cfg and not config.models:
            _parse_models_section(config, delegate_cfg["model_aliases"])

    # ── Prompt optimizer ─────────────────────────────────────────
    if "prompt_optimizer" in data:
        config.prompt_optimizer = bool(data["prompt_optimizer"])

    if "adaptive_memory" in data:
        config.adaptive_memory = bool(data["adaptive_memory"])

    if "auto_route" in data:
        config.auto_route = bool(data["auto_route"])
    if "quick_mode" in data:
        config.quick_mode = bool(data["quick_mode"])
    if "git_native" in data:
        config.git_native = bool(data["git_native"])
    if "banner" in data:
        banner_val = data["banner"]
        if banner_val is None or (isinstance(banner_val, str) and banner_val.strip() == ""):
            config.banner = "off"
        else:
            _banner_val = str(banner_val).lower().strip()
            if _banner_val in ("full", "compact", "off", "none", "false", "0"):
                config.banner = (
                    "off" if _banner_val in ("off", "none", "false", "0") else _banner_val
                )
    if "theme" in data:
        val = data["theme"]
        if isinstance(val, str) and val in ("default", "minimal", "dracula"):
            config.theme = val
    if "auto_route_fast_model" in data:
        config.auto_route_fast_model = (
            str(data["auto_route_fast_model"]) if data["auto_route_fast_model"] else None
        )

    if "parallel_tool_execution" in data:
        config.parallel_tool_execution = bool(data["parallel_tool_execution"])

    if "tool_trust" in data and isinstance(data["tool_trust"], dict):
        _valid_trust = {"always", "ask", "deny"}
        config.tool_trust = {
            str(k): str(v).lower()
            for k, v in data["tool_trust"].items()
            if str(v).lower() in _valid_trust
        }

    if "api_dangerous_tools" in data:
        val = data["api_dangerous_tools"]
        if isinstance(val, bool):
            config.api_dangerous_tools = val
        else:
            _log.warning("api_dangerous_tools must be a boolean, ignoring")

    if "enable_datascience_modules" in data:
        val = data["enable_datascience_modules"]
        if isinstance(val, bool):
            config.enable_datascience_modules = val
        else:
            _log.warning("enable_datascience_modules must be a boolean, ignoring")

    if "profiles" in data and isinstance(data["profiles"], dict):
        config.profiles = {
            str(k): dict(v) if isinstance(v, dict) else {} for k, v in data["profiles"].items()
        }

    # ── Verbosity / debug settings ───────────────────────────────
    if "verbosity" in data:
        val = _safe_int(data["verbosity"], "verbosity")
        if val is not None and 0 <= val <= 3:
            config.verbosity = val
            if val >= 1:
                config.debug = True
            if val >= 2:
                config.verbose = True
        elif val is not None:
            _log.warning("verbosity must be 0–3, using default 0")
    elif data.get("debug"):
        # Legacy: debug: true → verbosity 1
        config.debug = True
        if config.verbosity == 0:
            config.verbosity = 1

    # ── Allowed write paths ──────────────────────────────────────
    if "allowed_write_paths" in data:
        val = data["allowed_write_paths"]
        if isinstance(val, str):
            config.allowed_write_paths = [val]
        elif isinstance(val, list):
            config.allowed_write_paths = [str(p) for p in val]
        else:
            _log.warning("allowed_write_paths must be a string or list, ignoring")

    # ── Allowed read paths ──────────────────────────────────────
    if "allowed_read_paths" in data:
        val = data["allowed_read_paths"]
        if isinstance(val, str):
            config.allowed_read_paths = [val]
        elif isinstance(val, list):
            config.allowed_read_paths = [str(p) for p in val]
        else:
            _log.warning("allowed_read_paths must be a string or list, ignoring")
    # ── Shell curl/wget allowed domains ──────────────────────────
    # Preferred location: under the ``shell`` block (``shell.curl_wget_allowed_domains``).
    # Legacy top-level key ``shell_curl_wget_allowed_domains`` is still accepted
    # for backward compatibility but triggers a deprecation warning.
    _shell_block = data.get("shell")
    _shell_block_domains = None
    if isinstance(_shell_block, dict):
        _shell_block_domains = _shell_block.get("curl_wget_allowed_domains")

    _legacy_domains = data.get("shell_curl_wget_allowed_domains")

    def _parse_domains(val: Any, key_name: str) -> list[str] | None:
        if isinstance(val, str):
            return [val]
        elif isinstance(val, list):
            return [str(d) for d in val]
        else:
            _log.warning("%s must be a string or list, ignoring", key_name)
            return None

    if _shell_block_domains is not None and _legacy_domains is not None:
        _log.warning(
            "Config has both 'shell.curl_wget_allowed_domains' and "
            "top-level 'shell_curl_wget_allowed_domains'; "
            "using 'shell.curl_wget_allowed_domains'"
        )

    if _shell_block_domains is not None:
        parsed = _parse_domains(_shell_block_domains, "shell.curl_wget_allowed_domains")
        if parsed is not None:
            config.shell_curl_wget_allowed_domains = parsed
    elif _legacy_domains is not None:
        parsed = _parse_domains(_legacy_domains, "shell_curl_wget_allowed_domains")
        if parsed is not None:
            config.shell_curl_wget_allowed_domains = parsed
            _log.warning(
                "Top-level 'shell_curl_wget_allowed_domains' is deprecated; "
                "move it under 'shell:' as 'curl_wget_allowed_domains'"
            )

    # ── Shell operator-controlled policy extensions (#2392) ──────────
    # Opt-in, under the ``shell`` block. Both default empty (locked-down).

    def _parse_str_list(val: Any, key_name: str) -> list[str] | None:
        if isinstance(val, str):
            return [val]
        elif isinstance(val, list):
            return [str(x) for x in val]
        else:
            _log.warning("%s must be a string or list, ignoring", key_name)
            return None

    if isinstance(_shell_block, dict):
        _extra_safe = _shell_block.get("extra_safe_commands")
        if _extra_safe is not None:
            parsed_safe = _parse_str_list(_extra_safe, "shell.extra_safe_commands")
            if parsed_safe is not None:
                config.shell_extra_safe_commands = parsed_safe
        _allow_pats = _shell_block.get("allow_patterns")
        if _allow_pats is not None:
            parsed_pats = _parse_str_list(_allow_pats, "shell.allow_patterns")
            if parsed_pats is not None:
                config.shell_allow_patterns = parsed_pats

    # ── Plugin tool directories ───────────────────────────────────
    if "tool_dirs" in data:
        val = data["tool_dirs"]
        if isinstance(val, str):
            config.tool_dirs = [val]
        elif isinstance(val, list):
            config.tool_dirs = [str(p) for p in val]
        else:
            _log.warning("tool_dirs must be a string or list, ignoring")

    # ── Context compression ──────────────────────────────────────
    if "context_compression" in data:
        cc = data["context_compression"]
        if isinstance(cc, dict):
            config.context_compression = bool(cc.get("enabled", True))
            if "min_age" in cc:
                val = _safe_int(cc["min_age"], "context_compression.min_age")
                if val is not None and val >= 0:
                    config.context_compression_min_age = val
                elif val is not None:
                    _log.warning(
                        "context_compression.min_age must be >= 0, using default %d",
                        config.context_compression_min_age,
                    )
            if "min_chars" in cc:
                val = _safe_int(cc["min_chars"], "context_compression.min_chars")
                if val is not None and val >= 0:
                    config.context_compression_min_chars = val
                elif val is not None:
                    _log.warning(
                        "context_compression.min_chars must be >= 0, using default %d",
                        config.context_compression_min_chars,
                    )
            if "emergency_threshold" in cc:
                val = cc["emergency_threshold"]
                if isinstance(val, (int, float)) and 0 < val <= 1:
                    config.context_compression_emergency_threshold = float(val)
            if "human_msg_max_chars" in cc:
                val = _safe_int(
                    cc["human_msg_max_chars"], "context_compression.human_msg_max_chars"
                )
                if val is not None and val >= 0:
                    config.context_compression_human_msg_max_chars = val
            if "model" in cc:
                config.context_compression_model = str(cc["model"])
            # ── Tiered Context Cache keys ──────────────────────────────
            if "tiered_cache" in cc:
                config.tier_cache_enabled = bool(cc["tiered_cache"])
            for _frac_key, _frac_attr in (
                ("tier0_fraction", "tier0_fraction"),
                ("tier1_fraction", "tier1_fraction"),
                ("tier2_fraction", "tier2_fraction"),
            ):
                if _frac_key in cc:
                    _fval = _safe_float(cc[_frac_key], f"context_compression.{_frac_key}")
                    if _fval is not None:
                        if not (0.01 <= _fval <= 0.95):
                            raise ConfigError(
                                f"context_compression.{_frac_key} must be in [0.01, 0.95], "
                                f"got {_fval}"
                            )
                        setattr(config, _frac_attr, _fval)
            # Validate that the three tier fractions sum to <= 1.0
            _frac_sum = config.tier0_fraction + config.tier1_fraction + config.tier2_fraction
            if _frac_sum > 1.0 + 1e-9:
                raise ConfigError(
                    f"context_compression tier fractions must sum to <= 1.0, "
                    f"got {_frac_sum:.4f} "
                    f"(tier0={config.tier0_fraction}, tier1={config.tier1_fraction}, "
                    f"tier2={config.tier2_fraction})"
                )
        else:
            config.context_compression = bool(cc)

    # ── Search quality heuristic thresholds (#1593, Option B) ────
    if "search_quality" in data and isinstance(data["search_quality"], dict):
        sq = data["search_quality"]
        if "min_url_count" in sq:
            val = _safe_int(sq["min_url_count"], "search_quality.min_url_count")
            if val is not None and val >= 1:
                config.search_quality_min_url_count = val
            else:
                _log.warning(
                    "search_quality.min_url_count must be >= 1, using default %d",
                    config.search_quality_min_url_count,
                )
        if "min_chars" in sq:
            val = _safe_int(sq["min_chars"], "search_quality.min_chars")
            if val is not None and val >= 0:
                config.search_quality_min_chars = val
            else:
                _log.warning(
                    "search_quality.min_chars must be >= 0, using default %d",
                    config.search_quality_min_chars,
                )

    # ── Context message cap ──────────────────────────────────────
    if "context_max_messages" in data:
        val = _safe_int(data["context_max_messages"], "context_max_messages")
        if val is not None and val >= 0:
            config.context_max_messages = val
            # Operator set it explicitly → resolve_context_max_messages() must NOT
            # auto-scale over their choice (#2397).
            config.context_max_messages_is_default = False
        elif val is not None:
            _log.warning(
                "context_max_messages must be >= 0, using default %d",
                config.context_max_messages,
            )

    if "context_max_tokens" in data:
        val = _safe_int(data["context_max_tokens"], "context_max_tokens")
        if val is not None and val >= 0:
            config.context_max_tokens = val
            # Operator set it explicitly → resolve_context_max_tokens() must NOT
            # auto-scale over their choice (#2360).
            config.context_max_tokens_is_default = False
        elif val is not None:
            _log.warning(
                "context_max_tokens must be >= 0, using default %d",
                config.context_max_tokens,
            )

    # ── MCP servers ──────────────────────────────────────────────
    if "mcp_servers" in data and isinstance(data["mcp_servers"], dict):
        config.mcp_servers = dict(data["mcp_servers"])

    # ── Agent configurations ──────────────────────────────────────
    if "agents" in data and isinstance(data["agents"], dict):
        config.agents = dict(data["agents"])

    # ── RAG settings ──────────────────────────────────────────────
    if "rag" in data and isinstance(data["rag"], dict):
        rag_cfg = data["rag"]
        if "docs_dir" in rag_cfg:
            config.rag.docs_dir = rag_cfg["docs_dir"]
        if "vectordb_dir" in rag_cfg:
            config.rag.vectordb_dir = rag_cfg["vectordb_dir"]
        if "chunk_size" in rag_cfg:
            val = _safe_int(rag_cfg["chunk_size"], "rag.chunk_size")
            if val is not None and val > 0:
                config.rag.chunk_size = val
            elif val is not None:
                _log.warning(
                    "rag.chunk_size must be > 0, using default %d",
                    config.rag.chunk_size,
                )
        if "chunk_overlap" in rag_cfg:
            val = _safe_int(rag_cfg["chunk_overlap"], "rag.chunk_overlap")
            if val is not None and val >= 0:
                config.rag.chunk_overlap = val
            elif val is not None:
                _log.warning(
                    "rag.chunk_overlap must be >= 0, using default %d",
                    config.rag.chunk_overlap,
                )
        if "model" in rag_cfg:
            config.rag.model = rag_cfg["model"]
        if "score_threshold" in rag_cfg:
            fval = _safe_float(rag_cfg["score_threshold"], "rag.score_threshold")
            if fval is not None and 0.0 <= fval <= 1.0:
                config.rag.score_threshold = fval
            elif fval is not None:
                _log.warning("rag.score_threshold must be in [0.0, 1.0], using default")
        # #1981: BM25 hybrid retrieval flags.
        if "build_bm25_sidecar" in rag_cfg:
            config.rag.build_bm25_sidecar = bool(rag_cfg["build_bm25_sidecar"])
        if "use_bm25_hybrid" in rag_cfg:
            config.rag.use_bm25_hybrid = bool(rag_cfg["use_bm25_hybrid"])
        if "bm25_rrf_k" in rag_cfg:
            kval = _safe_int(rag_cfg["bm25_rrf_k"], "rag.bm25_rrf_k")
            if kval is not None and kval > 0:
                config.rag.bm25_rrf_k = kval
            elif kval is not None:
                _log.warning(
                    "rag.bm25_rrf_k must be > 0, using default %d",
                    config.rag.bm25_rrf_k,
                )
        if config.rag.chunk_overlap >= config.rag.chunk_size:
            _log.warning(
                "rag.chunk_overlap (%d) must be less than rag.chunk_size (%d); "
                "resetting chunk_overlap to default",
                config.rag.chunk_overlap,
                config.rag.chunk_size,
            )
            config.rag.chunk_overlap = RAGConfig().chunk_overlap

    # ── API server settings (#1879) ──────────────────────────────
    if "api" in data and isinstance(data["api"], dict):
        api_cfg = data["api"]
        # Rate-limit map merges into defaults so unspecified keys keep
        # the built-in value. Each value must parse as a SlowAPI-style
        # ``"<N>/<window>"`` spec — invalid values fall through to
        # ``APIConfig.__post_init__`` which raises ``ConfigError`` at
        # the merge point below.
        if "rate_limits" in api_cfg and isinstance(api_cfg["rate_limits"], dict):
            merged_limits = dict(config.api.rate_limits)
            for k, v in api_cfg["rate_limits"].items():
                if isinstance(v, str):
                    merged_limits[str(k)] = v
                else:
                    # NOTE: we intentionally log the type only, never the
                    # raw value. The ``data`` dict that ``api_cfg`` is sliced
                    # from also carries provider API keys further up the
                    # tree, and CodeQL's ``py/clear-text-logging-sensitive-data``
                    # query taints anything reachable from that root. Type
                    # alone is enough for an operator to diagnose the YAML.
                    _log.warning(
                        "api.rate_limits.%s must be a string '<N>/<window>' " "(got %s); ignoring",
                        k,
                        type(v).__name__,
                    )
            config.api.rate_limits = merged_limits
        # CIDR list accepts either a YAML list or a comma-separated string.
        if "trusted_proxy_cidrs" in api_cfg:
            raw = api_cfg["trusted_proxy_cidrs"]
            if isinstance(raw, list):
                config.api.trusted_proxy_cidrs = [str(c).strip() for c in raw if str(c).strip()]
            elif isinstance(raw, str):
                config.api.trusted_proxy_cidrs = [c.strip() for c in raw.split(",") if c.strip()]
            else:
                # Type-only log; see the rate_limits sibling block for the
                # CodeQL rationale.
                _log.warning(
                    "api.trusted_proxy_cidrs must be a list or comma-separated "
                    "string (got %s); ignoring",
                    type(raw).__name__,
                )
        # Optional Redis URL for the shared rate-limit counter
        # (#1879 Slice B). Empty string is treated as unset.
        if "redis_url" in api_cfg:
            raw_url = api_cfg["redis_url"]
            if raw_url is None or (isinstance(raw_url, str) and not raw_url.strip()):
                config.api.redis_url = None
            elif isinstance(raw_url, str):
                config.api.redis_url = raw_url.strip()
            else:
                _log.warning(
                    "api.redis_url must be a string (got %s); ignoring",
                    type(raw_url).__name__,
                )
        # Allowed CORS origins (#2059) — accepts a YAML list or a
        # comma-separated string, mirroring trusted_proxy_cidrs.
        if "cors_origins" in api_cfg:
            raw_origins = api_cfg["cors_origins"]
            if isinstance(raw_origins, list):
                config.api.cors_origins = [str(o).strip() for o in raw_origins if str(o).strip()]
            elif isinstance(raw_origins, str):
                config.api.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
            else:
                _log.warning(
                    "api.cors_origins must be a list or comma-separated string "
                    "(got %s); ignoring",
                    type(raw_origins).__name__,
                )
        # API content guardrails (#2056) — a mapping mirroring the assistant
        # ``services.assistant.guardrails`` schema. Copied verbatim; the
        # GuardrailPipeline validates its own sub-keys at construction time.
        if "guardrails" in api_cfg:
            raw_guardrails = api_cfg["guardrails"]
            if isinstance(raw_guardrails, dict):
                config.api.guardrails = dict(raw_guardrails)
            else:
                _log.warning(
                    "api.guardrails must be a mapping (got %s); ignoring",
                    type(raw_guardrails).__name__,
                )
        # Re-validate the merged APIConfig so invalid specs / CIDRs raise
        # ``ConfigError`` with the same diagnostic as a freshly-constructed
        # instance.
        APIConfig.__post_init__(config.api)

    # ── Research delegate ─────────────────────────────────────────
    if "research_delegate" in data and isinstance(data["research_delegate"], dict):
        rd = data["research_delegate"]
        if "enabled" in rd:
            config.research_delegate_enabled = bool(rd["enabled"])
        if "timeout" in rd:
            val = _safe_int(rd["timeout"], "research_delegate.timeout")
            if val is not None and val > 0:
                config.research_delegate_timeout = val
            elif val is not None:
                _log.warning("research_delegate.timeout must be > 0, using default")
        if "cap_ratio" in rd:
            fval = _safe_float(rd["cap_ratio"], "research_delegate.cap_ratio")
            if fval is not None and 0 < fval <= 1:
                config.research_delegate_cap_ratio = fval
            elif fval is not None:
                _log.warning("research_delegate.cap_ratio must be in (0, 1], using default")
        if "auto" in rd:
            config.research_delegate_auto = bool(rd["auto"])
        if "auto_threshold" in rd:
            fval = _safe_float(rd["auto_threshold"], "research_delegate.auto_threshold")
            if fval is not None and 0 < fval <= 1:
                config.research_delegate_auto_threshold = fval
            elif fval is not None:
                _log.warning("research_delegate.auto_threshold must be in (0, 1], using default")

    # ── Decision accountability (ADR-0052) ────────────────────────
    da_data = data.get("decision_accountability", {}) or {}
    if da_data:
        if "enabled" in da_data:
            config.decision_accountability_enabled = bool(da_data["enabled"])
        if "min_confidence_threshold" in da_data:
            fval = _safe_float(
                da_data["min_confidence_threshold"],
                "decision_accountability.min_confidence_threshold",
            )
            if fval is not None and 0.0 <= fval <= 10.0:
                config.decision_accountability_min_confidence = fval
            elif fval is not None:
                _log.warning(
                    "decision_accountability.min_confidence_threshold must be in [0, 10], "
                    "using default"
                )
        if "require_counter_plan" in da_data:
            config.decision_accountability_require_counter_plan = bool(
                da_data["require_counter_plan"]
            )
        if "report_uncertainty" in da_data:
            config.decision_accountability_report_uncertainty = bool(da_data["report_uncertainty"])

    # ── Task ownership classifier ─────────────────────────────────────────
    toc_data = data.get("task_ownership_classifier", {}) or {}
    if toc_data:
        if "enabled" in toc_data:
            config.task_ownership_classifier_enabled = bool(toc_data["enabled"])
        if "llm_fallback" in toc_data:
            config.task_ownership_classifier_llm_fallback = bool(toc_data["llm_fallback"])
        if "ambiguous_action" in toc_data:
            _toc_val = str(toc_data["ambiguous_action"]).lower().strip()
            if _toc_val in {"ask", "inform", "execute"}:
                config.task_ownership_ambiguous_action = _toc_val
            else:
                _log.warning(
                    "task_ownership_classifier.ambiguous_action must be ask/inform/execute;"
                    " got %r, using default 'ask'",
                    _toc_val,
                )

    # ── Pre-action confirmation gate ──────────────────────────────────────────
    pac_data = data.get("pre_action_confirmation", {}) or {}
    if pac_data:
        if "enabled" in pac_data:
            config.pre_action_confirmation_enabled = bool(pac_data["enabled"])

    # ── Audit log ─────────────────────────────────────────────────
    audit_data = data.get("audit_log", {}) or {}
    if audit_data:
        if "enabled" in audit_data:
            config.audit_log_enabled = bool(audit_data["enabled"])
        if "path" in audit_data:
            config.audit_log_path = str(audit_data["path"])

    # ── Redis session presence ─────────────────────────────────────
    if "redis_url" in data:
        config.redis_url = str(data.get("redis_url", ""))
    if "redis_session_ttl" in data:
        val = _safe_int(data["redis_session_ttl"], "redis_session_ttl")
        if val is not None and val > 0:
            config.redis_session_ttl = val

    # ── OIDC/SSO ──────────────────────────────────────────────────
    oidc_data = data.get("oidc", {}) or {}
    if oidc_data:
        config.oidc_enabled = bool(oidc_data.get("enabled", False))
        for _field, _attr in (
            ("issuer", "oidc_issuer"),
            ("audience", "oidc_audience"),
            ("jwks_uri", "oidc_jwks_uri"),
        ):
            _raw = oidc_data.get(_field)
            _val: str | None = str(_raw).strip() if _raw is not None else None
            setattr(config, _attr, _val or None)
        if "allow_insecure_oidc" in oidc_data:
            config.oidc_allow_insecure_oidc = bool(oidc_data["allow_insecure_oidc"])
        if "role_claim" in oidc_data:
            config.oidc_role_claim = str(oidc_data["role_claim"])
        _dr = str(oidc_data.get("default_role", "")).strip()
        if _dr in ("user", "admin"):
            config.oidc_default_role = _dr

    # ── Per-user quotas ───────────────────────────────────────────
    quota_data = data.get("quotas", {}) or {}
    if quota_data:
        # Tolerant numeric coercion (mirrors every other numeric config key):
        # a non-integer value warns-and-skips instead of crashing load_config.
        if "token_budget_per_day" in quota_data:
            val = _safe_int(quota_data["token_budget_per_day"], "quotas.token_budget_per_day")
            if val is not None and val > 0:
                config.quota_token_budget_per_day = val
        if "requests_per_hour" in quota_data:
            val = _safe_int(quota_data["requests_per_hour"], "quotas.requests_per_hour")
            if val is not None and val > 0:
                config.quota_requests_per_hour = val
        if "max_concurrent_sessions" in quota_data:
            val = _safe_int(quota_data["max_concurrent_sessions"], "quotas.max_concurrent_sessions")
            if val is not None and val > 0:
                config.quota_max_concurrent_sessions = val

    # ── Self-improvement loop ─────────────────────────────────────
    if "self_improve_auto_commit" in data:
        config.self_improve_auto_commit = bool(data["self_improve_auto_commit"])

    # ── Semantic tool index ───────────────────────────────────────
    if "semantic_tool_index" in data:
        config.semantic_tool_index = bool(data["semantic_tool_index"])

    # ── Legacy top-level provider/model → derive active_model_alias ──
    if config.active_model_alias is None and (_legacy_provider or _legacy_model):
        for alias, mc in config.models.items():
            if mc.provider == _legacy_provider and mc.model == _legacy_model:
                config.active_model_alias = alias
                break
        if config.active_model_alias is None and _legacy_model:
            _synth_alias = _legacy_model
            if _synth_alias not in config.models and _legacy_provider:
                config.models[_synth_alias] = ModelConfig(
                    provider=_legacy_provider, model=_legacy_model
                )
            config.active_model_alias = _synth_alias


def _parse_providers_section(config: Config, providers_data: dict[str, Any]) -> None:
    """Parse the providers section into ProviderConfig objects."""
    for name, provider_data in providers_data.items():
        if not isinstance(provider_data, dict):
            _log.warning("Invalid provider config for '%s': expected dict", name)
            continue

        provider_type = provider_data.get("type")
        if not provider_type:
            _log.warning("Provider '%s' missing required 'type' field", name)
            continue

        provider_type = str(provider_type).lower()

        from cogtrix_core.providers import PROVIDER_TYPES

        if provider_type not in PROVIDER_TYPES:
            _log.warning(
                "Provider '%s' has unknown type '%s' (supported: %s)",
                name,
                provider_type,
                ", ".join(sorted(PROVIDER_TYPES)),
            )
            continue

        try:
            config.providers[name] = ProviderConfig(
                name=name,
                type=provider_type,
                base_url=provider_data.get("base_url"),
                api_key=provider_data.get("api_key"),
                tool_instructions=provider_data.get("tool_instructions"),
            )
        except (ConfigError, ValueError, TypeError) as exc:
            _log.warning("Skipping provider '%s': %s", name, exc)
            continue

        # Auto-migrate model settings from provider section → models registry
        _prov_model = provider_data.get("model")
        if _prov_model:
            # Prefer model_name as alias; fall back to provider/model_name on collision
            _alias = str(_prov_model)
            if _alias in config.models:
                _alias = f"{name}/{_prov_model}"
            if _alias not in config.models and name not in config.models:
                _prov_temp = provider_data.get("temperature")
                _prov_ctx = (
                    provider_data.get("context_window")
                    or provider_data.get("context_length")
                    or provider_data.get("num_ctx")
                )
                _prov_max = provider_data.get("max_tokens")
                try:
                    config.models[_alias] = ModelConfig(
                        provider=name,
                        model=_prov_model,
                        temperature=(
                            _safe_float(_prov_temp, f"providers.{name}.temperature")
                            if _prov_temp is not None
                            else None
                        ),
                        context_window=(
                            _safe_int(_prov_ctx, f"providers.{name}.context_window")
                            if _prov_ctx is not None
                            else None
                        ),
                        max_tokens=(
                            _safe_int(_prov_max, f"providers.{name}.max_tokens")
                            if _prov_max is not None
                            else None
                        ),
                    )
                except (ConfigError, ValueError, TypeError) as exc:
                    _log.warning("Could not auto-migrate model from provider '%s': %s", name, exc)


def _parse_models_section(config: Config, models_data: dict[str, Any]) -> None:
    """Parse the models section into ModelConfig objects."""
    # Handle models.default — selects the active model alias
    if "default" in models_data:
        default_val = models_data.pop("default")
        if isinstance(default_val, str) and default_val:
            config.active_model_alias = default_val
        else:
            _log.warning("models.default must be a non-empty string, ignoring")

    for name, model_data in models_data.items():
        if isinstance(model_data, dict):
            provider = model_data.get("provider")
            model = model_data.get("model")
            if not provider or not model:
                _log.warning("Model '%s' missing required 'provider' or 'model' field", name)
                continue
            raw_ctx = (
                model_data.get("context_window")
                or model_data.get("context_length")
                or model_data.get("num_ctx")
            )
            raw_temperature = model_data.get("temperature")
            raw_max_tokens = model_data.get("max_tokens")
            raw_timeout = model_data.get("timeout")
            raw_model_kwargs = model_data.get("model_kwargs")
            if raw_model_kwargs is not None and not isinstance(raw_model_kwargs, dict):
                _log.warning(
                    "models.%s.model_kwargs must be a mapping, ignoring (got %s)",
                    name,
                    type(raw_model_kwargs).__name__,
                )
                raw_model_kwargs = None
            raw_supports_vision = model_data.get("supports_vision")
            supports_vision: bool | None = None
            if raw_supports_vision is not None:
                supports_vision = bool(raw_supports_vision)
            try:
                config.models[name] = ModelConfig(
                    provider=provider,
                    model=model,
                    context_window=(
                        _safe_int(raw_ctx, f"models.{name}.context_window")
                        if raw_ctx is not None
                        else None
                    ),
                    temperature=(
                        _safe_float(raw_temperature, f"models.{name}.temperature")
                        if raw_temperature is not None
                        else None
                    ),
                    max_tokens=(
                        _safe_int(raw_max_tokens, f"models.{name}.max_tokens")
                        if raw_max_tokens is not None
                        else None
                    ),
                    timeout=(
                        _safe_int(raw_timeout, f"models.{name}.timeout") or 180
                        if raw_timeout is not None
                        else 180
                    ),
                    model_kwargs=dict(raw_model_kwargs) if raw_model_kwargs else {},
                    supports_vision=supports_vision,
                )
            except (ConfigError, ValueError, TypeError) as exc:
                _log.warning("Invalid model config '%s': %s", name, exc)
        elif isinstance(model_data, str):
            # String format: "provider/model" or just "model"
            if "/" in model_data:
                parts = model_data.split("/", 1)
                config.models[name] = ModelConfig(provider=parts[0], model=parts[1])
            else:
                # Just model name — use first provider as fallback
                _fallback_prov = next(iter(config.providers), "ollama")
                config.models[name] = ModelConfig(provider=_fallback_prov, model=model_data)


def _set_service(config: Config, name: str, key: str, value: str) -> None:
    """Set a single key in a service's config dict, creating it if needed."""
    if name not in config.services:
        config.services[name] = {}
    config.services[name][key] = value


def _set_provider_key(config: Config, name: str, api_key: str) -> None:
    """Set the API key for a named provider, creating it if needed."""
    if name in config.providers:
        from cogtrix_core.providers.defaults import OPENAI_PRESETS

        preset = OPENAI_PRESETS.get(name)
        if preset:
            existing_url = config.providers[name].base_url
            canonical_url = preset["base_url"]
            if existing_url and existing_url.rstrip("/") != canonical_url.rstrip("/"):
                import sys

                _log.warning(
                    "SECURITY: %s_API_KEY applied to provider '%s' whose base_url "
                    "(%r) differs from the canonical preset URL (%r). "
                    "If this is unintentional, check providers.%s.base_url in your "
                    "config file.",
                    name.upper(),
                    name,
                    existing_url,
                    canonical_url,
                    name,
                )
                print(
                    f"  [!] WARNING: {name.upper()}_API_KEY applied to provider "
                    f"'{name}' with non-canonical base_url ({existing_url!r}). "
                    f"Expected: {canonical_url!r}. Check your config file.",
                    file=sys.stderr,
                    flush=True,
                )
        config.providers[name].api_key = api_key
    else:
        from cogtrix_core.providers.defaults import OPENAI_PRESETS

        # Check if it's an OpenAI-compatible preset (e.g. xai)
        preset = OPENAI_PRESETS.get(name)
        if preset:
            config.providers[name] = ProviderConfig(
                name=name,
                type="openai",
                api_key=api_key,
                base_url=preset["base_url"],
            )
            if name not in config.models:
                config.models[name] = ModelConfig(provider=name, model=preset["model"])
        else:
            from cogtrix_core.providers import PROVIDER_TYPES

            ptype = name if name in PROVIDER_TYPES else "openai"
            if ptype != name:
                _log.warning("Unknown provider type '%s', defaulting to openai", name)
            config.providers[name] = ProviderConfig(
                name=name,
                type=ptype,
                api_key=api_key,
            )


_OLLAMA_DEFAULT_PORT = "11434"


def _parse_ollama_address(value: str) -> str:
    """Parse an Ollama address into a full base URL.

    Accepts:
        ``host``          → ``http://host:11434``
        ``host:port``     → ``http://host:port``
        ``http://...``    → returned as-is

    Returns:
        A complete ``http://`` base URL suitable for the Ollama client.
    """
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return value
    # Already bracketed without port: [::1] or [2001:db8::1]
    if value.startswith("[") and value.endswith("]"):
        return f"http://{value}:{_OLLAMA_DEFAULT_PORT}"
    if ":" in value:
        # Exactly one colon → host:port; multiple colons → IPv6 address.
        # Bracketed IPv6 like [::1]:8080 is fine — rsplit on the last ":"
        # yields "[::1]" and "8080".
        if value.count(":") == 1 or (value.startswith("[") and "]:" in value):
            host, port = value.rsplit(":", 1)
            if port.isdigit():
                return f"http://{host}:{port}"
            # Bracketed host with non-numeric port — ignore the port part
            if host.startswith("[") and host.endswith("]"):
                _log.warning(
                    "Non-numeric port '%s' in Ollama address '%s', using default %s",
                    port,
                    value,
                    _OLLAMA_DEFAULT_PORT,
                )
                return f"http://{host}:{_OLLAMA_DEFAULT_PORT}"
            # Plain host:badport — treat hostname only, ignore invalid port
            _log.warning(
                "Invalid port %r in Ollama address %r, using default port",
                port,
                value,
            )
            return f"http://{host}:{_OLLAMA_DEFAULT_PORT}"
        # Multiple colons without brackets — bare IPv6; wrap in brackets
        return f"http://[{value}]:{_OLLAMA_DEFAULT_PORT}"
    return f"http://{value}:{_OLLAMA_DEFAULT_PORT}"


#: Matches ``COGTRIX_PROVIDER_<NAME>_API_KEY`` for the generic per-provider key
#: override (#2222). The captured ``name`` is lowercased to the provider name.
_PROVIDER_KEY_ENV_RE = re.compile(r"^COGTRIX_PROVIDER_(?P<name>.+)_API_KEY$")

#: Sensitive env vars dropped from ``os.environ`` immediately after they've been
#: copied into ``Config`` (#2223), so a shell/code-exec tool subprocess can't
#: inherit them and they don't linger in ``/proc/<pid>/environ``. ONLY secrets
#: that are consumed via ``Config`` are listed: every LLM provider key (read
#: through ``config.providers[].api_key``) and the search-tool service keys
#: (injected into each tool's config by ``configure_*_tool``). The generic
#: ``COGTRIX_PROVIDER_<NAME>_API_KEY`` names are matched dynamically via
#: :data:`_PROVIDER_KEY_ENV_RE` in addition to this set.
#:
#: Phase 2 (#2223): ``OPENWEATHER_API_KEY``, ``COGTRIX_WHATSAPP_API_KEY``,
#: ``COGTRIX_TELEGRAM_TOKEN``, and ``COGTRIX_SLACK_BOT_TOKEN`` are now also
#: included because their tools are config-authoritative (each declares
#: ``TOOL_SETUP`` which rebuilds the tool singleton from the injected ``Config``
#: before the env var is unset).
#:
#: Phase 3 (#2102): ``COGTRIX_JWT_SECRET``. It is consumed OUTSIDE the config
#: loader (``app.py`` / ``auth.py``) via :func:`secret_from_env_or_file`, which
#: reads the survives-unset process cache (#2103). :func:`_apply_env_vars` seeds
#: that cache *before* the unset so those consumers still resolve it once the env
#: var is gone — without #2103's cache routing this unset would be unsafe (the
#: documented read-once dependency of #2101/#2102).
#:
#: ``COGTRIX_DB_URL`` is deliberately NOT unset here: the engine layer
#: (``cogtrix_core/api/db/engine.py``) resolves it through its own ``data_dir``-aware,
#: lazily-reimported path that reads ``os.environ`` directly, so unsetting it from
#: the config loader rebinds the engine and breaks ``data_dir`` resolution. The
#: default SQLite URL carries no secret; unsetting a password-bearing DB URL is
#: left to dedicated engine-layer work.
_SECRETS_UNSET_AFTER_READ = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "BRAVE_API_KEY",
        "SERPAPI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENWEATHER_API_KEY",
        "COGTRIX_WHATSAPP_API_KEY",
        "COGTRIX_TELEGRAM_TOKEN",
        "COGTRIX_SLACK_BOT_TOKEN",
        "COGTRIX_JWT_SECRET",
    }
)

#: Secret names consumed outside :func:`_apply_env_vars` (at API startup) via
#: :func:`secret_from_env_or_file`. They are seeded into the process cache before
#: the post-read unset so those consumers survive it (#2101/#2102).
_SECRETS_SEED_BEFORE_UNSET = ("COGTRIX_JWT_SECRET",)


def _keep_env_secrets() -> bool:
    """True when ``COGTRIX_KEEP_ENV_SECRETS`` opts out of the post-read env unset.

    Escape hatch (#2102) for debugging: leaves the sensitive env vars in
    ``os.environ`` (and therefore in child-process environments) so an operator
    can inspect them. Off by default — the hardening (unset-after-read) is the
    default posture.
    """
    return os.environ.get("COGTRIX_KEEP_ENV_SECRETS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _unset_sensitive_env() -> None:
    """Drop config-authoritative secrets from ``os.environ`` (#2223).

    Called once, after :func:`_apply_env_vars` has copied the values into the
    ``Config``. Removes the static :data:`_SECRETS_UNSET_AFTER_READ` names plus
    any generic ``COGTRIX_PROVIDER_<NAME>_API_KEY`` keys.
    """
    for name in list(os.environ):
        if name in _SECRETS_UNSET_AFTER_READ or _PROVIDER_KEY_ENV_RE.match(name):
            os.environ.pop(name, None)


#: Process-level cache of secret env values, keyed by env-var name (#2233).
#: Populated the first time a secret is read from ``os.environ``; reused by later
#: ``load_config()`` calls after :func:`_unset_sensitive_env` has popped the var.
#: This is a plain in-process dict — NOT inherited by subprocesses — so the
#: #2223 goal (secrets out of the *inheritable* environment) still holds, while
#: the invariant "once a key is read from the env it stays available to Config
#: for the process lifetime, regardless of how many times config is re-resolved"
#: is preserved. The API re-resolves config (per session / turn / reload), which
#: is exactly when an env-only provider key would otherwise come back empty.
_SECRET_ENV_CACHE: dict[str, str] = {}


def _read_secret_from_file(name: str) -> str | None:
    """Resolve the ``<name>_FILE`` secret-file convention (#2103).

    Container/orchestrator secret delivery (Docker/Swarm secrets at
    ``/run/secrets/``, Kubernetes secret volumes, Vault-agent) mounts a secret
    as a *file*, which keeps it out of the process environment entirely
    (``docker inspect`` / ``/proc/<pid>/environ`` / child inheritance). If
    ``<name>_FILE`` is set, read the secret from that path, trimming a single
    trailing newline (secret files commonly end with one).

    Returns ``None`` when ``<name>_FILE`` is not set. Raises :class:`ConfigError`
    when it IS set but the target is missing, unreadable, or empty — a
    misconfigured secret mount must fail loudly, never silently yield an empty
    key (acceptance criterion of #2103).
    """
    file_var = f"{name}_FILE"
    path_str = os.environ.get(file_var)
    if not path_str:
        return None
    try:
        raw = Path(path_str).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"{file_var}={path_str!r} could not be read: {exc}. Ensure the secret "
            f"file exists and is readable by the process uid."
        ) from exc
    # Trim exactly one trailing newline (``\n`` or ``\r\n``); preserve any other
    # content verbatim so secrets that legitimately contain whitespace survive.
    if raw.endswith("\r\n"):
        secret = raw[:-2]
    elif raw.endswith("\n"):
        secret = raw[:-1]
    else:
        secret = raw
    if not secret:
        raise ConfigError(
            f"{file_var}={path_str!r} is empty. A secret file must contain the "
            f"secret value (a single trailing newline is trimmed)."
        )
    return secret


def _secret_env(name: str) -> str | None:
    """Read a secret from the environment, the ``_FILE`` convention, or the cache.

    Resolution order (honours the #2103 precedence explicit-env > ``_FILE`` >
    config-file > default — the config file is applied before this runs, so a
    ``None`` return leaves the config-file value in place):

      1. Live ``<name>`` env value (cached for the process lifetime).
      2. Process cache from an earlier load — the env var may since have been
         unset by #2223, and a value already resolved from ``_FILE`` lives here
         too, so the cache wins over re-reading the file (keeps explicit-env's
         precedence stable across re-resolutions when both ``<name>`` and
         ``<name>_FILE`` are set).
      3. ``<name>_FILE`` secret-file convention (#2103), then cache it.

    See :data:`_SECRET_ENV_CACHE`.
    """
    val = os.environ.get(name)
    if val:
        _SECRET_ENV_CACHE[name] = val
        return val
    cached = _SECRET_ENV_CACHE.get(name)
    if cached:
        return cached
    file_val = _read_secret_from_file(name)
    if file_val:
        _SECRET_ENV_CACHE[name] = file_val
        return file_val
    return None


def secret_from_env_or_file(name: str) -> str | None:
    """Public entry point for secret consumers OUTSIDE the config loader.

    The JWT signing secret (``cogtrix_core/api/app.py`` / ``cogtrix_core/api/auth.py``) and the
    database URL (``cogtrix_core/api/db/engine.py``) are read directly from the
    environment rather than through :func:`_apply_env_vars`. This wrapper gives
    them the same ``<name>`` → ``<name>_FILE`` → process-cache resolution (#2103)
    so e.g. ``COGTRIX_JWT_SECRET_FILE`` / ``COGTRIX_DB_URL_FILE`` work.
    """
    return _secret_env(name)


def _reset_secret_env_cache() -> None:
    """Clear the secret-env cache. For test isolation only — never call in app
    code (secrets are meant to persist for the process lifetime)."""
    _SECRET_ENV_CACHE.clear()


def _apply_env_vars(config: Config, *, unset_secrets: bool = True) -> None:
    """Apply settings from environment variables.

    When ``unset_secrets`` (the default), config-authoritative secret env vars
    are removed from ``os.environ`` after being read (#2223). Pass ``False`` in
    tests that need to assert environment contents.
    """
    # General settings
    if env_val := os.getenv("COGTRIX_MODEL"):
        config.active_model_alias = env_val
    if env_val := os.getenv("COGTRIX_SESSION"):
        config.session = env_val
    if env_val := os.getenv("COGTRIX_DATA_DIR"):
        config.data_dir = env_val

    # LLM provider API keys — via named providers. Read through _secret_env so an
    # env-only key survives later re-resolutions after #2223 unset it (#2233).
    if env_val := _secret_env("OPENAI_API_KEY"):
        _set_provider_key(config, "openai", env_val)
    if env_val := _secret_env("ANTHROPIC_API_KEY"):
        _set_provider_key(config, "anthropic", env_val)
    if env_val := _secret_env("GEMINI_API_KEY"):
        _set_provider_key(config, "google", env_val)
    if env_val := _secret_env("GROQ_API_KEY"):
        _set_provider_key(config, "groq", env_val)
    if env_val := _secret_env("XAI_API_KEY"):
        _set_provider_key(config, "xai", env_val)
    if env_val := _secret_env("DEEPSEEK_API_KEY"):
        _set_provider_key(config, "deepseek", env_val)

    # Generic per-provider key override (#2222). Custom / self-hosted providers
    # (e.g. a local vLLM "spark" endpoint) have no well-known *_API_KEY name, so
    # their key could only live inline in the config file. This lets ANY
    # provider's key come from the environment instead:
    #     COGTRIX_PROVIDER_<NAME>_API_KEY=<key>  ->  providers.<name>.api_key
    # <NAME> maps to a lowercased provider name (hyphenated names aren't
    # addressable this way — use a simple/underscore-free provider name). As
    # with the well-known keys above, the env value overrides the config file.
    # Consider both live env names and cached names (#2233): on a re-resolution
    # the generic key has been popped from os.environ, so discover it from the
    # cache too, then read the value through _secret_env.
    generic_names = {n for n in os.environ if _PROVIDER_KEY_ENV_RE.match(n)}
    generic_names |= {n for n in _SECRET_ENV_CACHE if _PROVIDER_KEY_ENV_RE.match(n)}
    # _FILE convention (#2103): a custom provider key may be delivered only as
    # COGTRIX_PROVIDER_<NAME>_API_KEY_FILE (no plain env var). Discover the base
    # name from the *_FILE entry so _secret_env() can resolve it from the file.
    for n in os.environ:
        if n.endswith("_FILE"):
            base = n[: -len("_FILE")]
            if _PROVIDER_KEY_ENV_RE.match(base):
                generic_names.add(base)
    for env_name in generic_names:
        m = _PROVIDER_KEY_ENV_RE.match(env_name)
        env_val = _secret_env(env_name)
        if m and env_val:
            _set_provider_key(config, m.group("name").lower(), env_val)

    # Ollama settings — update or create ollama provider entry
    ollama_url: str | None = None
    if env_val := os.getenv("COGTRIX_OLLAMA"):
        ollama_url = _parse_ollama_address(env_val)
    elif env_val := os.getenv("OLLAMA_BASE_URL"):
        ollama_url = env_val

    if ollama_url:
        if "ollama" in config.providers:
            config.providers["ollama"].base_url = ollama_url
        else:
            config.providers["ollama"] = ProviderConfig(
                name="ollama",
                type="ollama",
                base_url=ollama_url,
            )

    # Service API keys → services dict. Secret keys go through _secret_env so they
    # survive re-resolution after the #2223 unset (#2233); GOOGLE_CSE_ID is not a
    # secret (not unset), so it stays a plain env read.
    if env_val := _secret_env("OPENWEATHER_API_KEY"):
        _set_service(config, "openweather", "api_key", env_val)
    if env_val := _secret_env("TAVILY_API_KEY"):
        _set_service(config, "tavily", "api_key", env_val)
    if env_val := _secret_env("EXA_API_KEY"):
        _set_service(config, "exa", "api_key", env_val)
    if env_val := _secret_env("BRAVE_API_KEY"):
        _set_service(config, "brave", "api_key", env_val)
    if env_val := _secret_env("SERPAPI_API_KEY"):
        _set_service(config, "serpapi", "api_key", env_val)
    if env_val := _secret_env("GOOGLE_API_KEY"):
        _set_service(config, "google", "api_key", env_val)
    if env_val := os.getenv("GOOGLE_CSE_ID"):
        _set_service(config, "google", "cse_id", env_val)

    # Memory settings
    if env_val := os.getenv("COGTRIX_MEMORY_MODE"):
        config.memory_mode = env_val

    # Embedding overrides
    if env_val := os.getenv("COGTRIX_EMBEDDING_PROVIDER"):
        config.embedding_provider_override = env_val
    if env_val := os.getenv("OLLAMA_EMBEDDING_MODEL"):
        config.embedding_model_override = env_val

    # WhatsApp env vars. The api_key is a secret (unset by #2223) → read via
    # _secret_env so it survives re-resolution (#2233); URL/session are not.
    wa_url = os.environ.get("COGTRIX_WHATSAPP_URL")
    wa_key = _secret_env("COGTRIX_WHATSAPP_API_KEY")
    wa_session = os.environ.get("COGTRIX_WHATSAPP_SESSION")
    if wa_url or wa_key or wa_session:
        wa = config.services.setdefault("whatsapp", {})
        if wa_url:
            wa["waha_url"] = wa_url
        if wa_key:
            wa["api_key"] = wa_key
        if wa_session:
            wa["session"] = wa_session

    # Telegram env vars (bot_token is a secret → _secret_env, #2233)
    tg_token = _secret_env("COGTRIX_TELEGRAM_TOKEN")
    if tg_token:
        tg = config.services.setdefault("telegram", {})
        tg["bot_token"] = tg_token

    # Slack env vars (#2223 phase 2; secret → _secret_env, #2233)
    if env_val := _secret_env("COGTRIX_SLACK_BOT_TOKEN"):
        _set_service(config, "slack", "bot_token", env_val)

    # Allowed CORS origins — comma-separated; overrides api.cors_origins (#2059).
    if env_val := os.getenv("COGTRIX_CORS_ORIGINS"):
        origins = [o.strip() for o in env_val.split(",") if o.strip()]
        if origins:
            config.api.cors_origins = origins

    # Allowed write paths — comma-separated to match file_ops.py runtime parser.
    if env_val := os.getenv("COGTRIX_ALLOWED_WRITE_PATHS"):
        config.allowed_write_paths = [p.strip() for p in env_val.split(",") if p.strip()]

    # Allowed read paths — comma-separated.
    if env_val := os.getenv("COGTRIX_ALLOWED_READ_PATHS"):
        config.allowed_read_paths = [p.strip() for p in env_val.split(",") if p.strip()]

    # Plugin tool directories — comma-separated.
    if env_val := os.getenv("COGTRIX_TOOL_DIRS"):
        config.tool_dirs = [p.strip() for p in env_val.split(",") if p.strip()]

    # Data science modules — enable numpy/pandas/scipy in python_exec
    if env_val := os.getenv("COGTRIX_ENABLE_DATASCIENCE_MODULES"):
        config.enable_datascience_modules = env_val.lower() in ("true", "1", "yes")

    # Organization scoping for admin endpoints
    if env_val := os.getenv("COGTRIX_ENABLE_ORG_SCOPING"):
        config.enable_org_scoping = env_val.lower() in ("true", "1", "yes")

    # Read-once hardening: now that every secret above has been copied into
    # `config`, drop the config-authoritative ones from the environment (#2223).
    # COGTRIX_KEEP_ENV_SECRETS opts out for debugging (#2102).
    if unset_secrets and not _keep_env_secrets():
        # Seed the JWT secret and DB URL into the process cache before the unset
        # so the consumers that read them outside this loader (app.py / auth.py /
        # db engine, via secret_from_env_or_file → _secret_env) still resolve them
        # once the env var is gone (#2101/#2102).
        for _name in _SECRETS_SEED_BEFORE_UNSET:
            _secret_env(_name)
        _unset_sensitive_env()


def _apply_cli_args(config: Config, args) -> None:
    """Apply settings from command line arguments."""
    if hasattr(args, "provider") and args.provider:
        _log.warning("--provider is deprecated; use --model <alias> instead")
    if hasattr(args, "model") and args.model:
        config.active_model_alias = args.model
    if hasattr(args, "session") and args.session:
        config.session = args.session
    if hasattr(args, "data_dir") and args.data_dir:
        config.data_dir = args.data_dir
    if hasattr(args, "memory_mode") and args.memory_mode:
        config.memory_mode = args.memory_mode

    # Debug and logging settings — --verbosity takes priority over --debug
    raw_verbosity = getattr(args, "verbosity", None)
    debug_flag = getattr(args, "debug", False)
    if raw_verbosity is not None:
        config.verbosity = max(0, min(3, int(raw_verbosity)))
        if config.verbosity >= 1:
            config.debug = True
            if config.log_file is None:
                config.log_file = ""
        if config.verbosity >= 2:
            config.verbose = True
    elif debug_flag:
        # --debug implies full debug+verbose output (verbosity 2)
        config.verbosity = 2
        config.debug = True
        config.verbose = True
        if config.log_file is None:
            config.log_file = ""

    # --verbose without --debug: verbosity 1 (shows LLM interactions)
    if hasattr(args, "verbose") and args.verbose:
        config.verbose = True
        if config.verbosity == 0:
            config.verbosity = 1
            config.debug = True
            if config.log_file is None:
                config.log_file = ""

    # --log can be used without --debug
    if hasattr(args, "log") and args.log is not None:
        config.log_file = args.log

    # Allowed write paths
    if hasattr(args, "allow_write_path") and args.allow_write_path:
        config.allowed_write_paths = list(args.allow_write_path)

    # Allowed read paths
    if hasattr(args, "allow_read_path") and args.allow_read_path:
        config.allowed_read_paths = list(args.allow_read_path)
