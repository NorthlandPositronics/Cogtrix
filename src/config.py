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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    if isinstance(value, bool):
        _log.warning(
            "Invalid float for %s: %r (bool is not a valid float), skipping", field_name, value
        )
        return default
    try:
        return float(value)
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
            from src.providers.defaults import PROVIDER_TYPES

            if self.type not in PROVIDER_TYPES:
                raise ConfigError(
                    f"providers.{self.name}.type '{self.type}' is not a recognized provider type. "
                    f"Supported: {', '.join(sorted(PROVIDER_TYPES))}"
                )

    def get_base_url(self) -> str | None:
        """Get base URL with defaults for known types."""
        if self.base_url:
            return self.base_url
        from src.providers import get_default_base_url

        return get_default_base_url(self.type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
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


@dataclass
class RAGConfig:
    """Configuration for RAG document ingestion."""

    docs_dir: str = "docs"
    vectordb_dir: str = "vectordb"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    model: str | None = None  # references a key in Config.models for embedding
    score_threshold: float = 0.0  # minimum similarity score for RAG retrieval (M4.3)

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

    # Named flag profiles: profile_name -> {config_key: value}
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Parallel tool execution — run independent tool calls concurrently
    parallel_tool_execution: bool = True

    # File operations — extra directories allowed for write operations
    allowed_write_paths: list[str] = field(default_factory=list)

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
            if len(val) <= 8:
                return "***"
            return val[:4] + "..." + val[-4:]

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

        log.debug("\n".join(lines))

    def resolve_data_path(self, subpath: str) -> Path:
        """Resolve a data subpath against ``data_dir``.

        Absolute paths are returned as-is.  Relative paths starting with
        the legacy ``data/`` prefix (from configs written before the
        ``data_dir`` option existed) are normalized to avoid double-nesting.
        Path traversal sequences (``..``) that escape ``data_dir`` are rejected.
        """
        subpath = str(subpath)
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
        # Traversal check: use resolved absolute paths to catch ``..`` escapes.
        base_resolved = data_dir.resolve()
        result_resolved = result.resolve()
        if not result_resolved.is_relative_to(base_resolved):
            raise ConfigError(f"Path traversal detected in data path: {subpath!r}")
        return result_resolved

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
        """
        model_name = self.rag.model
        if model_name and model_name in self.models:
            mc = self.models[model_name]
            pc = self.providers.get(mc.provider)
            if pc:
                return pc.type, mc.model, pc.get_base_url(), pc.api_key
            import logging

            logging.getLogger("cogtrix").warning(
                "rag.model '%s' references provider '%s' which is not configured; "
                "falling back to active provider",
                model_name,
                mc.provider,
            )
        pc = self.get_active_provider()
        return pc.type, None, pc.get_base_url(), pc.api_key

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


def load_config(cli_args=None) -> Config:
    """
    Load configuration with priority:
    CLI args > Environment variables > Config file > Defaults

    If ``cli_args`` has a ``config_file`` attribute, that path is used
    directly instead of the normal search.

    Args:
        cli_args: Parsed command line arguments (argparse namespace)

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
    _apply_env_vars(config)

    # 4. Override with CLI arguments (highest priority)
    if cli_args:
        _apply_cli_args(config, cli_args)

    # 5. Resolve model alias against the models registry
    _resolve_model(config)

    # 6. If active_model_alias is still None, generate a default from first provider
    if config.active_model_alias is None and config.providers:
        first_prov = next(iter(config.providers.values()))
        from src.providers import get_default_model

        default_model = get_default_model(first_prov.type)
        _synth = f"{first_prov.name}/{default_model}"
        if _synth not in config.models:
            config.models[_synth] = ModelConfig(provider=first_prov.name, model=default_model)
        config.active_model_alias = _synth

    return config


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
        _banner_val = str(data["banner"]).lower().strip()
        if _banner_val in ("full", "compact", "off", "none", "false", "0"):
            config.banner = "off" if _banner_val in ("off", "none", "false", "0") else _banner_val
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
        if config.rag.chunk_overlap >= config.rag.chunk_size:
            _log.warning(
                "rag.chunk_overlap (%d) must be less than rag.chunk_size (%d); "
                "resetting chunk_overlap to default",
                config.rag.chunk_overlap,
                config.rag.chunk_size,
            )
            config.rag.chunk_overlap = RAGConfig().chunk_overlap

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
        if "role_claim" in oidc_data:
            config.oidc_role_claim = str(oidc_data["role_claim"])
        _dr = str(oidc_data.get("default_role", "")).strip()
        if _dr in ("user", "admin"):
            config.oidc_default_role = _dr

    # ── Per-user quotas ───────────────────────────────────────────
    quota_data = data.get("quotas", {}) or {}
    if quota_data:
        v = quota_data.get("token_budget_per_day")
        if v is not None and int(v) > 0:
            config.quota_token_budget_per_day = int(v)
        v = quota_data.get("requests_per_hour")
        if v is not None and int(v) > 0:
            config.quota_requests_per_hour = int(v)
        v = quota_data.get("max_concurrent_sessions")
        if v is not None and int(v) > 0:
            config.quota_max_concurrent_sessions = int(v)

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

        from src.providers import PROVIDER_TYPES

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
        config.providers[name].api_key = api_key
    else:
        from src.providers.defaults import OPENAI_PRESETS

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
            from src.providers import PROVIDER_TYPES

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


def _apply_env_vars(config: Config) -> None:
    """Apply settings from environment variables."""
    # General settings
    if env_val := os.getenv("COGTRIX_MODEL"):
        config.active_model_alias = env_val
    if env_val := os.getenv("COGTRIX_SESSION"):
        config.session = env_val
    if env_val := os.getenv("COGTRIX_DATA_DIR"):
        config.data_dir = env_val

    # LLM provider API keys — via named providers
    if env_val := os.getenv("OPENAI_API_KEY"):
        _set_provider_key(config, "openai", env_val)
    if env_val := os.getenv("ANTHROPIC_API_KEY"):
        _set_provider_key(config, "anthropic", env_val)
    if env_val := os.getenv("GEMINI_API_KEY"):
        _set_provider_key(config, "google", env_val)
    if env_val := os.getenv("XAI_API_KEY"):
        _set_provider_key(config, "xai", env_val)

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

    # Service API keys → services dict
    if env_val := os.getenv("OPENWEATHER_API_KEY"):
        _set_service(config, "openweather", "api_key", env_val)
    if env_val := os.getenv("TAVILY_API_KEY"):
        _set_service(config, "tavily", "api_key", env_val)
    if env_val := os.getenv("EXA_API_KEY"):
        _set_service(config, "exa", "api_key", env_val)
    if env_val := os.getenv("BRAVE_API_KEY"):
        _set_service(config, "brave", "api_key", env_val)
    if env_val := os.getenv("SERPAPI_API_KEY"):
        _set_service(config, "serpapi", "api_key", env_val)
    if env_val := os.getenv("GOOGLE_API_KEY"):
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

    # WhatsApp env vars
    wa_url = os.environ.get("COGTRIX_WHATSAPP_URL")
    wa_key = os.environ.get("COGTRIX_WHATSAPP_API_KEY")
    wa_session = os.environ.get("COGTRIX_WHATSAPP_SESSION")
    if wa_url or wa_key or wa_session:
        wa = config.services.setdefault("whatsapp", {})
        if wa_url:
            wa["waha_url"] = wa_url
        if wa_key:
            wa["api_key"] = wa_key
        if wa_session:
            wa["session"] = wa_session

    # Telegram env vars
    tg_token = os.environ.get("COGTRIX_TELEGRAM_TOKEN")
    if tg_token:
        tg = config.services.setdefault("telegram", {})
        tg["bot_token"] = tg_token

    # Allowed write paths
    if env_val := os.getenv("COGTRIX_ALLOWED_WRITE_PATHS"):
        config.allowed_write_paths = [p.strip() for p in env_val.split(":") if p.strip()]

    # Plugin tool directories
    if env_val := os.getenv("COGTRIX_TOOL_DIRS"):
        config.tool_dirs = [p.strip() for p in env_val.split(":") if p.strip()]


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
