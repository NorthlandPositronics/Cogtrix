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
        num_ctx: 32768
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
    """Configuration for a single LLM provider."""

    name: str
    type: str  # "openai", "ollama", "anthropic", or "google"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    # Provider-level defaults; overridden by model-level settings from ModelConfig
    temperature: float | None = None
    num_ctx: int | None = None  # Context window size (tokens) for any provider
    max_tokens: int | None = None  # Max output tokens per LLM call
    tool_instructions: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(
                f"providers.{self.name}.temperature must be between 0.0 and 2.0, "
                f"got {self.temperature}"
            )
        if self.num_ctx is not None and self.num_ctx < 256:
            raise ConfigError(f"providers.{self.name}.num_ctx must be >= 256, got {self.num_ctx}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigError(
                f"providers.{self.name}.max_tokens must be > 0, got {self.max_tokens}"
            )
        if self.type:
            from src.providers import PROVIDER_TYPES

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

    def get_model(self) -> str:
        """Get model with defaults for known types."""
        if self.model:
            return self.model
        from src.providers import get_default_model

        return get_default_model(self.type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": "***" if self.api_key else None,  # Don't expose key
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "tool_instructions": self.tool_instructions,
        }


@dataclass
class ModelConfig:
    """Configuration for a named model in the models registry."""

    provider: str  # references a key in Config.providers
    model: str  # actual model name at the provider
    num_ctx: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None  # Max output tokens per LLM call

    def __post_init__(self) -> None:
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.num_ctx is not None and self.num_ctx < 256:
            raise ConfigError(f"Context window (num_ctx) must be >= 256, got {self.num_ctx}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass
class RAGConfig:
    """Configuration for RAG document ingestion."""

    docs_dir: str = "docs"
    vectordb_dir: str = "vectordb"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    model: str | None = None  # references a key in Config.models for embedding

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"rag.chunk_overlap ({self.chunk_overlap}) must be less than "
                f"rag.chunk_size ({self.chunk_size})"
            )


@dataclass
class Config:
    """Application configuration with defaults."""

    # General settings
    provider: str = "ollama"
    model: str | None = None
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

    # Parallel tool execution — run independent tool calls concurrently
    parallel_tool_execution: bool = True

    # File operations — extra directories allowed for write operations
    allowed_write_paths: list[str] = field(default_factory=list)

    # Context compression — summarize old ToolMessages during agent loop
    context_compression: bool = True
    context_compression_min_age: int = 6
    context_compression_min_chars: int = 2000
    context_compression_model: str | None = None  # model name or "provider/model"

    # MCP server configurations
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # RAG settings
    rag: RAGConfig = field(default_factory=RAGConfig)

    # Logging and debug settings
    debug: bool = False
    verbose: bool = False  # Log full message content without truncation
    log_file: str | None = None  # None = no logging, "" = default file

    # Track where config was loaded from (for display)
    config_file_path: Path | None = None

    # Internal: resolved model config from the models registry (set by _resolve_model)
    _active_model: "ModelConfig | None" = field(default=None, repr=False)

    # Internal: True when --provider was explicitly passed via CLI
    _cli_provider_override: bool = field(default=False, repr=False)

    # Embedding overrides populated from env vars (read by cogtrix.py)
    embedding_provider_override: str | None = None
    embedding_model_override: str | None = None

    # Research delegate settings
    research_delegate_enabled: bool = True
    research_delegate_timeout: int = 300
    research_delegate_cap_ratio: float = 0.85

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
        return result

    def get_provider_config(self, name: str | None = None) -> ProviderConfig:
        """Get configuration for a provider by name.

        Args:
            name: Provider name (uses self.provider if None)

        Returns:
            ProviderConfig for the requested provider

        Raises:
            ValueError: If provider is not configured
        """
        provider_name = name or self.provider
        if provider_name in self.providers:
            return self.providers[provider_name]
        raise ValueError(f"Unknown provider: '{provider_name}'. Available: {self.list_providers()}")

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        return sorted(self.providers.keys())

    def get_model_config(self, name: str | None = None) -> ModelConfig | None:
        """Look up a model in the models registry. Returns None if not found."""
        model_name = name or self.model
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
        pc = self.get_provider_config()
        return pc.type, None, pc.get_base_url(), pc.api_key

    def find_model_entry(self, target: str) -> "tuple[str | None, ModelConfig | None]":
        """Resolve *target* to a (canonical_alias, ModelConfig) pair.

        Resolution order:
        1. Exact alias key in self.models.
        2. Scan self.models values for .model == target (first match).
        3. Scan self.providers values for .model == target (synthesize ModelConfig).

        Returns (alias_or_None, ModelConfig_or_None). Both are None when not found.
        """
        if target in self.models:
            return target, self.models[target]
        for alias, mc in self.models.items():
            if mc.model == target:
                return alias, mc
        for pname, pc in self.providers.items():
            if pc.model == target:
                return None, ModelConfig(provider=pname, model=target)
        return None, None

    def resolve_provider_config(self) -> "ProviderConfig":
        """Get provider config with active model params merged.

        Returns a **clone** of the provider config so the original in
        ``self.providers`` is never mutated.  Model-specific ``num_ctx``
        and ``temperature`` from the active ``ModelConfig`` are merged
        into the clone.
        """
        from copy import copy

        pc = copy(self.get_provider_config())
        pc.model = self.model or pc.model
        if self._active_model:
            if self._active_model.num_ctx is not None:
                pc.num_ctx = self._active_model.num_ctx
            if self._active_model.temperature is not None:
                pc.temperature = self._active_model.temperature
            if self._active_model.max_tokens is not None:
                pc.max_tokens = self._active_model.max_tokens
        return pc


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

    # 5. Resolve model name against the models registry
    _resolve_model(config)

    # 6. Resolve final model based on provider if not explicitly set
    if config.model is None:
        try:
            provider_cfg = config.get_provider_config()
            config.model = provider_cfg.get_model()
        except ValueError:
            # Provider not configured — leave model as None; it will be
            # resolved from the provider's default when the LLM is created.
            pass

    return config


def _resolve_model(config: Config) -> None:
    """Resolve model name against the models registry.

    If config.model matches a key in config.models, update config.provider
    and store the resolved ModelConfig.  If not found, treat as a literal
    model name on the current provider.
    """
    if not config.model:
        config._active_model = None
        return

    mc = config.get_model_config()
    if mc is None:
        # Literal model name — no ModelConfig to merge
        config._active_model = None
        return

    # Resolve from models registry — skip provider override when CLI --provider was used
    if not config._cli_provider_override:
        config.provider = mc.provider
    config.model = mc.model
    config._active_model = mc


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
    if "provider" in data:
        config.provider = data["provider"]
    if "model" in data:
        config.model = data["model"]
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

    if "parallel_tool_execution" in data:
        config.parallel_tool_execution = bool(data["parallel_tool_execution"])

    # ── Allowed write paths ──────────────────────────────────────
    if "allowed_write_paths" in data:
        val = data["allowed_write_paths"]
        if isinstance(val, str):
            config.allowed_write_paths = [val]
        elif isinstance(val, list):
            config.allowed_write_paths = [str(p) for p in val]
        else:
            _log.warning("allowed_write_paths must be a string or list, ignoring")

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
            if "model" in cc:
                config.context_compression_model = str(cc["model"])
        else:
            config.context_compression = bool(cc)

    # ── MCP servers ──────────────────────────────────────────────
    if "mcp_servers" in data and isinstance(data["mcp_servers"], dict):
        config.mcp_servers = dict(data["mcp_servers"])

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

        raw_temperature = provider_data.get("temperature")
        raw_num_ctx = provider_data.get("num_ctx")
        raw_max_tokens = provider_data.get("max_tokens")
        try:
            config.providers[name] = ProviderConfig(
                name=name,
                type=provider_type,
                base_url=provider_data.get("base_url"),
                model=provider_data.get("model"),
                api_key=provider_data.get("api_key"),
                tool_instructions=provider_data.get("tool_instructions"),
                temperature=(
                    _safe_float(raw_temperature, f"providers.{name}.temperature")
                    if raw_temperature is not None
                    else None
                ),
                num_ctx=(
                    _safe_int(raw_num_ctx, f"providers.{name}.num_ctx")
                    if raw_num_ctx is not None
                    else None
                ),
                max_tokens=(
                    _safe_int(raw_max_tokens, f"providers.{name}.max_tokens")
                    if raw_max_tokens is not None
                    else None
                ),
            )
        except (ConfigError, ValueError, TypeError) as exc:
            _log.warning("Skipping provider '%s': %s", name, exc)


def _parse_models_section(config: Config, models_data: dict[str, Any]) -> None:
    """Parse the models section into ModelConfig objects."""
    for name, model_data in models_data.items():
        if isinstance(model_data, dict):
            provider = model_data.get("provider")
            model = model_data.get("model")
            if not provider or not model:
                _log.warning("Model '%s' missing required 'provider' or 'model' field", name)
                continue
            raw_num_ctx = model_data.get("num_ctx")
            raw_temperature = model_data.get("temperature")
            raw_max_tokens = model_data.get("max_tokens")
            try:
                config.models[name] = ModelConfig(
                    provider=provider,
                    model=model,
                    num_ctx=(
                        _safe_int(raw_num_ctx, f"models.{name}.num_ctx")
                        if raw_num_ctx is not None
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
                )
            except (ConfigError, ValueError, TypeError) as exc:
                _log.warning("Invalid model config '%s': %s", name, exc)
        elif isinstance(model_data, str):
            # String format: "provider/model" or just "model"
            if "/" in model_data:
                parts = model_data.split("/", 1)
                config.models[name] = ModelConfig(provider=parts[0], model=parts[1])
            else:
                # Just model name — use current provider
                config.models[name] = ModelConfig(provider=config.provider, model=model_data)


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
                model=preset["model"],
            )
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
    if env_val := os.getenv("COGTRIX_PROVIDER"):
        config.provider = env_val
    if env_val := os.getenv("COGTRIX_MODEL"):
        config.model = env_val
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


def _apply_cli_args(config: Config, args) -> None:
    """Apply settings from command line arguments."""
    if hasattr(args, "provider") and args.provider:
        config.provider = args.provider
        config._cli_provider_override = True
    if hasattr(args, "model") and args.model:
        config.model = args.model
    if hasattr(args, "session") and args.session:
        config.session = args.session
    if hasattr(args, "data_dir") and args.data_dir:
        config.data_dir = args.data_dir
    if hasattr(args, "memory_mode") and args.memory_mode:
        config.memory_mode = args.memory_mode

    # Debug and logging settings
    if hasattr(args, "debug") and args.debug:
        config.debug = True
        # Debug mode auto-enables logging and verbose if not already set
        if config.log_file is None:
            config.log_file = ""  # Empty string = use default file
        config.verbose = True  # Debug mode enables verbose logging

    # --verbose can be used without --debug
    if hasattr(args, "verbose") and args.verbose:
        config.verbose = True

    # --log can be used without --debug
    if hasattr(args, "log") and args.log is not None:
        config.log_file = args.log

    # Allowed write paths
    if hasattr(args, "allow_write_path") and args.allow_write_path:
        config.allowed_write_paths = list(args.allow_write_path)
