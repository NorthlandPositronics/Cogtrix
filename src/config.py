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

JSON example::

    {
      "provider": "spark-cluster",
      "session": "default",

      "inference": {
        "spark-cluster": {
          "type": "openai",
          "base_url": "http://192.168.70.254:8080/v1",
          "model": "gpt-oss",
          "api_key": "sk-...",
          "num_ctx": 32768,
          "temperature": 0.7
        },
        "openai": { "type": "openai", "model": "gpt-4.1", "api_key": "sk-..." }
      },

      "embedding": { "provider": "ollama", "model": "nomic-embed-text-v2-moe" },

      "services": {
        "openweather": { "api_key": "..." },
        "tavily":      { "api_key": "..." },
        "exa":         { "api_key": "..." },
        "brave":       { "api_key": "..." },
        "serpapi":     { "api_key": "..." },
        "google":      { "api_key": "...", "cse_id": "..." }
      },

      "model_aliases": {
        "fast": "spark-cluster/nemotron-nano",
        "reasoning": {
          "provider": "spark-cluster", "model": "gpt-oss",
          "timeout": 400, "temperature": 0.3
        }
      },

      "delegate": { ... },
      "memory":   { ... },
      "rag":      { ... }
    }

YAML equivalent::

    provider: spark-cluster
    session: default

    inference:
      spark-cluster:
        type: openai
        base_url: "http://192.168.70.254:8080/v1"
        model: gpt-oss
        api_key: "sk-..."
        num_ctx: 32768
        temperature: 0.7
      openai:
        type: openai
        model: gpt-4.1
        api_key: "sk-..."

    embedding:
      provider: ollama
      model: nomic-embed-text-v2-moe

    services:
      openweather:
        api_key: "..."
      tavily:
        api_key: "..."

    model_aliases:
      fast: spark-cluster/nemotron-nano
      reasoning:
        provider: spark-cluster
        model: gpt-oss
        timeout: 400
        temperature: 0.3

Backward compatibility:
    - ``"providers": {...}`` is accepted as an alias for ``"inference"``
    - Top-level ``"openweather": {"api_key": ...}`` etc. still work
    - ``"delegate": {"model_aliases": ...}`` still works (merged with
      top-level ``"model_aliases"``; top-level takes priority)
    - Legacy ``"openai": {...}`` / ``"ollama": {...}`` sections still work
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger("cogtrix")


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
    temperature: float | None = None
    num_ctx: int | None = None  # Context window size (tokens) for any provider
    tool_instructions: str | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialisation."""
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.num_ctx is not None and self.num_ctx < 256:
            raise ValueError(f"Context window (num_ctx) must be >= 256, got {self.num_ctx}")

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
            "tool_instructions": self.tool_instructions,
        }


@dataclass
class RAGConfig:
    """Configuration for RAG document ingestion."""

    docs_dir: str = "docs"
    vectordb_dir: str = "data/vectordb"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_provider: str = "ollama"
    embedding_model: str | None = None


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding subsystem."""

    provider: str = "ollama"
    model: str | None = None


@dataclass
class Config:
    """Application configuration with defaults."""

    # General settings
    provider: str = "ollama"
    model: str | None = None
    session: str = "default"

    # Inference providers (LLM backends)
    # Populated from "inference" (preferred) or "providers" (backward compat)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # External services — flat dict of {service_name: {config...}}
    # Populated from "services" section or legacy top-level keys
    services: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Legacy OpenAI settings (for backward compatibility)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    # Legacy Ollama settings (for backward compatibility)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    # Memory settings
    memory_mode: str = "conversation"
    memory_config: dict[str, Any] | None = field(default=None)
    memory_modes: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Model aliases (top-level; also read from delegate.model_aliases for compat)
    model_aliases: dict[str, Any] | None = field(default=None)

    # Delegate tool settings
    delegate_enabled: bool = True
    delegate_default_timeout: int = 60
    delegate_allowed_providers: list | None = field(default=None)
    delegate_allowed_models: list[str] | None = field(default=None)

    # Prompt optimizer — rewrite complex prompts before agent execution
    prompt_optimizer: bool = True

    # Context compression — summarize old ToolMessages during agent loop
    context_compression: bool = True
    context_compression_min_age: int = 6
    context_compression_min_chars: int = 2000
    context_compression_model: str | None = None  # model alias or "provider/model"

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

    def get_provider_config(self, name: str | None = None) -> ProviderConfig:
        """
        Get configuration for a provider by name.

        Args:
            name: Provider name (uses self.provider if None)

        Returns:
            ProviderConfig for the requested provider

        Raises:
            ValueError: If provider is not configured
        """
        provider_name = name or self.provider

        # Check named providers first
        if provider_name in self.providers:
            return self.providers[provider_name]

        # Fallback to legacy built-in providers
        if provider_name == "openai":
            return ProviderConfig(
                name="openai",
                type="openai",
                model=self.openai_model,
                api_key=self.openai_api_key,
            )
        elif provider_name == "ollama":
            return ProviderConfig(
                name="ollama",
                type="ollama",
                base_url=self.ollama_base_url,
                model=self.ollama_model,
            )

        raise ValueError(
            f"Unknown provider: '{provider_name}'. " f"Available: {self.list_providers()}"
        )

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        # Start with named providers
        names = list(self.providers.keys())
        # Add legacy providers if not overridden
        if "openai" not in names:
            names.append("openai")
        if "ollama" not in names:
            names.append("ollama")
        return sorted(names)


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

    # 5. Resolve model alias if -m matches an alias name
    _resolve_model_alias(config)

    # 6. Resolve final model based on provider if not explicitly set
    if config.model is None:
        try:
            provider_cfg = config.get_provider_config()
            config.model = provider_cfg.get_model()
        except ValueError:
            # Fallback for unknown provider — match the default (ollama)
            config.model = "qwen3:8b"

    return config


def _resolve_model_alias(config: Config) -> None:
    """
    Resolve model alias if config.model matches an alias name.

    Model aliases from ``model_aliases`` (or legacy ``delegate.model_aliases``)
    can be used with the ``-m`` flag.  If matched, updates both
    ``config.provider`` and ``config.model``.  If not matched, the value
    is treated as a literal model name.

    Alias formats supported:
    - String: "provider/model" or just "model"
    - Object: {"provider": "...", "model": "...", "num_ctx": ..., ...}
    """
    if not config.model:
        return

    aliases = config.model_aliases or {}

    # Check if it's an alias
    if config.model not in aliases:
        # Not an alias - treat as literal model name
        # Update the provider config to use this model
        if config.provider and config.provider in config.providers:
            config.providers[config.provider].model = config.model
        return

    alias_value = aliases[config.model]

    # Object format: {"provider": "...", "model": "...", ...}
    if isinstance(alias_value, dict):
        resolved_provider = alias_value.get("provider", config.provider)
        resolved_model = alias_value.get("model")

        config.provider = resolved_provider
        if resolved_model:
            config.model = resolved_model

        # Update provider config with alias settings (model, num_ctx, temperature)
        if config.provider in config.providers:
            prov_cfg = config.providers[config.provider]
            # Override model in provider config
            if resolved_model:
                prov_cfg.model = resolved_model
            if "num_ctx" in alias_value and alias_value["num_ctx"] is not None:
                prov_cfg.num_ctx = int(alias_value["num_ctx"])
            if "temperature" in alias_value and alias_value["temperature"] is not None:
                prov_cfg.temperature = float(alias_value["temperature"])
        return

    # String format: "provider/model" or just "model"
    if isinstance(alias_value, str):
        if "/" in alias_value:
            parts = alias_value.split("/", 1)
            config.provider = parts[0]
            config.model = parts[1]
            # Update provider config model
            if config.provider in config.providers:
                config.providers[config.provider].model = parts[1]
        else:
            # Just model name, keep current provider
            config.model = alias_value
            # Update provider config model
            if config.provider in config.providers:
                config.providers[config.provider].model = alias_value


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
    """
    Apply settings from configuration file (JSON or YAML).

    Supports the current structure (``inference``, ``embedding``,
    ``services``, ``model_aliases``) as well as all legacy formats
    (``providers``, top-level service keys, ``delegate.model_aliases``).

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

    # ── Inference providers ────────────────────────────────────────
    # Preferred key: "inference"; backward-compat: "providers"
    inference_data = data.get("inference") if "inference" in data else data.get("providers")
    if isinstance(inference_data, dict):
        _parse_providers_section(config, inference_data)

    # Legacy top-level "openai" / "ollama" sections
    if "openai" in data and isinstance(data["openai"], dict):
        openai_cfg = data["openai"]
        if "api_key" in openai_cfg:
            config.openai_api_key = openai_cfg["api_key"]
        if "model" in openai_cfg:
            config.openai_model = openai_cfg["model"]
        if "openai" not in config.providers:
            _add_legacy_provider(config, "openai", openai_cfg)

    if "ollama" in data and isinstance(data["ollama"], dict):
        ollama_cfg = data["ollama"]
        if "base_url" in ollama_cfg:
            config.ollama_base_url = ollama_cfg["base_url"]
        if "model" in ollama_cfg:
            config.ollama_model = ollama_cfg["model"]
        if "ollama" not in config.providers:
            _add_legacy_provider(config, "ollama", ollama_cfg)

    # ── Embedding ──────────────────────────────────────────────────
    if "embedding" in data and isinstance(data["embedding"], dict):
        emb_cfg = data["embedding"]
        if "provider" in emb_cfg:
            config.embedding.provider = emb_cfg["provider"]
        if "model" in emb_cfg:
            config.embedding.model = emb_cfg["model"]

    # ── Services (external APIs) ──────────────────────────────────
    # New format: consolidated "services" section
    if "services" in data and isinstance(data["services"], dict):
        for svc_name, svc_cfg in data["services"].items():
            if isinstance(svc_cfg, dict):
                config.services[svc_name] = svc_cfg

    # Legacy format: top-level keys like "openweather", "tavily", etc.
    # These are merged into services (services section takes priority).
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

    # ── Model aliases ─────────────────────────────────────────────
    # Top-level "model_aliases" is the preferred location.
    if "model_aliases" in data and isinstance(data["model_aliases"], dict):
        config.model_aliases = data["model_aliases"]

    # ── Memory settings ───────────────────────────────────────────
    if "memory" in data and isinstance(data["memory"], dict):
        memory_cfg = data["memory"]
        if "mode" in memory_cfg:
            config.memory_mode = memory_cfg["mode"]
        if "modes" in memory_cfg and isinstance(memory_cfg["modes"], dict):
            # Store all mode configs for live mode switching
            config.memory_modes = dict(memory_cfg["modes"])
            mode = config.memory_mode
            if mode in memory_cfg["modes"]:
                config.memory_config = memory_cfg["modes"][mode]

    # ── Delegate tool settings ────────────────────────────────────
    if "delegate" in data and isinstance(data["delegate"], dict):
        delegate_cfg = data["delegate"]
        if "enabled" in delegate_cfg:
            config.delegate_enabled = bool(delegate_cfg["enabled"])
        # "max_depth" is accepted for backward compat but not used
        if "default_timeout" in delegate_cfg:
            config.delegate_default_timeout = int(delegate_cfg["default_timeout"])
        if "allowed_providers" in delegate_cfg:
            config.delegate_allowed_providers = delegate_cfg["allowed_providers"]
        if "allowed_models" in delegate_cfg:
            config.delegate_allowed_models = delegate_cfg["allowed_models"]
        # Backward compat: delegate.model_aliases → config.model_aliases
        if "model_aliases" in delegate_cfg and config.model_aliases is None:
            config.model_aliases = delegate_cfg["model_aliases"]

    # ── Prompt optimizer ─────────────────────────────────────────
    if "prompt_optimizer" in data:
        config.prompt_optimizer = bool(data["prompt_optimizer"])

    # ── Context compression ──────────────────────────────────────
    if "context_compression" in data:
        cc = data["context_compression"]
        if isinstance(cc, dict):
            config.context_compression = bool(cc.get("enabled", True))
            if "min_age" in cc:
                config.context_compression_min_age = int(cc["min_age"])
            if "min_chars" in cc:
                config.context_compression_min_chars = int(cc["min_chars"])
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
            config.rag.chunk_size = int(rag_cfg["chunk_size"])
        if "chunk_overlap" in rag_cfg:
            config.rag.chunk_overlap = int(rag_cfg["chunk_overlap"])
        if "embedding_provider" in rag_cfg:
            config.rag.embedding_provider = rag_cfg["embedding_provider"]
        if "embedding_model" in rag_cfg:
            config.rag.embedding_model = rag_cfg["embedding_model"]

    # ── Sync embedding into RAG if RAG hasn't overridden ──────────
    # The "embedding" section is the source of truth; if rag.embedding_*
    # were not explicitly set, inherit from the embedding section.
    if "embedding" in data and isinstance(data["embedding"], dict):
        rag_section = data.get("rag", {})
        if "embedding_provider" not in rag_section:
            config.rag.embedding_provider = config.embedding.provider
        if "embedding_model" not in rag_section:
            config.rag.embedding_model = config.embedding.model


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

        from src.providers import PROVIDER_TYPES

        if provider_type not in PROVIDER_TYPES:
            _log.warning(
                "Provider '%s' has unknown type '%s' (supported: %s)",
                name,
                provider_type,
                ", ".join(sorted(PROVIDER_TYPES)),
            )
            continue

        config.providers[name] = ProviderConfig(
            name=name,
            type=provider_type,
            base_url=provider_data.get("base_url"),
            model=provider_data.get("model"),
            api_key=provider_data.get("api_key"),
            temperature=provider_data.get("temperature"),
            num_ctx=provider_data.get("num_ctx"),
            tool_instructions=provider_data.get("tool_instructions"),
        )


def _add_legacy_provider(config: Config, name: str, legacy_data: dict[str, Any]) -> None:
    """Convert legacy provider config to new ProviderConfig format.

    ``name`` is always ``"openai"`` or ``"ollama"`` — legacy configs
    don't carry an explicit ``type`` field, so the section name doubles
    as the provider type.
    """
    config.providers[name] = ProviderConfig(
        name=name,
        type=name,
        base_url=legacy_data.get("base_url"),
        model=legacy_data.get("model"),
        api_key=legacy_data.get("api_key"),
    )


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
            config.providers[name] = ProviderConfig(
                name=name,
                type=name,
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
    if ":" in value:
        # Exactly one colon → host:port; multiple colons → IPv6 address.
        # Bracketed IPv6 like [::1]:8080 is fine — rsplit on the last ":"
        # yields "[::1]" and "8080".
        if value.count(":") == 1 or (value.startswith("[") and "]:" in value):
            host, port = value.rsplit(":", 1)
            if port.isdigit():
                return f"http://{host}:{port}"
        # Multiple colons without brackets — bare IPv6; treat as hostname
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

    # LLM provider API keys
    if env_val := os.getenv("OPENAI_API_KEY"):
        config.openai_api_key = env_val
    if env_val := os.getenv("ANTHROPIC_API_KEY"):
        _set_provider_key(config, "anthropic", env_val)
    if env_val := os.getenv("GEMINI_API_KEY"):
        _set_provider_key(config, "google", env_val)
    if env_val := os.getenv("XAI_API_KEY"):
        _set_provider_key(config, "xai", env_val)

    # Ollama settings
    # COGTRIX_OLLAMA accepts "host:port" or just "host" (default port: 11434)
    if env_val := os.getenv("COGTRIX_OLLAMA"):
        config.ollama_base_url = _parse_ollama_address(env_val)
    # Legacy env var (full URL) — overridden by COGTRIX_OLLAMA if both set
    if env_val := os.getenv("OLLAMA_BASE_URL"):
        if not os.getenv("COGTRIX_OLLAMA"):
            config.ollama_base_url = env_val

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


def _apply_cli_args(config: Config, args) -> None:
    """Apply settings from command line arguments."""
    if hasattr(args, "provider") and args.provider:
        config.provider = args.provider
    if hasattr(args, "model") and args.model:
        config.model = args.model
    if hasattr(args, "session") and args.session:
        config.session = args.session
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
