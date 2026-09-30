"""Configuration management for Clawd Codex."""

from __future__ import annotations

import json
import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .branding import PROJECT_DIR_NAME, PROJECT_SETTINGS_FILE, USER_DIR_NAME, env_lookup


def get_user_dir() -> Path:
    """User-level state directory (default ``~/.clawd``).

    Override with ``INCRESCENT_HOME`` / ``CLAWD_HOME`` (useful for CI and tests).
    """
    override, _ = env_lookup("HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / USER_DIR_NAME


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    config_dir = get_user_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def _get_default_config_from_providers() -> dict[str, Any]:
    """Build default config using provider info registry."""
    from src.providers import PROVIDER_INFO

    return {
        "default_provider": "anthropic",
        "providers": {
            name: {
                "api_key": "",
                "base_url": info["default_base_url"],
                "default_model": info["default_model"],
            }
            for name, info in PROVIDER_INFO.items()
        },
        "session": {
            "auto_save": True,
            "max_history": 100
        }
    }


def get_default_config() -> dict[str, Any]:
    """Generate default configuration."""
    return _get_default_config_from_providers()


def _encode_api_key(api_key: str) -> str:
    """Encode API key for basic obfuscation."""
    return base64.b64encode(api_key.encode()).decode()


def _decode_api_key(encoded_key: str) -> str:
    """Decode API key."""
    try:
        return base64.b64decode(encoded_key.encode()).decode()
    except Exception:
        # If decoding fails, return as-is (might be plain text)
        return encoded_key


def load_config(create_if_missing: bool = True) -> dict[str, Any]:
    """Load configuration from file.

    Args:
        create_if_missing: write a default config file when none exists.

    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()

    if not config_path.exists():
        config = get_default_config()
        if create_if_missing:
            save_config(config)
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Decode API keys
        for provider_name, provider_config in config.get("providers", {}).items():
            if provider_config.get("api_key"):
                provider_config["api_key"] = _decode_api_key(provider_config["api_key"])

        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    config_copy = json.loads(json.dumps(config))

    # Encode API keys
    for provider_name, provider_config in config_copy.get("providers", {}).items():
        if provider_config.get("api_key"):
            provider_config["api_key"] = _encode_api_key(provider_config["api_key"])

    if os.name == "nt":
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
    else:
        fd = os.open(
            config_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
        os.chmod(config_path, 0o600)


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (anthropic, openai, glm, minimax)

    Returns:
        Provider configuration dictionary
    """
    config = load_config()
    providers = config.get("providers", {})

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider]


def set_api_key(provider: str, api_key: str, base_url: Optional[str] = None,
                default_model: Optional[str] = None) -> None:
    """Set API key for a provider.

    Args:
        provider: Provider name (anthropic, openai, glm, minimax)
        api_key: API key to set
        base_url: Optional base URL override
        default_model: Optional default model override
    """
    config = load_config()

    if provider not in config.get("providers", {}):
        # Add new provider if it doesn't exist
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][provider] = {}

    config["providers"][provider]["api_key"] = api_key

    if base_url is not None:
        config["providers"][provider]["base_url"] = base_url

    if default_model is not None:
        config["providers"][provider]["default_model"] = default_model

    save_config(config)


def set_default_provider(provider: str) -> None:
    """Set the default provider.

    Args:
        provider: Provider name
    """
    config = load_config()
    config["default_provider"] = provider
    save_config(config)


def get_default_provider() -> str:
    """Get the default provider.

    Returns:
        Default provider name
    """
    config = load_config()
    return config.get("default_provider", "anthropic")


# ---------------------------------------------------------------------------
# Layered provider configuration
# ---------------------------------------------------------------------------
#
# Precedence (highest first):
#   1. explicit CLI argument        (--provider / --model / --base-url)
#   2. environment variable         (INCRESCENT_*/CLAWD_* and provider key vars)
#   3. project configuration        (<project>/.clawd/settings.json: provider, model)
#   4. user configuration           (~/.clawd/config.json)
#   5. safe default                 (anthropic + provider default model)
#
# Secrets policy: API keys are read only from environment variables or the user
# config. They are never accepted on the command line (argv is visible to other
# processes) and never read from project files (repositories are shared).
# Project files also may not set a base URL: a cloned repository must not be
# able to redirect your API key to another server.

DEFAULT_PROVIDER = "anthropic"
PROJECT_ALLOWED_KEYS = frozenset({"provider", "model"})


class ConfigError(Exception):
    """Invalid or incomplete configuration (CLI exit code 2)."""


@dataclass(frozen=True)
class ResolvedProviderConfig:
    provider: str
    model: str
    base_url: Optional[str]
    api_key: str = field(repr=False, default="")
    # field -> where the value came from ("cli", "env:NAME", "project:<path>", "user", "default")
    sources: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> dict[str, Any]:
        """Safe-to-print summary (never includes the key itself)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "set" if self.api_key else "missing",
            "sources": dict(self.sources),
        }


def get_project_settings_path(project_root: str | Path) -> Path:
    return Path(project_root) / PROJECT_DIR_NAME / PROJECT_SETTINGS_FILE


def load_project_settings(project_root: str | Path | None) -> tuple[dict[str, Any], Optional[Path], list[str]]:
    """Read project settings, keeping only non-secret keys."""
    if project_root is None:
        return {}, None, []
    path = get_project_settings_path(project_root)
    if not path.is_file():
        return {}, None, []
    warnings: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, path, [f"Ignoring unreadable project settings {path}: {e}"]
    if not isinstance(data, dict):
        return {}, path, [f"Ignoring project settings {path}: expected a JSON object"]
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key in PROJECT_ALLOWED_KEYS and isinstance(value, str) and value.strip():
            clean[key] = value.strip()
        elif key == "permissions":
            continue  # handled by load_permission_rules (tighten-only)
        elif key not in PROJECT_ALLOWED_KEYS:
            warnings.append(
                f"Ignoring '{key}' in project settings {path}: only {sorted(PROJECT_ALLOWED_KEYS)} "
                "may be set per project (API keys and base URLs belong in env vars or the user config)"
            )
    return clean, path, warnings


def _provider_env(provider: str, kind: str, env: Mapping[str, str]) -> tuple[Optional[str], Optional[str]]:
    from src.providers import PROVIDER_INFO

    info = PROVIDER_INFO.get(provider) or {}
    for name in info.get(kind, []) or []:  # type: ignore[union-attr]
        value = env.get(name)
        if value is not None and value.strip():
            return value.strip(), name
    return None, None


def resolve_provider_config(
    *,
    cli_provider: Optional[str] = None,
    cli_model: Optional[str] = None,
    cli_base_url: Optional[str] = None,
    project_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedProviderConfig:
    """Resolve provider, model, base URL and API key from all config layers."""
    from src.providers import PROVIDER_INFO

    env = os.environ if env is None else env
    user = load_config(create_if_missing=False) if get_config_path().exists() else {}
    user_providers = user.get("providers", {}) if isinstance(user.get("providers"), dict) else {}
    project, project_path, warnings = load_project_settings(project_root)
    project_label = f"project:{project_path}" if project_path else "project"
    sources: dict[str, str] = {}

    # Provider
    env_provider, env_provider_var = env_lookup("PROVIDER", env)
    if cli_provider:
        provider, sources["provider"] = cli_provider, "cli"
    elif env_provider:
        provider, sources["provider"] = env_provider, f"env:{env_provider_var}"
    elif project.get("provider"):
        provider, sources["provider"] = project["provider"], project_label
    elif user.get("default_provider"):
        provider, sources["provider"] = str(user["default_provider"]), "user"
    else:
        provider, sources["provider"] = DEFAULT_PROVIDER, "default"
    provider = provider.strip().lower()
    if provider not in PROVIDER_INFO:
        raise ConfigError(
            f"Unknown provider '{provider}' (from {sources['provider']}). "
            f"Available: {', '.join(sorted(PROVIDER_INFO))}"
        )
    info = PROVIDER_INFO[provider]
    raw_user_cfg = user_providers.get(provider)
    user_provider_cfg = raw_user_cfg if isinstance(raw_user_cfg, dict) else {}

    # Model (a project model applies only when the project did not pin another provider)
    env_model, env_model_var = env_lookup("MODEL", env)
    project_model_applies = bool(project.get("model")) and project.get("provider", provider).lower() == provider
    if cli_model:
        model, sources["model"] = cli_model, "cli"
    elif env_model:
        model, sources["model"] = env_model, f"env:{env_model_var}"
    elif project_model_applies:
        model, sources["model"] = project["model"], project_label
    elif user_provider_cfg.get("default_model"):
        model, sources["model"] = str(user_provider_cfg["default_model"]), "user"
    else:
        model, sources["model"] = info["default_model"], "default"

    # Base URL (never from project files)
    env_base_url, env_base_var = _provider_env(provider, "base_url_env", env)
    if cli_base_url:
        base_url, sources["base_url"] = cli_base_url, "cli"
    elif env_base_url:
        base_url, sources["base_url"] = env_base_url, f"env:{env_base_var}"
    elif user_provider_cfg.get("base_url"):
        base_url, sources["base_url"] = str(user_provider_cfg["base_url"]), "user"
    else:
        base_url, sources["base_url"] = info["default_base_url"], "default"

    # API key (env or user config only)
    env_key, env_key_var = _provider_env(provider, "api_key_env", env)
    if env_key:
        api_key, sources["api_key"] = env_key, f"env:{env_key_var}"
    elif user_provider_cfg.get("api_key"):
        api_key, sources["api_key"] = str(user_provider_cfg["api_key"]), "user"
    else:
        api_key, sources["api_key"] = "", "missing"

    return ResolvedProviderConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        sources=sources,
        warnings=tuple(warnings),
    )


def missing_api_key_message(resolved: ResolvedProviderConfig) -> str:
    from src.providers import PROVIDER_INFO

    env_names = PROVIDER_INFO.get(resolved.provider, {}).get("api_key_env", [])  # type: ignore[union-attr]
    hint = f"set {' or '.join(env_names)}" if env_names else "set the provider API key environment variable"
    return (
        f"No API key configured for provider '{resolved.provider}'. "
        f"Either {hint}, or run 'clawd login'."
    )


def create_provider(resolved: ResolvedProviderConfig):
    """Instantiate the provider described by a resolved configuration."""
    from src.providers import get_provider_class

    if not resolved.api_key:
        raise ConfigError(missing_api_key_message(resolved))
    provider_class = get_provider_class(resolved.provider)
    return provider_class(api_key=resolved.api_key, base_url=resolved.base_url, model=resolved.model)


# ---------------------------------------------------------------------------
# Permission rules (Phase 2)
# ---------------------------------------------------------------------------
#
# ~/.clawd/config.json and <project>/.clawd/settings.json may contain:
#     "permissions": {"allow": ["Bash(npm test*)"], "ask": ["Write(*.lock)"], "deny": ["Bash(git push*)"]}
# User rules may allow, ask or deny. Project rules (shipped inside a repository)
# may only tighten (ask/deny): a cloned repo must not be able to pre-approve actions.

def load_permission_rules(project_root: str | Path | None = None) -> tuple[list[Any], list[str]]:
    from src.security.permissions import Decision, PermissionRule

    rules: list[Any] = []
    warnings: list[str] = []

    def collect(section: Any, source: str, allowed: tuple[str, ...]) -> None:
        if not isinstance(section, dict):
            return
        for decision in ("deny", "ask", "allow"):
            for text in section.get(decision) or []:
                if decision not in allowed:
                    warnings.append(f"Ignoring {decision} rule {text!r} from {source}: project settings may only add ask/deny rules")
                    continue
                try:
                    rules.append(PermissionRule.parse(str(text), Decision(decision.upper())))
                except ValueError as e:
                    warnings.append(f"Ignoring invalid rule {text!r} from {source}: {e}")

    if get_config_path().exists():
        collect(load_config(create_if_missing=False).get("permissions"), "user config", ("allow", "ask", "deny"))
    if project_root is not None:
        path = get_project_settings_path(project_root)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            if isinstance(data, dict):
                collect(data.get("permissions"), str(path), ("ask", "deny"))
    return rules, warnings


def load_session_policy(project_root: str | Path | None = None):
    """Session policy (Phase 1 defaults) plus configured rules. Returns (policy, warnings)."""
    from src.security.permissions import PermissionPolicy

    rules, warnings = load_permission_rules(project_root)
    return PermissionPolicy.session_default().with_rules(rules), warnings
