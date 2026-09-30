"""LLM Providers for Clawd Codex."""

from __future__ import annotations

from typing import TypedDict

from .base import BaseProvider, ChatMessage, ChatResponse


# Provider metadata for login/UI
class ProviderInfo(TypedDict):
    label: str
    default_base_url: str
    default_model: str
    available_models: list[str]
    # Environment variables checked (in order) for the API key / base URL.
    api_key_env: list[str]
    base_url_env: list[str]
    # Wire format spoken by the provider ("anthropic" | "openai").
    wire_format: str


PROVIDER_INFO: dict[str, ProviderInfo] = {
    "anthropic": {
        "label": "Anthropic Claude",
        "api_key_env": ["ANTHROPIC_API_KEY"],
        "base_url_env": ["ANTHROPIC_BASE_URL"],
        "wire_format": "anthropic",
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-6",
        "available_models": [
            # Claude 4 series (latest)
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-0",
            "claude-sonnet-4-20250514",
            "claude-opus-4-6",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
            "claude-opus-4-0",
            "claude-opus-4-20250514",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            # Legacy
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "openai": {
        "label": "OpenAI GPT",
        "api_key_env": ["OPENAI_API_KEY"],
        "base_url_env": ["OPENAI_BASE_URL"],
        "wire_format": "openai",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.4",
        "available_models": [
            # GPT-5.4 series (latest flagship)
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            # GPT-5.2 series (previous)
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.2-mini",
            "gpt-5.2-nano",
            # GPT-5.3-Codex (coding-specialized)
            "gpt-5.3-codex",
            # Legacy GPT-4 series
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
    },
    "glm": {
        "label": "Zhipu GLM (z.ai)",
        "api_key_env": ["ZHIPUAI_API_KEY", "GLM_API_KEY", "ZAI_API_KEY"],
        "base_url_env": ["GLM_BASE_URL"],
        "wire_format": "openai",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "zai/glm-5",
        "available_models": [
            # GLM-5 series (latest, requires zai/ prefix)
            "zai/glm-5",
            "zai/glm-5-turbo",
            # GLM-4 series (standard, zai/ prefix)
            "zai/glm-4",
            "zai/glm-4-plus",
            "zai/glm-4-air",
            "zai/glm-4-flash",
            "zai/glm-4.5",
            "zai/glm-4.6",
            "zai/glm-4.7",
            # GLM-3 series (legacy)
            "zai/glm-3-turbo",
        ],
    },
    "minimax": {
        "label": "Minimax AI",
        "api_key_env": ["MINIMAX_API_KEY"],
        "base_url_env": ["MINIMAX_BASE_URL"],
        "wire_format": "anthropic",
        "default_base_url": "https://api.minimaxi.com/anthropic",
        "default_model": "MiniMax-M2.7",
        "available_models": [
            # M2 series (latest)
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "M2-her",
            # Historical
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2",
        ],
    },
}


def get_provider_info(provider_name: str) -> ProviderInfo:
    """Get provider info by name."""
    if provider_name not in PROVIDER_INFO:
        raise ValueError(f"Unknown provider: {provider_name}")
    return PROVIDER_INFO[provider_name]


# Provider implementations, imported lazily so optional SDKs are not required
# just to start the CLI. New providers (e.g. Gemini, Qwen, Ollama) register an
# entry here plus a PROVIDER_INFO record, or call ``register_provider``.
_PROVIDER_CLASSES: dict[str, tuple[str, str]] = {
    "anthropic": ("src.providers.anthropic_provider", "AnthropicProvider"),
    "openai": ("src.providers.openai_provider", "OpenAIProvider"),
    "glm": ("src.providers.glm_provider", "GLMProvider"),
    "minimax": ("src.providers.minimax_provider", "MinimaxProvider"),
}


def register_provider(name: str, module: str, class_name: str, info: ProviderInfo) -> None:
    """Register an additional provider implementation (extension point)."""
    _PROVIDER_CLASSES[name] = (module, class_name)
    PROVIDER_INFO[name] = info


def get_provider_class(provider_name: str):
    """Get provider class by name."""
    entry = _PROVIDER_CLASSES.get(provider_name)
    if entry is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    import importlib

    module = importlib.import_module(entry[0])
    return getattr(module, entry[1])


# Legacy registry for display purposes
AVAILABLE_PROVIDERS: dict[str, str] = {k: v["label"] for k, v in PROVIDER_INFO.items()}


__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "get_provider_class",
    "register_provider",
    "get_provider_info",
    "PROVIDER_INFO",
    "AVAILABLE_PROVIDERS",
]
