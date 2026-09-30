from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec
from ...observability import is_sensitive_key, redact


def _is_sensitive_setting(setting: str) -> bool:
    return any(is_sensitive_key(part) for part in setting.split("."))


def _is_endpoint_setting(setting: str) -> bool:
    last = setting.split(".")[-1].lower()
    return last in {"base_url", "baseurl", "endpoint", "url"}


class ConfigTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Config",
            description='Get or set Clawd configuration values (e.g. "default_provider", "providers.openai.base_url").',
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "setting": {"type": "string"},
                    "value": {"oneOf": [{"type": "string"}, {"type": "boolean"}, {"type": "number"}]},
                },
                "required": ["setting"],
            },
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def check_permissions(self, tool_input: dict[str, Any], context: ToolContext) -> PermissionResult:
        setting = tool_input.get("setting")
        if not isinstance(setting, str) or "value" not in tool_input:
            return PermissionResult.allow()  # reads are redacted in run()
        if _is_sensitive_setting(setting):
            return PermissionResult.deny(f"Config tool may not set credentials ({setting}); use env vars or 'clawd login'")
        if _is_endpoint_setting(setting):
            return PermissionResult.ask(
                message=f"Changing the API endpoint '{setting}' redirects where your API key is sent; approval required",
            )
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        from src import config as config_mod

        setting = tool_input["setting"]
        value_provided = "value" in tool_input
        if not isinstance(setting, str) or not setting:
            raise ToolInputError("setting must be a non-empty string")

        cfg = config_mod.load_config()

        if not value_provided:
            value = _get_setting(cfg, setting)
            # Never hand credentials to the model: mask secret keys at any depth.
            value = "[REDACTED]" if _is_sensitive_setting(setting) and value not in (None, "") else redact(value)
            return ToolResult(
                name="Config",
                output={"success": True, "operation": "get", "setting": setting, "value": value},
            )

        value = tool_input.get("value")
        if _is_sensitive_setting(setting):
            raise ToolPermissionError(f"Config tool may not set credentials ({setting})")
        prev = redact(_get_setting(cfg, setting))
        _set_setting(cfg, setting, value)
        config_mod.save_config(cfg)
        return ToolResult(
            name="Config",
            output={
                "success": True,
                "operation": "set",
                "setting": setting,
                "previousValue": prev,
                "newValue": value,
            },
        )


def _get_setting(cfg: dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_setting(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: Any = cfg
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise ToolInputError(f"cannot set {key}: encountered non-object at {part}")
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    last = parts[-1]
    if not isinstance(cur, dict):
        raise ToolInputError(f"cannot set {key}: encountered non-object at {last}")
    cur[last] = value

