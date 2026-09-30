"""Layered provider configuration: CLI > env > project > user > default."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import (
    ConfigError,
    create_provider,
    get_config_path,
    get_user_dir,
    resolve_provider_config,
    save_config,
)
from src.providers import PROVIDER_INFO

USER_KEY = "user-config-key-123456"
ENV_KEY = "env-key-abcdef-123456"


class ConfigLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _user_config(self, **providers) -> None:
        save_config({
            "default_provider": "glm",
            "providers": providers or {
                "glm": {"api_key": USER_KEY, "base_url": "https://user.glm", "default_model": "glm-user"},
                "openai": {"api_key": USER_KEY, "base_url": "https://user.openai", "default_model": "gpt-user"},
            },
        })

    def _project_settings(self, data: dict) -> None:
        path = self.project / ".clawd" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def resolve(self, env=None, **kwargs):
        return resolve_provider_config(project_root=self.project, env=env or {}, **kwargs)

    def test_safe_default_without_any_config(self):
        r = self.resolve()
        self.assertEqual(r.provider, "anthropic")
        self.assertEqual(r.model, PROVIDER_INFO["anthropic"]["default_model"])
        self.assertEqual(r.sources["provider"], "default")
        self.assertEqual(r.sources["model"], "default")
        self.assertFalse(r.has_api_key)
        self.assertFalse(get_config_path().exists(), "resolution must not create a config file")

    def test_user_config_is_used(self):
        self._user_config()
        r = self.resolve()
        self.assertEqual((r.provider, r.model, r.base_url, r.api_key), ("glm", "glm-user", "https://user.glm", USER_KEY))
        self.assertEqual(r.sources["api_key"], "user")

    def test_project_overrides_user(self):
        self._user_config()
        self._project_settings({"provider": "openai", "model": "gpt-project"})
        r = self.resolve()
        self.assertEqual((r.provider, r.model), ("openai", "gpt-project"))
        self.assertTrue(r.sources["provider"].startswith("project:"))
        self.assertEqual(r.base_url, "https://user.openai")  # base URL still from user config

    def test_env_overrides_project(self):
        self._user_config()
        self._project_settings({"provider": "openai", "model": "gpt-project"})
        r = self.resolve(env={"CLAWD_PROVIDER": "glm", "CLAWD_MODEL": "glm-env"})
        self.assertEqual((r.provider, r.model), ("glm", "glm-env"))
        self.assertEqual(r.sources["provider"], "env:CLAWD_PROVIDER")

    def test_increscent_prefix_wins_over_clawd(self):
        r = self.resolve(env={"CLAWD_PROVIDER": "glm", "INCRESCENT_PROVIDER": "openai"})
        self.assertEqual(r.provider, "openai")
        self.assertEqual(r.sources["provider"], "env:INCRESCENT_PROVIDER")

    def test_cli_overrides_everything(self):
        self._user_config()
        self._project_settings({"provider": "openai", "model": "gpt-project"})
        r = self.resolve(env={"CLAWD_PROVIDER": "glm", "CLAWD_MODEL": "glm-env"},
                         cli_provider="minimax", cli_model="cli-model", cli_base_url="https://cli")
        self.assertEqual((r.provider, r.model, r.base_url), ("minimax", "cli-model", "https://cli"))
        self.assertEqual({r.sources[k] for k in ("provider", "model", "base_url")}, {"cli"})

    def test_project_model_ignored_when_project_pins_other_provider(self):
        self._user_config()
        self._project_settings({"provider": "openai", "model": "gpt-project"})
        r = self.resolve(cli_provider="glm")
        self.assertEqual(r.model, "glm-user")

    def test_provider_env_api_key_beats_user_config(self):
        self._user_config()
        r = self.resolve(env={"ZHIPUAI_API_KEY": ENV_KEY})
        self.assertEqual(r.api_key, ENV_KEY)
        self.assertEqual(r.sources["api_key"], "env:ZHIPUAI_API_KEY")

    def test_env_base_url(self):
        r = self.resolve(env={"OPENAI_BASE_URL": "https://proxy.local/v1"}, cli_provider="openai")
        self.assertEqual(r.base_url, "https://proxy.local/v1")

    def test_project_cannot_set_secrets_or_endpoints(self):
        self._project_settings({"provider": "openai", "api_key": "stolen", "base_url": "https://evil.example"})
        r = self.resolve()
        self.assertEqual(r.provider, "openai")
        self.assertEqual(r.api_key, "")
        self.assertEqual(r.base_url, PROVIDER_INFO["openai"]["default_base_url"])
        self.assertEqual(len(r.warnings), 2)

    def test_unknown_provider_is_config_error(self):
        with self.assertRaises(ConfigError):
            self.resolve(cli_provider="nope")

    def test_invalid_project_json_is_a_warning_not_a_crash(self):
        path = self.project / ".clawd" / "settings.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        r = self.resolve()
        self.assertEqual(r.provider, "anthropic")
        self.assertTrue(r.warnings)

    def test_secret_never_in_repr_or_describe(self):
        r = self.resolve(env={"ANTHROPIC_API_KEY": ENV_KEY})
        self.assertNotIn(ENV_KEY, repr(r))
        self.assertNotIn(ENV_KEY, json.dumps(r.describe()))
        self.assertEqual(r.describe()["api_key"], "set")

    def test_create_provider_requires_key(self):
        with self.assertRaises(ConfigError):
            create_provider(self.resolve())
        provider = create_provider(self.resolve(env={"OPENAI_API_KEY": ENV_KEY}, cli_provider="openai", cli_model="m"))
        self.assertEqual(provider.model, "m")
        self.assertEqual(provider.message_format, "openai")

    def test_home_override(self):
        other = Path(self.tmp.name) / "custom-home"
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"CLAWD_HOME": str(other)}):
            self.assertEqual(get_user_dir(), other)
            self.assertEqual(get_config_path(), other / "config.json")


if __name__ == "__main__":
    unittest.main()
