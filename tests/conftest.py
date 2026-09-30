"""Test-suite isolation.

Every test runs with:

* ``Path.home()`` pointing at a fresh temporary directory, so config files,
  sessions, prompt history and user skills never touch the developer's real
  ``~/.clawd``. Tests that patch ``Path.home`` themselves still work: their
  inner patch takes precedence.
* Clawd/Increscent and provider credential environment variables removed, so
  a developer's real API keys or overrides cannot change test outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_PREFIXES = ("CLAWD_", "INCRESCENT_")
_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ZHIPUAI_API_KEY", "GLM_API_KEY", "ZAI_API_KEY", "GLM_BASE_URL",
    "MINIMAX_API_KEY", "MINIMAX_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_user_environment(tmp_path_factory, monkeypatch):
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES) or name in _PROVIDER_ENV:
            monkeypatch.delenv(name, raising=False)
    yield fake_home
