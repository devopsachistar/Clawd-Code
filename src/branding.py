"""Product identity in one place.

The runtime reads names, directories and environment-variable prefixes from
here instead of hard-coding them. A future rename (Clawd Code -> Increscent
Code) then changes these constants instead of the whole codebase.

Environment variables are looked up with every prefix in ``ENV_PREFIXES`` in
order, so ``INCRESCENT_MODEL`` takes precedence over ``CLAWD_MODEL``.
"""

from __future__ import annotations

import os
from typing import Mapping

APP_NAME = "Clawd Codex"
CLI_NAMES: tuple[str, ...] = ("clawd", "increscent")
USER_DIR_NAME = ".clawd"          # ~/.clawd  (user config, sessions, history)
PROJECT_DIR_NAME = ".clawd"       # <project>/.clawd  (project settings, skills)
PROJECT_SETTINGS_FILE = "settings.json"
ENV_PREFIXES: tuple[str, ...] = ("INCRESCENT", "CLAWD")


def env_lookup(suffix: str, env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """Return ``(value, variable_name)`` for the first set ``<PREFIX>_<suffix>``."""
    source = os.environ if env is None else env
    for prefix in ENV_PREFIXES:
        name = f"{prefix}_{suffix}"
        value = source.get(name)
        if value is not None and value.strip() != "":
            return value.strip(), name
    return None, None
