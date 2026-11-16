# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Settings from ~/.config/aipython/config.toml, overridden by environment variables.

    model = "anthropic/claude-haiku-5-5"
    api_base = "http://localhost:1234/v1"   # optional, for OpenAI-compatible servers

    [keys]                                  # per provider; else the provider's env var
    anthropic = "sk-ant-..."
    openai = "sk-..."
"""

import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_MODEL = "anthropic/claude-haiku-5-5"


def config_path():
    if os.getenv("AIPYTHON_CONFIG"):
        return Path(os.environ["AIPYTHON_CONFIG"]).expanduser()
    base = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "aipython" / "config.toml"


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    api_base: str | None = None
    keys: dict = field(default_factory=dict)
    path: Path | None = None
    warnings: list = field(default_factory=list)


def load(path=None):
    path = path or config_path()
    settings = Settings(path=path)
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            settings.warnings.append(f"could not read {path}: {e}")
            data = {}
        settings.model = data.get("model", settings.model)
        settings.api_base = data.get("api_base")
        settings.keys = dict(data.get("keys", {}))
        if settings.keys and os.name == "posix" and path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            settings.warnings.append(
                f"{path} holds API keys but is readable by others; run: chmod 600 {path}")
    settings.model = os.getenv("AIPYTHON_MODEL", settings.model)
    settings.api_base = os.getenv("AIPYTHON_API_BASE", settings.api_base)
    return settings


def mask(key):
    return key[:7] + "…" + key[-4:] if len(key) > 16 else "…"
