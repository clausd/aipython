# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io

from IPython.core.interactiveshell import InteractiveShell

from aipython import config
from aipython.llm import Translator

CONFIG = """
model = "openai/gpt-5.4"
api_base = "http://localhost:1234/v1"

[keys]
anthropic = "sk-ant-from-config"
openai = "sk-openai-from-config"
"""


def write(tmp_path, text, mode=0o600):
    path = tmp_path / "config.toml"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AIPYTHON_MODEL", raising=False)
    settings = config.load(tmp_path / "missing.toml")
    assert settings.model == config.DEFAULT_MODEL and settings.keys == {}


def test_file_and_env_override(tmp_path, monkeypatch):
    path = write(tmp_path, CONFIG)
    monkeypatch.delenv("AIPYTHON_MODEL", raising=False)
    assert config.load(path).model == "openai/gpt-5.4"
    monkeypatch.setenv("AIPYTHON_MODEL", "ollama_chat/qwen2.5-coder")
    settings = config.load(path)
    assert settings.model == "ollama_chat/qwen2.5-coder"
    assert settings.api_base == "http://localhost:1234/v1"
    assert settings.warnings == []


def test_readable_key_file_warns(tmp_path):
    assert "chmod 600" in config.load(write(tmp_path, CONFIG, 0o644)).warnings[0]


def test_bad_toml_warns(tmp_path):
    assert "could not read" in config.load(write(tmp_path, "model = ")).warnings[0]


def test_key_follows_the_models_provider(tmp_path, monkeypatch):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    monkeypatch.setattr("aipython.llm.supports_tools", lambda model: False)
    sent = []

    def completion(**kwargs):
        sent.append(kwargs)
        return iter([ModelResponseStream(choices=[StreamingChoices(
            index=0, finish_reason="stop", delta=Delta(content="1"))])])

    translator = Translator(config.load(write(tmp_path, CONFIG)), completion=completion)
    ip = InteractiveShell.instance()
    for model, key in [("anthropic/claude-haiku-5-5", "sk-ant-from-config"),
                       ("claude-sonnet-5-5", "sk-ant-from-config"),
                       ("openai/gpt-5.4", "sk-openai-from-config"),
                       ("ollama_chat/qwen2.5-coder", None)]:
        translator.model = model
        translator.translate(ip, "x", out=io.StringIO())
        assert sent[-1].get("api_key") == key, model
    assert sent[-1]["api_base"] == "http://localhost:1234/v1"
