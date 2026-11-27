# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from IPython.core.interactiveshell import InteractiveShell

from aipython import config, context
from aipython.llm import Translation, Translator
from aipython.log import SessionLog
from aipython.shell import AIShell


class FakeTranslator:
    model = "fake"
    settings = config.Settings()

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def translate(self, shell, request, history=(), last_error=None, out=None):
        self.calls.append(SimpleNamespace(request=request, history=history, last_error=last_error))
        return Translation(code=self.replies[request], model="fake", input_tokens=1000,
                           output_tokens=20, cost=0.0002, seconds=0.5,
                           messages=[{"role": "user", "content": request}])


@pytest.fixture
def ip():
    shell = InteractiveShell.instance()
    shell.user_ns["df"] = pd.DataFrame({"region": ["N", "S", "N"], "sales": [1, 2, 3]})
    yield shell
    shell.reset()


@pytest.fixture
def ai(ip, tmp_path):
    a = AIShell(ip, log=SessionLog(tmp_path), translator=FakeTranslator({
        "total sales by region": "totals = df.groupby('region').sales.sum()\ntotals",
        "sort df": "df = df.sort_values('sales')",
        "fix that": "y = 1",
    }))
    a.install()
    yield a
    a.uninstall()


def test_python_is_not_sent(ip, ai):
    ip.run_cell("z = df.sales.sum()")
    assert ip.user_ns["z"] == 6
    assert ai.translator.calls == []


def test_prose_is_translated_and_run(ip, ai):
    result = ip.run_cell("total sales by region", store_history=True)
    assert result.success
    assert ip.user_ns["totals"].to_dict() == {"N": 4, "S": 2}
    assert "groupby" in ip.user_ns["In"][-1]


def test_forced_translation_and_subcommands(ip, ai):
    ip.run_cell("%ai sort df")
    assert list(ip.user_ns["df"].sales) == [1, 2, 3]
    ip.run_cell("%ai off")
    assert not ai.enabled
    ip.run_cell("%ai sort df")  # %ai <text> works while off
    assert len(ai.translator.calls) == 2


def test_errors_and_history_reach_the_model(ip, ai):
    ip.run_cell("1 / 0")
    ip.run_cell("fix that")
    call = ai.translator.calls[-1]
    assert call.last_error.startswith("ZeroDivisionError")
    assert call.history[-1]["raw"] == "1 / 0"


def test_ai_failure_falls_back_to_python(ip, ai):
    ai.translator.replies = {}
    result = ip.run_cell("this is not python")
    assert not result.success  # KeyError in translator -> cell runs as typed -> SyntaxError


def test_tools(ip):
    assert "'sales': int64" in context.namespace_summary(ip)
    assert "N" in context.column_values(ip, "df", "region")
    assert "column df['sales']" in context.search_namespace(ip, "sal")
    text, is_error = context.run_tool(ip, "inspect_variable", {"name": "nope"})
    assert is_error


def test_totals_and_exit_summary(ip, ai, capsys):
    ip.run_cell("total sales by region")
    ip.run_cell("sort df")
    ip.run_cell("%ai")
    out = capsys.readouterr().out
    assert "this session: 2 translations · 2,000 in / 40 out tokens · $0.0004" in out
    ai.print_totals()
    assert capsys.readouterr().out == (
        "aipython: 2 translations · 2,000 in / 40 out tokens · $0.0004\n")


def test_session_log_is_off_by_default(ip, ai):
    ip.run_cell("total sales by region")
    assert not ai.log.path.exists() and not ai.log.debug_path.exists()


def test_session_log_reads_as_python(ip, ai):
    ip.run_cell("%ai log on")
    ip.run_cell("total sales by region")
    ip.run_cell("z = 1")
    ip.run_cell("%pwd")
    ip.run_cell("1 / 0")
    text = ai.log.path.read_text()
    assert "%ai log" not in text
    assert ("] total sales by region\n#     ✨ fake · 1,000 in · 20 out · $0.0002 · 0.5s\n"
            "totals = df.groupby('region').sales.sum()\ntotals\n") in text
    assert "]\nz = 1\n" in text
    assert "] %pwd\nget_ipython().run_line_magic('pwd', '')\n" in text
    assert "1 / 0\n# ! ZeroDivisionError: division by zero\n" in text
    compile(text, "log", "exec")  # the log itself is valid Python


def test_debug_log_has_the_full_exchange(ip, ai):
    import json
    ip.run_cell("%ai debug on")
    ip.run_cell("total sales by region")
    entry = json.loads(ai.log.debug_path.read_text())
    assert entry["request"] == "total sales by region"
    assert entry["messages"][0]["content"] == "total sales by region"
    assert entry["input_tokens"] == 1000 and "groupby" in entry["code"]
    assert not ai.log.path.exists()  # debug and the .py log are independent


def test_cost_formatting():
    from aipython.llm import format_cost
    assert [format_cost(c) for c in (0.00003, 0.0123, 2.5)] == ["<$0.0001", "$0.0123", "$2.50"]


def test_log_settings_from_config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(f'log = true\ndebug = true\nlog_dir = "{tmp_path}/logs"\n')
    settings = config.load(path)
    assert settings.log and settings.debug and settings.log_dir == tmp_path / "logs"
    assert not config.Settings().log


# --- Translator tool loop with litellm stream chunks -------------------------

def _chunk(content=None, tool_call=None, finish=None):
    from litellm.types.utils import (ChatCompletionDeltaToolCall, Delta, Function,
                                     ModelResponseStream, StreamingChoices)
    calls = None
    if tool_call:
        calls = [ChatCompletionDeltaToolCall(id=tool_call[0], index=0, type="function",
                                             function=Function(name=tool_call[1],
                                                               arguments=tool_call[2]))]
    return ModelResponseStream(choices=[StreamingChoices(
        index=0, finish_reason=finish, delta=Delta(content=content, tool_calls=calls))])


def test_translator_runs_tools_then_returns_code(ip, monkeypatch):
    monkeypatch.setattr("aipython.llm.supports_tools", lambda model: True)
    turns = iter([
        [_chunk(tool_call=("t1", "column_values", '{"variable": "df", "column": "region"}'),
                finish="tool_calls")],
        [_chunk("```python\ndf[df.region"), _chunk(" == 'N']\n```", finish="stop")],
    ])
    sent = []

    def completion(**kwargs):
        sent.append({**kwargs, "messages": list(kwargs["messages"])})
        return iter(next(turns))

    out = io.StringIO()
    result = Translator(completion=completion).translate(ip, "only north", out=out)
    assert result.code == "df[df.region == 'N']"
    assert result.messages[-1] == {"role": "assistant",
                                   "content": "```python\ndf[df.region == 'N']\n```"}
    assert sent[0]["tools"][0]["type"] == "function"
    tool_message = sent[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "t1"
    assert "unique=2" in tool_message["content"]
    assert "column_values(df, region)" in out.getvalue()


def test_models_without_tools_get_none(ip, monkeypatch):
    monkeypatch.setattr("aipython.llm.supports_tools", lambda model: False)
    sent = []

    def completion(**kwargs):
        sent.append(kwargs)
        return iter([_chunk("df.head()", finish="stop")])

    result = Translator(completion=completion).translate(ip, "peek", out=io.StringIO())
    assert result.code == "df.head()"
    assert sent[0]["tools"] is None


def test_missing_module_hint(ip, ai, capsys):
    ip.run_cell("import surely_not_a_real_module_xyz")
    assert "%ai install surely_not_a_real_module_xyz" in capsys.readouterr().err


def test_ai_install(ip, ai, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("aipython.install.install", lambda pkgs: calls.append(pkgs) or True)
    ip.run_cell("%ai install polars numpy")
    assert calls == [["polars", "numpy"]]
    assert "installed" in capsys.readouterr().out
