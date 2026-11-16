# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from IPython.core.interactiveshell import InteractiveShell

from aipython import context
from aipython.llm import Translator
from aipython.shell import AIShell


class FakeTranslator:
    model = "fake"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def translate(self, shell, request, history=(), last_error=None, out=None):
        self.calls.append(SimpleNamespace(request=request, history=history, last_error=last_error))
        return self.replies[request]


@pytest.fixture
def ip():
    shell = InteractiveShell.instance()
    shell.user_ns["df"] = pd.DataFrame({"region": ["N", "S", "N"], "sales": [1, 2, 3]})
    yield shell
    shell.reset()


@pytest.fixture
def ai(ip):
    a = AIShell(ip, FakeTranslator({
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


# --- Translator tool loop against a fake Anthropic client ---------------------

class FakeStream:
    def __init__(self, message):
        self.message = message
        self.text_stream = [b.text for b in message.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def test_translator_runs_tools_then_returns_code(ip):
    tool_use = SimpleNamespace(type="tool_use", id="t1", name="column_values",
                               input={"variable": "df", "column": "region"})
    turns = iter([
        SimpleNamespace(stop_reason="tool_use", content=[tool_use]),
        SimpleNamespace(stop_reason="end_turn",
                        content=[SimpleNamespace(type="text", text="```python\ndf[df.region == 'N']\n```")]),
    ])
    sent = []

    def stream(**kwargs):
        sent.append(kwargs)
        return FakeStream(next(turns))

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    code = Translator(client=client).translate(ip, "only north", out=io.StringIO())
    assert code == "df[df.region == 'N']"
    tool_result = sent[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and "unique=2" in tool_result["content"]
