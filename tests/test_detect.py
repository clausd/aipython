# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from IPython.core.interactiveshell import InteractiveShell

from aipython.detect import ai_magic_text, classify


@pytest.fixture(scope="module")
def ip():
    return InteractiveShell.instance()


def check(ip, raw, ns=()):
    code = ip.transform_cell(raw)
    known = set(ns) | set(ip.user_ns) | set(ip.magics_manager.magics["line"])
    return classify(raw, code, known)[0]


@pytest.mark.parametrize("raw", [
    "x = 1", "print(len([1, 2]))", "df.head()", "for i in range(3):\n    print(i)",
    "%timeit 1 + 1", "!ls", "df?", "cd /tmp", "ls -la", "pwd", "import os; os.getcwd()",
])
def test_python_runs_untouched(ip, raw):
    assert check(ip, raw, ns={"df"})


@pytest.mark.parametrize("raw", [
    "show me the top 5 rows of df", "what's the mean of x?", "sort df by price descending",
    "describe", "prnt(x)", "df.groupby(region).sales.sum()", "reset the index of df",
    "history of my commands",
])
def test_non_python_goes_to_ai(ip, raw):
    assert not check(ip, raw, ns={"df", "x"})


def test_ai_magic_text(ip):
    assert ai_magic_text(ip.transform_cell("%ai sort df")) == "sort df"
    assert ai_magic_text(ip.transform_cell("%time 1")) is None
