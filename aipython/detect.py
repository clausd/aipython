# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Decide whether a cell is real Python or something the AI should interpret."""

import ast
import builtins
import re

import pyflakes.checker
import pyflakes.messages

_MAGIC_CALL = re.compile(r"^get_ipython\(\)\.run_line_magic\(")
_BARE_WORD = re.compile(r"^[A-Za-z][A-Za-z']*$")
HELP_MAGICS = {"pinfo", "pinfo2", "psearch"}


def undefined_names(code, known):
    """Names that `code` reads but that are defined neither in it nor in `known`."""
    tree = ast.parse(code)
    checker = pyflakes.checker.Checker(tree, builtins=set(known) | set(dir(builtins)))
    return sorted({
        m.message_args[0] for m in checker.messages
        if isinstance(m, pyflakes.messages.UndefinedName)
    })


def _magic_call(code):
    """(name, arg) if `code` is a single get_ipython().run_line_magic(...) call."""
    if not _MAGIC_CALL.match(code.strip()):
        return None
    try:
        call = ast.parse(code.strip(), mode="eval").body
        name, arg = (a.value for a in call.args[:2])
    except (SyntaxError, ValueError, AttributeError):
        return None
    return name, arg


def looks_like_prose_magic(raw, code):
    """True when IPython turned prose into a magic call.

    Two cases: help syntax ("what is x?" becomes `x?`, i.e. %pinfo x) and automagic
    ("reset the index of df" becomes %reset). Real help requests (`x?`, `?x`, `x??`),
    explicit `%magic` lines and argument-like lines ("ls -la", "cd data") are left alone.
    """
    if raw.lstrip().startswith("%"):
        return False
    call = _magic_call(code)
    if not call:
        return False
    if call[0] in HELP_MAGICS:
        return raw.strip().strip("?").strip() != call[1].strip()
    words = call[1].split()
    return len(words) >= 2 and all(_BARE_WORD.match(w) for w in words)


def classify(raw, code, namespace):
    """Return (is_python, reason) for a cell.

    `raw` is what the user typed, `code` is the cell after IPython's own transforms
    (magics, shell escapes, help syntax) have been applied.
    """
    if not code.strip():
        return True, "empty"
    try:
        missing = undefined_names(code, namespace)
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    if missing:
        return False, "undefined: " + ", ".join(missing)
    if looks_like_prose_magic(raw, code):
        return False, "prose turned into a magic"
    return True, "ok"


def ai_magic_text(code):
    """The text of a `%ai <text>` cell, or None if the cell is something else."""
    call = _magic_call(code)
    return call[1] if call and call[0] == "ai" else None
