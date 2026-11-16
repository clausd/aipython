# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Turn natural language / pseudo-Python into runnable Python with Claude."""

import os
import re
import sys

import anthropic

from . import context

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
DEFAULT_MODEL = os.getenv("AIPYTHON_MODEL", MODELS["haiku"])
MAX_TOOL_ROUNDS = 4

SYSTEM = """\
You are embedded in an IPython session. The user typed a cell that is not valid Python as-is: \
it is natural language, pseudo-Python, or Python with mistakes. Reply with the Python code that \
does what they meant. Your reply is executed directly in their session.

- Output only code. No markdown fences, no prose before or after.
- Use the variables, imports and column names that exist in the session; never invent data.
- If it is ordinary Python with a typo or a slip, fix it minimally.
- End with an expression when the user wants to see a result (IPython displays it), as in a notebook.
- Import what you need. Prefer the libraries already imported.
- Keep it short: what a skilled user would have typed.
- If the user asks a question that needs no code, answer in Python comments (`# ...`).
- If something you need is not in the summary below (exact column values, a nested structure), \
look it up with the tools first rather than guessing. Most requests need no tools."""

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```\s*$", re.DOTALL)
DIM, RESET = "\033[2m", "\033[0m"


def strip_fences(text):
    text = text.strip()
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text


def build_prompt(shell, request, history, last_error):
    parts = ["<namespace>", context.namespace_summary(shell), "</namespace>"]
    if history:
        parts.append("<recent_cells>")
        for entry in history:
            parts.append(f"In: {entry['raw']}")
            if entry.get("code") and entry["code"] != entry["raw"]:
                parts.append(f"Ran: {entry['code']}")
            if entry.get("error"):
                parts.append(f"Error: {entry['error']}")
        parts.append("</recent_cells>")
    if last_error:
        parts.append(f"<last_error>\n{last_error}\n</last_error>")
    parts.append(f"<cell>\n{request}\n</cell>")
    return "\n".join(parts)


class _Echo:
    """Writes streamed code to the terminal, dimmed and indented."""

    def __init__(self, out):
        self.out = out
        self.started = False

    def write(self, text):
        if not self.started:
            self.out.write(f"{DIM}✨ ")
            self.started = True
        self.out.write(text.replace("\n", "\n   "))
        self.out.flush()

    def note(self, text):
        self.close()
        self.started = False
        self.out.write(f"{DIM}   ⋯ {text}{RESET}\n")
        self.out.flush()

    def close(self):
        if self.started:
            self.out.write(f"{RESET}\n")
            self.out.flush()


class Translator:
    def __init__(self, model=DEFAULT_MODEL, client=None):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def translate(self, shell, request, history=(), last_error=None, out=sys.stdout):
        """Return Python code for `request`, streaming it to `out` as it arrives."""
        messages = [{"role": "user",
                     "content": build_prompt(shell, request, history, last_error)}]
        echo = _Echo(out)
        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM,
                    tools=context.tool_schemas(),
                    output_config={"effort": "low"},
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        echo.write(text)
                    message = stream.get_final_message()

                if message.stop_reason == "refusal":
                    raise RuntimeError("the model declined this request")
                if message.stop_reason != "tool_use":
                    return strip_fences("".join(
                        b.text for b in message.content if b.type == "text"))

                messages.append({"role": "assistant", "content": message.content})
                results = []
                for block in message.content:
                    if block.type != "tool_use":
                        continue
                    echo.note(f"{block.name}({', '.join(map(str, block.input.values()))})")
                    text, is_error = context.run_tool(shell, block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": text, "is_error": is_error})
                messages.append({"role": "user", "content": results})
            raise RuntimeError("gave up after too many lookups")
        finally:
            echo.close()
