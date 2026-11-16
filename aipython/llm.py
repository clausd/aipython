# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Turn natural language / pseudo-Python into runnable Python with any LLM.

Models are reached through litellm, so any model string it understands works:
"anthropic/claude-haiku-5-5", "openai/gpt-5.4", "ollama_chat/qwen2.5-coder", ...
"""

import json
import os
import re
import sys

from . import context

MODELS = {
    "haiku": "anthropic/claude-haiku-5-5",
    "sonnet": "anthropic/claude-sonnet-5-5",
    "opus": "anthropic/claude-opus-5-5",
}
DEFAULT_MODEL = os.getenv("AIPYTHON_MODEL", MODELS["haiku"])
API_BASE = os.getenv("AIPYTHON_API_BASE")  # e.g. an OpenAI-compatible local server
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


def _litellm():
    """litellm takes over a second to import, so it is loaded on first use (or in the
    background at startup, see preload)."""
    import litellm
    litellm.suppress_debug_info = True
    return litellm


def preload():
    import threading
    threading.Thread(target=_litellm, daemon=True).start()


def supports_tools(model):
    try:
        return "tools" in (_litellm().get_supported_openai_params(model) or [])
    except Exception:
        return False


class Translator:
    def __init__(self, model=DEFAULT_MODEL, completion=None):
        self.model = model
        self._completion = completion

    def completion(self, **kwargs):
        if self._completion is not None:
            return self._completion(**kwargs)
        if API_BASE:
            kwargs["api_base"] = API_BASE
        return _litellm().completion(**kwargs)

    def translate(self, shell, request, history=(), last_error=None, out=sys.stdout):
        """Return Python code for `request`, streaming it to `out` as it arrives."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(shell, request, history, last_error)},
        ]
        tools = context.tool_schemas() if supports_tools(self.model) else None
        echo = _Echo(out)
        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                chunks = []
                for chunk in self.completion(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    max_tokens=4096,
                    reasoning_effort="low",
                    drop_params=True,
                    stream=True,
                ):
                    chunks.append(chunk)
                    text = chunk.choices[0].delta.content if chunk.choices else None
                    if text:
                        echo.write(text)
                response = _litellm().stream_chunk_builder(chunks, messages=messages)
                choice = response.choices[0]
                message = choice.message

                if choice.finish_reason == "content_filter":
                    raise RuntimeError("the model declined this request")
                if not message.tool_calls:
                    return strip_fences(message.content or "")

                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in message.tool_calls],
                })
                for call in message.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        text, is_error = f"invalid JSON arguments: {e}", True
                    else:
                        echo.note(f"{name}({', '.join(map(str, args.values()))})")
                        text, is_error = context.run_tool(shell, name, args)
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": f"Error: {text}" if is_error else text})
            raise RuntimeError("gave up after too many lookups")
        finally:
            echo.close()
