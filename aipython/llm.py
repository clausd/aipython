# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Turn natural language / pseudo-Python into runnable Python with any LLM.

Models are reached through litellm, so any model string it understands works:
"anthropic/claude-haiku-5-5", "openai/gpt-5.4", "ollama_chat/qwen2.5-coder", ...
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field

from . import config, context

MODELS = {
    "haiku": "anthropic/claude-haiku-5-5",
    "sonnet": "anthropic/claude-sonnet-5-5",
    "opus": "anthropic/claude-opus-5-5",
}
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

    def usage(self, translation):
        self.close()
        self.started = False
        self.out.write(f"{DIM}   ↳ {translation.usage_text()}{RESET}\n")
        self.out.flush()


def format_cost(cost):
    if cost >= 1:
        return f"${cost:.2f}"
    return f"${cost:.4f}" if cost >= 0.0001 else "<$0.0001"


@dataclass
class Translation:
    """Generated code plus what it cost. `messages` is the full exchange, for debug logs."""
    code: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    seconds: float = 0.0
    messages: list = field(default_factory=list)

    def usage_text(self):
        parts = [f"{self.input_tokens:,} in · {self.output_tokens:,} out"]
        if self.cost:
            parts.append(format_cost(self.cost))
        parts.append(f"{self.seconds:.1f}s")
        return " · ".join(parts)


def _cost(response, model):
    try:
        return _litellm().completion_cost(completion_response=response, model=model)
    except Exception:
        return None


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


def provider(model):
    try:
        return _litellm().get_llm_provider(model)[1]
    except Exception:
        return None


class Translator:
    def __init__(self, settings=None, completion=None):
        self.settings = settings or config.Settings()
        self.model = self.settings.model
        self._completion = completion

    def completion(self, **kwargs):
        if self.settings.api_base:
            kwargs["api_base"] = self.settings.api_base
        key = self.settings.keys.get(provider(self.model)) if self.settings.keys else None
        if key:
            kwargs["api_key"] = key
        return (self._completion or _litellm().completion)(**kwargs)

    def translate(self, shell, request, history=(), last_error=None, out=sys.stdout):
        """Return a Translation for `request`, streaming the code to `out` as it arrives."""
        result = Translation(code="", model=self.model)
        started = time.monotonic()
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
                    stream_options={"include_usage": True},
                ):
                    chunks.append(chunk)
                    text = chunk.choices[0].delta.content if chunk.choices else None
                    if text:
                        echo.write(text)
                response = _litellm().stream_chunk_builder(chunks, messages=messages)
                choice = response.choices[0]
                message = choice.message
                if response.usage:
                    result.input_tokens += response.usage.prompt_tokens or 0
                    result.output_tokens += response.usage.completion_tokens or 0
                cost = _cost(response, self.model)
                if cost is not None:
                    result.cost = (result.cost or 0) + cost

                if choice.finish_reason == "content_filter":
                    raise RuntimeError("the model declined this request")
                if not message.tool_calls:
                    messages.append({"role": "assistant", "content": message.content})
                    result.code = strip_fences(message.content or "")
                    result.seconds = time.monotonic() - started
                    result.messages = messages
                    echo.usage(result)
                    return result

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
