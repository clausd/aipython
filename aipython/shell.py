# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Hook the AI into IPython's input pipeline.

Every cell first goes through IPython's own transforms (magics, !shell, x?). If the result
is Python whose names all resolve, it runs untouched. Otherwise the cell is sent to the model
and the code it returns runs in its place. `In[n]` holds the code that ran; the prompt
history (up-arrow) keeps what the user typed.
"""

import collections
import sys

from IPython.core.magic import Magics, line_magic, magics_class

from . import config, context, detect
from .llm import MODELS, Translator, provider

HISTORY_CELLS = 8
HELP = """\
%ai                 show status
%ai on | off        turn automatic translation on or off
%ai model <name>    haiku, sonnet, opus, or any litellm model, e.g. openai/gpt-5.4,
                    ollama_chat/qwen2.5-coder
%ai config          show settings and where they come from
%ai context         show the namespace summary the model sees
%ai last            show the last generated code
%ai <anything>      translate even if it already looks like Python"""


class AIShell:
    def __init__(self, shell, translator=None):
        self.shell = shell
        self.translator = translator or Translator(config.load())
        self.enabled = True
        self.history = collections.deque(maxlen=HISTORY_CELLS)
        self.last_error = None
        self.last_code = None
        self._raw = ""
        self._generated = None

    # -- IPython hooks -------------------------------------------------------

    def capture_raw(self, lines):
        """Cleanup transformer: remember the cell exactly as typed."""
        self._raw = "".join(lines)
        self._generated = None
        return lines

    def transform(self, lines):
        """Post transformer: swap non-Python cells for generated code."""
        code = "".join(lines)
        forced = (detect.ai_magic_text(code) or "").split()
        if forced and forced[0] not in Magic.SUBCOMMANDS:
            request = " ".join(forced)
        elif self.enabled and not detect.classify(self._raw, code, self.known_names())[0]:
            request = self._raw.strip()
        else:
            return lines

        try:
            generated = self.translator.translate(
                self.shell, request, list(self.history), self.last_error)
        except KeyboardInterrupt:
            print("(cancelled)")
            return []
        except Exception as e:
            print(f"🤖 AI unavailable ({type(e).__name__}: {e}); running the cell as typed",
                  file=sys.stderr)
            return lines
        self._generated = self.last_code = generated
        return generated.splitlines(keepends=True) + ["\n"]

    def post_run_cell(self, result):
        raw = result.info.raw_cell if result.info else self._raw
        error = result.error_before_exec or result.error_in_exec
        self.last_error = f"{type(error).__name__}: {error}" if error is not None else None
        if raw.strip():
            self.history.append({"raw": raw.strip(), "code": self._generated,
                                 "error": self.last_error})

    def known_names(self):
        magics = self.shell.magics_manager.magics
        return set(self.shell.user_ns) | set(magics["line"]) | set(magics["cell"])

    # -- install -------------------------------------------------------------

    def install(self):
        self.shell.input_transformers_cleanup.insert(0, self.capture_raw)
        self.shell.input_transformers_post.append(self.transform)
        self.shell.events.register("post_run_cell", self.post_run_cell)
        self.shell.register_magics(Magic(self.shell, self))

    def uninstall(self):
        self.shell.input_transformers_cleanup.remove(self.capture_raw)
        self.shell.input_transformers_post.remove(self.transform)
        self.shell.events.unregister("post_run_cell", self.post_run_cell)


@magics_class
class Magic(Magics):
    SUBCOMMANDS = {"on", "off", "model", "config", "context", "last", "help"}

    def __init__(self, shell, ai):
        super().__init__(shell)
        self.ai = ai

    @line_magic
    def ai(self, line):
        """Control aipython. `%ai help` lists the subcommands."""
        cmd, _, arg = line.strip().partition(" ")
        if cmd in ("on", "off"):
            self.ai.enabled = cmd == "on"
        elif cmd == "model" and arg:
            self.ai.translator.model = MODELS.get(arg.strip(), arg.strip())
        elif cmd == "config":
            self.show_config()
            return
        elif cmd == "context":
            print(context.namespace_summary(self.shell))
            return
        elif cmd == "last":
            print(self.ai.last_code or "(nothing generated yet)")
            return
        elif cmd == "help":
            print(HELP)
            return
        state = "on" if self.ai.enabled else "off"
        print(f"aipython {state}, model {self.ai.translator.model}")

    def show_config(self):
        settings = self.ai.translator.settings
        found = "" if settings.path.exists() else " (not found)"
        print(f"config file: {settings.path}{found}")
        print(f"model:       {self.ai.translator.model}")
        if settings.api_base:
            print(f"api_base:    {settings.api_base}")
        current = provider(self.ai.translator.model)
        for name, key in settings.keys.items():
            used = "  <- in use" if name == current else ""
            print(f"key {name}: {config.mask(key)}{used}")
        if current not in settings.keys:
            print(f"key {current}: from the environment, if set")
