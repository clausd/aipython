# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Session logs.

With logging on (it is off by default), a session is written as a .py file: each cell's code
in order, with what the user typed as a comment above any code the model wrote. It reads as
the Python the session became, and `ipython <file>` replays it. With debug on, a .jsonl file
next to it holds the full exchange with the model for each translation.
"""

import datetime
import json


def _comment(text):
    return "\n".join(f"# {line}".rstrip() for line in text.splitlines())


class SessionLog:
    def __init__(self, directory, enabled=False, debug=False):
        self.directory = directory
        self.enabled = enabled
        self.debug = debug
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = directory / f"{stamp}.py"
        self.debug_path = directory / f"{stamp}.jsonl"

    def _append(self, path, text):
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if new and path == self.path:
                f.write(f"# aipython session {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
                        f"# replay with: ipython {path.name}\n\n")
            f.write(text)

    def cell(self, number, raw, code, translation=None, error=None):
        """Record one executed cell: `code` is what ran, `raw` what was typed."""
        if not self.enabled:
            return
        raw, code = raw.strip(), (code or "").strip()
        if translation:
            header = (f"# [{number}] " + _comment(raw)[2:] + "\n"
                      f"#     ✨ {translation.model} · {translation.usage_text()}")
        elif code != raw:  # IPython syntax such as %magic or !shell
            header = f"# [{number}] " + _comment(raw)[2:]
        else:
            header = f"# [{number}]"
        text = f"{header}\n{code}\n"
        if error:
            text += f"# ! {error.splitlines()[0][:200]}\n"
        self._append(self.path, text + "\n")

    def exchange(self, number, request, translation):
        """With debug on, record the full model exchange for one translation."""
        if not self.debug:
            return
        entry = {
            "cell": number,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "request": request,
            "model": translation.model,
            "input_tokens": translation.input_tokens,
            "output_tokens": translation.output_tokens,
            "cost": translation.cost,
            "seconds": round(translation.seconds, 2),
            "code": translation.code,
            "messages": translation.messages,
        }
        self._append(self.debug_path, json.dumps(entry, default=str) + "\n")
