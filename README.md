# aipython

**IPython that understands what you meant.**

[![CI](https://github.com/clausd/aipython/actions/workflows/ci.yml/badge.svg)](https://github.com/clausd/aipython/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aipython)](https://pypi.org/project/aipython/)
[![License: MPL 2.0](https://img.shields.io/badge/license-MPL%202.0-blue)](LICENSE)

`aipython` is a plain IPython shell with one difference: when a cell isn't valid Python, an LLM
turns it into Python and runs it. Type code when you know the code, type what you want when you
don't, and mix the two freely. Real Python never touches the model and runs with no delay.

## A real session

An excerpt from an actual session: the lines after `In [n]:` are what was typed, the dimmed
`✨` lines are the code the model wrote, streamed in as it was generated and then run.
Outputs are left out.

```
In [1]: Find the most recent csv in ~/Downloads and load it as a pandas df
✨ import glob, os
   import pandas as pd

   latest_csv = max(glob.glob(os.path.expanduser("~/Downloads/*.csv")), key=os.path.getmtime)
   df = pd.read_csv(latest_csv)
   print(latest_csv)
   df.head()

In [2]: list the column names
✨ df.columns.tolist()

In [6]: df.date.min()

In [8]: convert the date column to actual dates
✨ df['date'] = pd.to_datetime(df['date'], errors='coerce')
   df['date'].dtype

In [9]: df.date.min()

In [13]: list all unique dates
✨ sorted(df['date'].dt.date.unique())
```

Cells 6 and 9 are ordinary Python, so they ran directly without a model call.

## Install

```bash
pipx install aipython            # or: uv tool install aipython / pip install aipython
export ANTHROPIC_API_KEY=...      # the default model is Claude Haiku
aipython
```

Or load it into any IPython session with `%load_ext aipython`.

If you work with pandas and friends, install aipython into the same environment as them
(`pip install aipython`) so the generated code can import them. A `pipx` install lives in its own
environment.

## How it works

Every cell goes through the same steps:

1. **IPython first.** IPython's own transforms run as usual, so `%magics`, `!shell` commands and
   `obj?` help all behave normally.
2. **Is it Python?** The cell is parsed and checked with pyflakes against your live variables.
   If it parses and every name it uses exists, it runs untouched. This check is local and instant.
3. **If not, ask the model.** Syntax errors, unknown names (`describe`, `prnt(x)`), and English
   that IPython would mistake for a magic or help request (`reset the index of df`,
   `what is x?`) go to the model. The model writes Python, which streams to your terminal and runs.

The model gets enough context to write code that fits your session:

- a summary of your variables: types, shapes, DataFrame columns and dtypes, imported modules,
  function signatures
- your last few cells, the code generated for them, and any errors, so "fix that" and
  "try again" work
- tools to look closer when the summary isn't enough: `inspect_variable` (dtypes, null counts,
  first rows), `column_values` (exact category spellings) and `search_namespace`

`In[n]` holds the code that actually ran. Up-arrow brings back what you typed.

## Token use

Each translation ends with one dim line showing what it cost:

```
In [2]: list the column names
✨ df.columns.tolist()
   ↳ 1,204 in · 14 out · $0.0001 · 0.8s
```

`%ai` shows the running total for the session, and it is printed when you exit:

```
aipython: 12 translations · 15,880 in / 702 out tokens · $0.0019
```

Costs come from litellm's price list. They are left out for models without a known price, and
for free local models; token counts are always shown.

## Session logs

Off by default. Turn on with `%ai log on`, or `log = true` in the config file, and the session
is written as a Python file: every cell in order, with what you typed as a comment above the
code the model wrote for it.

```python
# [1] Find the most recent csv in ~/Downloads and load it as a pandas df
#     ✨ anthropic/claude-haiku-5-5 · 1,204 in · 61 out · $0.0002 · 1.1s
import glob, os
import pandas as pd
...

# [6]
df.date.min()

# [11]
df.date.uniq()
# ! AttributeError: 'Series' object has no attribute 'uniq'
```

It reads as the Python your session became, and `ipython <file>` replays it. Logs go to
`~/.local/share/aipython/sessions/` (or `$XDG_DATA_HOME/aipython/sessions/`, or `log_dir` in the
config), one file per session, named by start time. `%ai log` shows the current file.

To see everything the model was given and did (the variable summary, recent cells, tool calls
and their results, token counts), turn on the debug log with `%ai debug on` or `debug = true`.
It writes one JSON line per translation to a `.jsonl` file next to the session log. The two
logs are independent.

Even with both off, IPython's own history keeps both versions of every cell: `%history` lists
the code that ran, `%history -r` what you typed.

## What is sent to the model

Only cells that aren't Python are sent. With each one goes the variable summary above,
including short previews of small values (up to 80 characters each), plus your recent cells.
More is sent only if the model asks for it through a tool: the first 5 rows of a DataFrame,
the first 10 values of a Series, the most common values of one column, or a longer preview
(up to 1500 characters) of another object. If that is too much for your data, use a
local model (see below) or `%ai off`.

## Commands

```
%ai                 status and tokens used this session
%ai on | off        turn automatic translation on or off
%ai model <name>    haiku (default), sonnet, opus, or any litellm model
%ai config          settings, config file location, which key is in use
%ai context         the variable summary the model sees
%ai last            the last generated code
%ai log [on | off]  write this session as a .py file (off by default); show where
%ai debug on | off  also log the full exchange with the model (.jsonl)
%ai <anything>      send this to the model even if it looks like Python
```

## Models

Models are reached through [litellm](https://docs.litellm.ai/docs/providers), so any provider it
supports works. Choose one with `%ai model <name>` or the `AIPYTHON_MODEL` environment variable,
and set that provider's usual API key:

```bash
AIPYTHON_MODEL=anthropic/claude-sonnet-5-5   # ANTHROPIC_API_KEY
AIPYTHON_MODEL=openai/gpt-5.4                # OPENAI_API_KEY
AIPYTHON_MODEL=ollama_chat/qwen2.5-coder     # local Ollama, no key
```

For any OpenAI-compatible server (LM Studio, vLLM, llama.cpp), use `openai/<model>` and set
`AIPYTHON_API_BASE`, e.g. `http://localhost:1234/v1`.

Models without tool calling still work, with only the variable summary to go on. Small local
models are noticeably worse at this than hosted ones.

## Configuration

Without a config file, aipython uses your environment: the provider's usual API key variable and
the default model. To keep settings and keys in one place, create
`~/.config/aipython/config.toml`:

```toml
model = "anthropic/claude-sonnet-5-5"
# api_base = "http://localhost:1234/v1"
# log = true               # write each session as a .py file (default false)
# debug = true             # also log the full exchange with the model (default false)
# log_dir = "~/notes/aipython"

[keys]                     # one per provider, used when that provider's model is selected
anthropic = "sk-ant-..."
openai = "sk-..."
```

- Keys are passed directly to the model call and never put into the environment, so code you
  run in the session can't read them.
- A provider without a key in the file falls back to its environment variable.
- `AIPYTHON_MODEL` and `AIPYTHON_API_BASE` override the file; `AIPYTHON_CONFIG` points to a
  different file, and `$XDG_CONFIG_HOME` is respected.
- aipython warns at startup if the file is readable by other users (`chmod 600` it).

## Limitations

- **Valid Python is trusted as Python.** `df.date.uniq()` parses and `df` exists, so it runs and
  fails with an `AttributeError` instead of being fixed. Typing `fix that` afterwards works,
  because the error is passed to the model.
- **Generated code runs without asking.** It streams to the screen as it runs, so you see what
  happened, but there is no confirmation step. Use `%ai off` when that matters.
- A line of English ending in `:` makes IPython wait for more input, as it would for a `for`
  loop. Press Enter again.

## Development

```bash
uv run --extra test pytest
```

Releases go to PyPI through GitHub Actions with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). To release, bump `__version__`
in `aipython/__init__.py`, push, and run `gh release create vX.Y.Z --generate-notes`.

One-time setup on PyPI: add a pending publisher for project `aipython`, owner `clausd`,
repository `aipython`, workflow `publish.yml`, environment `pypi`.

## License

[Mozilla Public License 2.0](LICENSE).
