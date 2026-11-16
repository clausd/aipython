# aipython — IPython that understands what you meant

`aipython` is IPython, except cells that aren't valid Python get turned into Python by an LLM
and run. Real Python never touches the model.

```
In [1]: df = pd.read_csv("sales.csv")
In [2]: total sales per region, biggest first
✨ df.groupby('region')['sales'].sum().sort_values(ascending=False)
Out[2]: ...
In [3]: prnt(df.shape)
✨ print(df.shape)
```

## How a cell is handled

1. IPython's own transforms run first, so `%magics`, `!shell`, `obj?` work as usual.
2. The result is parsed and checked with pyflakes against your live namespace.
   It parses and every name resolves → runs untouched, no latency.
3. Otherwise (syntax error, undefined names like `describe` or `prnt`, or prose after an
   automagic word like `reset the index of df`) the cell goes to the model with a summary of
   your namespace (variables, DataFrame columns and dtypes), recent cells, and the last error.
   The generated code streams to the terminal and runs. The model can call small tools to look
   closer (`inspect_variable`, `column_values`, `search_namespace`).

`In[n]` holds the code that ran; up-arrow history keeps what you typed.

## Install

```bash
pipx install aipython          # or: uv tool install aipython / pip install aipython
export ANTHROPIC_API_KEY=...    # for the default model, Claude Haiku
aipython                       # or, in any IPython: %load_ext aipython
```

Tip: install it next to your data libraries (`pip install aipython` in the same environment as
pandas etc.) so generated code can import them.

## Controls

```
%ai                 status
%ai on | off        automatic translation on/off
%ai model <name>    haiku (default), sonnet, opus, or any litellm model
%ai config          settings, config file location, which key is in use
%ai context         the namespace summary the model sees
%ai last            last generated code
%ai <anything>      translate even if it already looks like Python
```

## Models

Models are reached through [litellm](https://docs.litellm.ai/docs/providers), so any provider
it supports works. Pick one with `%ai model <name>` or the `AIPYTHON_MODEL` environment variable,
and set that provider's usual API key variable:

```bash
AIPYTHON_MODEL=anthropic/claude-sonnet-5-5   # ANTHROPIC_API_KEY
AIPYTHON_MODEL=openai/gpt-5.4                # OPENAI_API_KEY
AIPYTHON_MODEL=ollama_chat/qwen2.5-coder     # local Ollama, no key
```

For any OpenAI-compatible server (LM Studio, vLLM, llama.cpp), use `openai/<model>` and set
`AIPYTHON_API_BASE`, e.g. `http://localhost:1234/v1`.

Models without tool calling still work; they just can't look deeper than the namespace summary.
Small local models are noticeably worse at this than hosted ones.

## Config file

Settings and keys can live in `~/.config/aipython/config.toml` (or `$XDG_CONFIG_HOME/aipython/`,
or wherever `AIPYTHON_CONFIG` points):

```toml
model = "anthropic/claude-sonnet-5-5"
# api_base = "http://localhost:1234/v1"

[keys]                     # per provider, used when that provider's model is selected
anthropic = "sk-ant-..."
openai = "sk-..."
```

Keys are passed straight to the model call, not put into the environment, so code you run in
the session can't see them. A provider with no key here falls back to its usual environment
variable. `AIPYTHON_MODEL` and `AIPYTHON_API_BASE` override the file. aipython warns if the file
is readable by other users (`chmod 600` it).

## Development

```bash
uv run --extra test pytest
```

## Releasing

Releases are published to PyPI by GitHub Actions (`.github/workflows/publish.yml`) using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API token is needed.

One-time setup: on PyPI, add a *pending publisher* for project `aipython` with owner `clausd`,
repository `aipython`, workflow `publish.yml`, environment `pypi`.

Each release: bump `__version__` in `aipython/__init__.py`, commit, push, then

```bash
gh release create v0.2.0 --generate-notes
```

## License

[Mozilla Public License 2.0](LICENSE).
