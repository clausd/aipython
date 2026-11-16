# aipython — IPython that understands what you meant

`aipython` is IPython, except cells that aren't valid Python get turned into Python by Claude
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
   automagic word like `reset the index of df`) the cell goes to Claude with a summary of
   your namespace (variables, DataFrame columns and dtypes), recent cells, and the last error.
   The generated code streams to the terminal and runs. The model can call small tools to look
   closer (`inspect_variable`, `column_values`, `search_namespace`).

`In[n]` holds the code that ran; up-arrow history keeps what you typed.

## Install

```bash
pipx install aipython          # or: uv tool install aipython / pip install aipython
export ANTHROPIC_API_KEY=...
aipython                       # or, in any IPython: %load_ext aipython
```

Tip: install it next to your data libraries (`pip install aipython` in the same environment as
pandas etc.) so generated code can import them.

## Controls

```
%ai                 status
%ai on | off        automatic translation on/off
%ai model sonnet    haiku (default), sonnet, opus, or a full model id
%ai context         the namespace summary the model sees
%ai last            last generated code
%ai <anything>      translate even if it already looks like Python
```

The default model can also be set with `AIPYTHON_MODEL`.

## Development

```bash
uv venv && uv pip install -e ".[test]"
.venv/bin/pytest
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
