# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Describe the user's namespace to the model, compactly by default and in depth on request."""

import inspect
import re
import types

MAX_SUMMARY_CHARS = 8000
MAX_COLUMNS = 40
SKIP_NAMES = {"In", "Out", "exit", "quit", "get_ipython"}


def user_variables(shell):
    """The user's own names: no IPython internals, no _-prefixed names."""
    hidden = shell.user_ns_hidden
    return {
        k: v for k, v in shell.user_ns.items()
        if not k.startswith("_") and k not in SKIP_NAMES
        and not (k in hidden and hidden[k] is v)
    }


def _short_repr(value, limit=80):
    try:
        r = repr(value)
    except Exception:
        return "<unrepresentable>"
    return r if len(r) <= limit else r[: limit - 3] + "..."


def _columns(df, limit=MAX_COLUMNS):
    cols = [f"{c!r}: {t}" for c, t in list(df.dtypes.items())[:limit]]
    more = f", ... +{len(df.columns) - limit} more" if len(df.columns) > limit else ""
    return "{" + ", ".join(cols) + more + "}"


def describe_short(value):
    """One line describing a value."""
    kind = type(value).__name__
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        if kind == "DataFrame":
            return f"DataFrame {value.shape} columns {_columns(value)}"
        if kind == "Series":
            return f"Series len={len(value)} dtype={value.dtype} name={value.name!r}"
    if module.startswith("numpy") and hasattr(value, "shape"):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        try:
            return f"function{inspect.signature(value)}"
        except (TypeError, ValueError):
            return "function"
    if isinstance(value, type):
        return f"class {value.__name__}"
    if isinstance(value, (list, tuple, set, dict)) and len(value) > 5:
        return f"{kind} len={len(value)} e.g. {_short_repr(value, 60)}"
    return f"{kind} = {_short_repr(value)}"


def namespace_summary(shell):
    """Compact text summary of everything in scope, for the system context."""
    variables = user_variables(shell)
    modules = {k: v for k, v in variables.items() if isinstance(v, types.ModuleType)}
    lines = []
    if modules:
        lines.append("imports: " + ", ".join(
            k if k == v.__name__ else f"{k}={v.__name__}" for k, v in modules.items()))
    for name, value in variables.items():
        if name not in modules:
            lines.append(f"{name}: {describe_short(value)}")
    text = "\n".join(lines) or "(namespace is empty)"
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n... (truncated; use the tools to look further)"
    return text


# --- tools the model can call -------------------------------------------------

def _lookup(shell, name):
    """Resolve a dotted name like `df` or `obj.attr` in the user namespace."""
    head, *rest = name.split(".")
    if head not in shell.user_ns:
        raise KeyError(f"no variable named {head!r}")
    value = shell.user_ns[head]
    for attr in rest:
        value = getattr(value, attr)
    return value


def inspect_variable(shell, name):
    value = _lookup(shell, name)
    kind = type(value).__name__
    module = type(value).__module__ or ""
    if module.startswith("pandas") and kind == "DataFrame":
        return "\n".join([
            f"DataFrame shape={value.shape}",
            f"index: {type(value.index).__name__} {_short_repr(list(value.index[:5]))}",
            "dtypes / non-null counts:",
            *(f"  {c!r}: {t} ({value[c].notna().sum()} non-null)"
              for c, t in value.dtypes.items()),
            "head:",
            value.head(5).to_string(max_cols=30, max_colwidth=40),
        ])
    if module.startswith("pandas") and kind == "Series":
        return "\n".join([
            f"Series len={len(value)} dtype={value.dtype} name={value.name!r}",
            value.head(10).to_string(),
        ])
    if callable(value):
        doc = (inspect.getdoc(value) or "").split("\n\n")[0]
        try:
            sig = str(inspect.signature(value))
        except (TypeError, ValueError):
            sig = "(...)"
        return f"{kind} {name}{sig}\n{doc}"
    out = [f"type: {module}.{kind}", f"repr: {_short_repr(value, 1500)}"]
    if isinstance(value, dict):
        out.append(f"keys: {_short_repr(list(value)[:50], 1500)}")
    attrs = [a for a in dir(value) if not a.startswith("_")]
    if attrs and not isinstance(value, (str, int, float, list, dict, tuple)):
        out.append(f"attributes: {', '.join(attrs[:60])}")
    return "\n".join(out)


def column_values(shell, variable, column):
    df = _lookup(shell, variable)
    s = df[column]
    out = [f"{column!r}: dtype={s.dtype} nulls={s.isna().sum()} unique={s.nunique()}"]
    if s.dtype.kind in "iufM":
        out.append(f"min={s.min()} max={s.max()}")
    out.append("most common:\n" + s.value_counts(dropna=False).head(20).to_string())
    return "\n".join(out)


def search_namespace(shell, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for name, value in user_variables(shell).items():
        if rx.search(name):
            hits.append(f"variable {name}: {describe_short(value)}")
        if type(value).__name__ == "DataFrame":
            hits += [f"column {name}[{c!r}] ({t})" for c, t in value.dtypes.items()
                     if rx.search(str(c))]
        elif isinstance(value, dict):
            hits += [f"key {name}[{k!r}]" for k in value if rx.search(str(k))]
    return "\n".join(hits[:100]) or "no matches"


TOOLS = {
    "inspect_variable": (inspect_variable, {
        "description": "Detailed view of a variable (dotted names allowed): for a DataFrame all "
                       "column dtypes, non-null counts and the first rows; for objects their "
                       "attributes; for callables their signature and docstring.",
        "properties": {"name": {"type": "string"}},
    }),
    "column_values": (column_values, {
        "description": "Value distribution of one DataFrame column: dtype, null count, number "
                       "of unique values, range, and the most common values. Use it to get "
                       "exact category spellings before filtering.",
        "properties": {"variable": {"type": "string"}, "column": {"type": "string"}},
    }),
    "search_namespace": (search_namespace, {
        "description": "Case-insensitive regex search over variable names, DataFrame column "
                       "names and dict keys in the user's session.",
        "properties": {"pattern": {"type": "string"}},
    }),
}


def tool_schemas():
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": spec["properties"],
                "required": list(spec["properties"]),
                "additionalProperties": False,
            },
            "strict": True,
        }
        for name, (_, spec) in TOOLS.items()
    ]


def run_tool(shell, name, args):
    """Run a tool; returns (text, is_error)."""
    try:
        return TOOLS[name][0](shell, **args), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", True
