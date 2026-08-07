"""Regenerate docs/api.md from the source.

Reads the modules with ast rather than importing them, so it runs without the
heavy dependencies installed and without executing the Streamlit script.

    python docs/build_api_docs.py
"""

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "api.md"

# Order is the order you would meet these modules working through the project.
MODULES = [
    ("data_functions.py", "Loading the published metrics tables."),
    ("plot_functions.py", "The matplotlib charts the notebook draws."),
    ("c3d_functions.py", "Reading, repairing and measuring the raw motion capture."),
    ("percentiles.py", "Ranking the model's inputs across the dataset."),
    ("dashboard.py", "Building the animated Plotly figure."),
    ("streamlit_app.py", "The Streamlit front end."),
]


def signature(node):
    """Render a def line the way it appears in the source, defaults included."""
    args = node.args
    parts = []

    positional = args.posonlyargs + args.args
    padding = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, padding):
        parts.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return f"{node.name}({', '.join(parts)})"


def constants(tree):
    """Module level names in SCREAMING_CASE, with the value where it is short."""
    found = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                value = ast.unparse(node.value)
                found.append((target.id, value if len(value) <= 90 else "..."))
    return found


def describe(node):
    """First paragraph of a docstring, then whatever follows, both dedented."""
    doc = ast.get_docstring(node)
    if not doc:
        return "", ""
    first, _, rest = doc.strip().partition("\n\n")
    return " ".join(first.split()), textwrap.dedent(rest).strip()


def render_module(path, blurb):
    tree = ast.parse((ROOT / path).read_text())
    summary, detail = describe(tree)

    lines = [f"## `{path}`", ""]
    lines += [f"*{blurb}*", ""]
    if summary:
        lines += [summary, ""]
    if detail:
        lines += [detail, ""]

    names = constants(tree)
    if names:
        lines += ["### Constants", "", "| Name | Value |", "| --- | --- |"]
        lines += [f"| `{name}` | `{value}` |" for name, value in names]
        lines += [""]

    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    public = [n for n in functions if not n.name.startswith("_")]
    private = [n for n in functions if n.name.startswith("_")]

    if public:
        lines += ["### Functions", ""]
        for node in public:
            lines += [f"#### `{signature(node)}`", ""]
            summary, detail = describe(node)
            lines += [summary or "_No docstring._", ""]
            if detail:
                lines += [detail, ""]

    if private:
        lines += ["### Internal helpers", "", "| Function | What it does |", "| --- | --- |"]
        for node in private:
            summary, _ = describe(node)
            lines.append(f"| `{signature(node)}` | {summary or '—'} |")
        lines += [""]

    return lines


def main():
    lines = [
        "# API reference",
        "",
        "Every module, constant and function in the project, generated from the source by",
        "`docs/build_api_docs.py`. Rerun that script after changing a signature or a",
        "docstring so this file stays honest.",
        "",
        "## Contents",
        "",
    ]
    lines += [f"- [`{path}`](#{path.replace('.', '')}) — {blurb}" for path, blurb in MODULES]
    lines += [""]

    for path, blurb in MODULES:
        lines += render_module(path, blurb)

    OUTPUT.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
