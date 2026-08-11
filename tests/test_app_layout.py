"""Static guard against Streamlit layout bugs in app.py.

Streamlit forbids nesting an expander inside another expander (it raises at
render time and crashes the app). This parses app.py's AST and fails if any
`with st.expander(...)` block contains another `st.expander(...)` call — the
exact production crash this test was added for.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def _is_st_expander(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expander"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    )


def _contains_expander_call(nodes) -> bool:
    return any(_is_st_expander(n) for stmt in nodes for n in ast.walk(stmt))


def test_no_nested_st_expander():
    tree = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if any(_is_st_expander(item.context_expr) for item in node.items):
            # This `with` opens an expander; its body must not open another.
            if _contains_expander_call(node.body):
                offenders.append(node.lineno)
    assert not offenders, f"nested st.expander found at app.py line(s): {offenders}"
