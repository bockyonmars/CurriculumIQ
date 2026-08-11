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


def test_developer_details_hidden_by_default():
    """The diagnostics flag is off unless explicitly enabled via env."""
    from src import config
    assert config.SHOW_DEVELOPER_DETAILS is False


def _tests_show_dev_flag(test: ast.AST) -> bool:
    # Matches `config.SHOW_DEVELOPER_DETAILS` used as a truthy If test.
    return any(
        isinstance(n, ast.Attribute) and n.attr == "SHOW_DEVELOPER_DETAILS"
        for n in ast.walk(test)
    )


def test_developer_panel_is_gated_by_flag():
    """The '🛠 Developer details' expander must sit inside an
    `if config.SHOW_DEVELOPER_DETAILS:` block, so it never renders by default."""
    tree = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))

    # Locate the dev-details expander `with` node.
    dev_with = None
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            _is_st_expander(item.context_expr)
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and "Developer details" in str(item.context_expr.args[0].value)
            for item in node.items
        ):
            dev_with = node
            break
    assert dev_with is not None, "Developer details expander not found in app.py"

    # It must be inside an `if config.SHOW_DEVELOPER_DETAILS:` subtree.
    gated = any(
        isinstance(node, ast.If)
        and _tests_show_dev_flag(node.test)
        and any(dev_with is d for d in ast.walk(node))
        for node in ast.walk(tree)
    )
    assert gated, "Developer details panel is not gated by SHOW_DEVELOPER_DETAILS"


def test_stale_whats_next_section_removed():
    src = APP.read_text(encoding="utf-8")
    assert "What's next" not in src
    assert "Roadmap" not in src


def test_primary_stage_labels_present():
    src = APP.read_text(encoding="utf-8")
    for label in ("Choose PDF", "Prepare curriculum", "Ask questions"):
        assert label in src
