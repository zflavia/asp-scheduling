"""Utility helpers for DEAP Genetic Programming individuals.

Features
========
* **simplify_individual** – încearcă:
  1. `deap.gp.simplify` (SymPy‑powered) – dacă funcționează.
  2. Peephole simplifier cu reguli algebrice + constant‑fold.
  În caz de eroare, întoarce individul original.

* **tree_str** – ascii‑tree pentru un `PrimitiveTree`.
"""
from __future__ import annotations

import operator as _op
from numbers import Number
from typing import Any, Sequence, Union, Dict, Tuple

from deap import gp

try:
    import sympy

    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_const(node) -> bool:
    return isinstance(node, Number)


def _const_val(node) -> Union[int, float]:
    return float(node)


# ---------------------------------------------------------------------------
# Simplifier (recursive)
# ---------------------------------------------------------------------------

def _simplify_rec(name: str, args: Sequence[Any]):
    children = list(args)

    # algebraic rules ------------------------------------------------------
    if name == "add":
        # scoatem +0
        children = [c for c in children if not (_is_const(c) and _const_val(c) == 0)]
        if not children:
            return 0.0
        if len(children) == 1:
            return children[0]

        # reguli de semn: add(x, -y) -> sub(x, y), add(-x, y) -> sub(y, x)
        if len(children) == 2:
            left, right = children

            # add(x, neg(y)) -> sub(x, y)
            if isinstance(right, tuple) and right[0] == "neg":
                return ("sub", left, right[1])

            # add(neg(x), y) -> sub(y, x)
            if isinstance(left, tuple) and left[0] == "neg":
                return ("sub", right, left[1])

    elif name == "mul":
        if any(_is_const(c) and _const_val(c) == 0 for c in children):
            return 0.0
        children = [c for c in children if not (_is_const(c) and _const_val(c) == 1)]
        if not children:
            return 1.0
        if len(children) == 1:
            return children[0]


    elif name == "sub" and len(children) == 2:
        left, right = children

        # sub(x, x) -> 0  (deja aveai regula asta)
        if left == right:
            return 0.0

        # sub(x, 0) -> x
        if _is_const(right) and _const_val(right) == 0:
            return left

        # sub(0, x) -> neg(x)
        if _is_const(left) and _const_val(left) == 0:
            return ("neg", right)

        # sub(x, -c) -> add(x, c)  (c > 0 pentru că -c < 0)
        if _is_const(right) and _const_val(right) < 0:
            return ("add", left, -_const_val(right))

        # sub(x, neg(y)) -> add(x, y)
        if isinstance(right, tuple) and right[0] == "neg":
            return ("add", left, right[1])


    elif name in ("max", "min") and len(children) == 2 and children[0] == children[1]:
        # max(x, x) -> x, min(x, x) -> x
        return children[0]

    elif name == "neg":
        child = children[0]
        if isinstance(child, tuple) and child[0] == "neg": # neg(neg(x)) → x
            return child[1]
        if _is_const(child):  # neg(constant) → constant
            return -_const_val(child)
        return (name, child)


    # ----------------------- logical / if rules ---------------------------
    # protected_if(cond, a, b) – dacă cond e boolean constant, alege direct ramura
    if name == "protected_if" and len(children) == 3:
        cond, a, b = children
        if isinstance(cond, bool):
            return a if cond else b

    # case a<a and a>a
    if name in ("gt", "lt") and len(children) == 2 and children[0] == children[1]:
        return False

    # constant folding pentru comparații gt / lt
    if name in ("gt", "lt") and len(children) == 2 and all(_is_const(c) for c in children):
        a, b = (_const_val(c) for c in children)
        try:
            if name == "gt":
                return bool(a > b)
            else:  # "lt"
                return bool(a < b)
        except Exception:  # pragma: no cover
            pass

    # constant folding -----------------------------------------------------
    if all(_is_const(c) for c in children):
        consts = [_const_val(c) for c in children]
        try:
                if name == "add":
                    return sum(consts)
                elif name == "sub":
                    return consts[0] - consts[1]
                elif name == "mul":
                    return _op.mul(consts[0], consts[1])
                elif name == "max":
                    return max(consts)
                elif name == "min":
                    return min(consts)
                elif name == "neg":
                    return -consts[0]
                elif name == "protected_div":
                    return 1.0 if consts[1] == 0 else consts[0] / consts[1]
        except Exception:  # pragma: no cover
            pass

    if len(children) == 1:
        return children[0]

    return (name, *children)


# ---------------------------------------------------------------------------
# Tree <-> nested tuple conversion
# ---------------------------------------------------------------------------

def _to_nested(expr: gp.PrimitiveTree, idx: int = 0):
    """Convert DEAP tree to nested tuples usable by peephole simplifier.

    Returns: (nested_repr, next_index)
    """
    node = expr[idx]
    arity = getattr(node, "arity", 0)

    if arity == 0:  # terminal
        if hasattr(node, "value"):
            return node.value, idx + 1
        return node.name, idx + 1

    children = []
    next_idx = idx + 1
    for _ in range(arity):
        sub, next_idx = _to_nested(expr, next_idx)
        children.append(sub)

    return (node.name, *children), next_idx


def _from_nested(nested, pset):
    if not isinstance(nested, tuple):
        return str(nested)
    name, *children = nested
    child_strs = ( _from_nested(c, pset) for c in children )
    return f"{name}({', '.join(child_strs)})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_individual(ind: gp.PrimitiveTree, pset):
    """Return a *simplified* PrimitiveTree, or the original one on failure."""

    # 1) SymPy path --------------------------------------------------------
    if _HAS_SYMPY:
        try:
            simp = gp.simplify(ind, pset)

            raw_tree = simp if isinstance(simp, gp.PrimitiveTree) else gp.PrimitiveTree.from_string(str(simp), pset)
            ret = type(ind)(raw_tree)
            ret.fitness.values = ind.fitness.values
            return ret
        except Exception:
            pass  # fall through

    # 2) Peephole path -----------------------------------------------------
    nested, _ = _to_nested(ind)

    def walk(expr):
        if not isinstance(expr, tuple):
            return expr
        name, *children = expr
        return _simplify_rec(name, [walk(c) for c in children])

    simplified_nested = walk(nested)
    raw_expr = _from_nested(simplified_nested, pset)

    try:
        raw_tree = gp.PrimitiveTree.from_string(raw_expr, pset)
        ret = type(ind)(raw_tree)
        ret.fitness.values = ind.fitness.values
        return ret
    except Exception:  # Any parser error → give back original
        return ind


# ---------------------------------------------------------------------------
# ASCII tree printer
# ---------------------------------------------------------------------------
def tree_str(expr: gp.PrimitiveTree) -> str:
    """Return a *Unicode* tree diagram using ├─/└─ branches."""

    def _label(node):
        if getattr(node, "arity", 0) == 0:
            return str(node.value) if hasattr(node, "value") else node.name
        return node.name

    lines: list[str] = []

    def rec(idx: int, prefix: str, is_last: bool):
        node = expr[idx]
        branch = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{branch}{_label(node)}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        child_idx = idx + 1
        arity = getattr(node, "arity", 0)
        for i in range(arity):
            rec(child_idx, child_prefix, i == arity - 1)
            child_idx = expr.searchSubtree(child_idx).stop

    rec(0, "", True)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Infix pretty‑printer (mathematical formula)
# ---------------------------------------------------------------------------

# precedence mapping: lower number = binds weaker
_PRECEDENCE: Dict[str, int] = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "protected_div": 2,
    "neg": 3,
    "max": 0,
    "min": 0,
    "gt": 0,
    "lt": 0,
    "protected_if": -1,   # leagă cel mai slab
}

_SYMBOLS: Dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "protected_div": "/",
    "neg": "-",
    "gt": ">",
    "lt": "<",
    "protected_if": "if",
}

def infix_str(expr: gp.PrimitiveTree) -> str:
    """Return an infix formula with **minimal but sufficient** parentheses."""

    def rec(idx: int) -> Tuple[str, int]:
        node = expr[idx]
        name = node.name
        arity = getattr(node, "arity", 0)

        # terminals -------------------------------------------------------
        if arity == 0:
            if hasattr(node, "value"):
                return str(node.value), 4  # high prec.
            return node.name, 4

        # unary neg -------------------------------------------------------
        if name == "neg":
            sub_str, sub_prec = rec(idx + 1)
            if sub_prec < _PRECEDENCE["neg"]:
                sub_str = f"({sub_str})"
            return f"-{sub_str}", _PRECEDENCE["neg"]

        # n‑ary / binary --------------------------------------------------
        child_idx = idx + 1
        parts: list[str] = []
        precs: list[int] = []
        for _ in range(arity):
            s, p = rec(child_idx)
            parts.append(s)
            precs.append(p)
            child_idx = expr.searchSubtree(child_idx).stop

        if name in ("add", "mul"):
            op = _SYMBOLS[name]
            cur_prec = _PRECEDENCE[name]
            wrapped = [f"({s})" if p < cur_prec else s for s, p in zip(parts, precs)]
            return f" {op} ".join(wrapped), cur_prec

        if name in ("sub", "protected_div"):
            op = _SYMBOLS[name]
            cur_prec = _PRECEDENCE[name]
            left = parts[0]
            if precs[0] < cur_prec:
                left = f"({left})"
            right = parts[1]
            # for non‑associative ops we need () even when equal precedence on RHS
            if precs[1] <= cur_prec:
                right = f"({right})"
            return f"{left} {op} {right}", cur_prec

        # functions max/min ---------------------------------------------
        return f"{name}({', '.join(parts)})", _PRECEDENCE[name]

    txt, _ = rec(0)
    return txt