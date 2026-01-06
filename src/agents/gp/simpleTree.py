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

    ch = list(args)

    # Helpers --------------------------------------------------------------
    def is0(x):
        return _is_const(x) and _const_val(x) == 0

    def is1(x):
        return _is_const(x) and _const_val(x) == 1

    def isneg(x):
        return isinstance(x, tuple) and len(x) >= 2 and x[0] == "neg"

    def neg_arg(x):
        return x[1]  # only call if isneg(x)

    # Logical / if rules ---------------------------------------------------
    if name == "protected_if" and len(ch) == 3:
        cond, a, b = ch
        if isinstance(cond, bool):
            return a if cond else b

    if name in ("gt", "lt") and len(ch) == 2:
        a, b = ch
        if a == b:
            return False
        if _is_const(a) and _is_const(b):
            av, bv = _const_val(a), _const_val(b)
            return bool(av > bv) if name == "gt" else bool(av < bv)

    # Unary neg ------------------------------------------------------------
    if name == "neg" and len(ch) == 1:
        x = ch[0]
        if isneg(x):  # neg(neg(x)) -> x
            return neg_arg(x)
        if _is_const(x):  # neg(const) -> const
            return -_const_val(x)
        return ("neg", x)

    # Binary algebra -------------------------------------------------------
    if name == "add" and len(ch) == 2:
        a, b = ch
        if is0(a): return b
        if is0(b): return a
        if isneg(b): return ("sub", a, neg_arg(b))  # add(x, neg(y)) -> sub(x, y)
        if isneg(a): return ("sub", b, neg_arg(a))  # add(neg(x), y) -> sub(y, x)

    if name == "sub" and len(ch) == 2:
        a, b = ch
        if a == b: return 0.0
        if is0(b): return a
        if is0(a): return ("neg", b)
        if _is_const(b) and _const_val(b) < 0:  # sub(x, -c) -> add(x, c)
            return ("add", a, -_const_val(b))
        if isneg(b):  # sub(x, neg(y)) -> add(x, y)
            return ("add", a, neg_arg(b))

    if name == "mul" and len(ch) == 2:
        a, b = ch
        if is0(a) or is0(b): return 0.0
        if is1(a): return b
        if is1(b): return a

    if name in ("min", "max") and len(ch) == 2 and ch[0] == ch[1]:
        return ch[0]

    # Constant folding -----------------------------------------------------
    if all(_is_const(x) for x in ch):
        vals = [_const_val(x) for x in ch]
        try:
            if name == "add":          return vals[0] + vals[1]
            if name == "sub":          return vals[0] - vals[1]
            if name == "mul":          return _op.mul(vals[0], vals[1])
            if name == "min":          return min(vals)
            if name == "max":          return max(vals)
            if name == "protected_div": return 1.0 if vals[1] == 0 else vals[0] / vals[1]
            # neg handled earlier, but harmless if it gets here
            if name == "neg":          return -vals[0]
        except Exception:
            pass

    # Canonical cleanup for n-ary add/mul (optional) ----------------------
    # If your trees are strictly binary, you can drop this block.
    if name == "add":
        ch = [x for x in ch if not is0(x)]
        if not ch: return 0.0
        if len(ch) == 1: return ch[0]
    if name == "mul":
        if any(is0(x) for x in ch): return 0.0
        ch = [x for x in ch if not is1(x)]
        if not ch: return 1.0
        if len(ch) == 1: return ch[0]

    # Default --------------------------------------------------------------
    return (name, *ch) if len(ch) != 1 else ch[0]


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

    #simplified_nested = walk(nested)
    simplified_nested = nested
    for _ in range(5):  # limită de siguranță
        new_nested = walk(simplified_nested)
        if new_nested == simplified_nested:
            break
        simplified_nested = new_nested
    raw_expr = _from_nested(simplified_nested, pset)

    try:
        raw_tree = gp.PrimitiveTree.from_string(raw_expr, pset)
        ret = type(ind)(raw_tree)
        return ret
    except Exception as e:  # Any parser error → give back original
        print("[simplify] parse failed for:", raw_expr)
        print("[simplify] error:", repr(e))
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