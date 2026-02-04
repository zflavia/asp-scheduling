import random

def generate_random_value_for_erc():
    return round(random.uniform(-5, 5), 2)

def protected_if(cond, true_expr, false_expr):
    """
    Evaluare IF .
    - cond > 0   → return true_expr
    - cond <= 0  → return false_expr
    """
    try:
        if cond is None:
            return false_expr

        # NaN as False
        if isinstance(cond, float) and (cond != cond):
            return false_expr

        return true_expr if cond > 0 else false_expr
    except Exception:
        return false_expr

def gt(x, y):
    return x > y

def lt(x, y):
    return x < y

def protected_div(a: float, b: float) -> float:
    """
    Protected division to avoid division-by-zero errors.
    Returns `a` if `b` is very close to zero, otherwise `a / b`.
    """
    if abs(b) < 1e-9:
        if a > 1e-9: return 1e9  # Large positive number
        if a < -1e-9: return -1e9  # large negative number
        return 0.0  # If a is 0
    return a / b