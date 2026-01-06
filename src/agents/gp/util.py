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
    Diviziune protejată pentru a evita erorile de împărțire la zero.
    Returnează `a` dacă `b` este foarte aproape de zero, altfel `a / b`.
    """
    # Daca b e 0, am putea returna o valoare mare daca a e pozitiv,
    # sau a insusi, sau 1.0. Alegerea depinde de cum vrem sa penalizam/interpretam.
    # Varianta initiala `else a` poate fi problematica daca `a` e mic si `b` e aproape de 0.
    # O valoare mare ar putea fi mai sigura pentru a evita prioritati neasteptat de mari.
    if abs(b) < 1e-9:
        if a > 1e-9: return 1e9  # Numar mare pozitiv
        if a < -1e-9: return -1e9  # Numar mare negativ
        return 0.0  # Daca si a e 0
    return a / b