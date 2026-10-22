"""
Simple TeX in the answers, as the models write it ($3 \\times 4 = 12$, \\frac{a}{b}, \\boxed{11}),
turned into plain symbols for the terminal. The same rules as the window (ui/chat.js): a pair of
$ is math only if it holds a command or an operator, so "$5 and $10" stays text.
"""
from __future__ import annotations

import re

TEX = {"times": "×", "cdot": "·", "div": "÷", "pm": "±", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
       "ne": "≠", "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→", "Rightarrow": "⇒", "implies": "⇒",
       "pi": "π", "alpha": "α", "beta": "β", "theta": "θ", "lambda": "λ", "mu": "μ", "sigma": "σ", "Delta": "Δ",
       "sum": "Σ", "quad": "  ", "qquad": "    ", "ldots": "…", "dots": "…", "circ": "°"}
SUP = dict(zip("0123456789n-", "⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁻"))
MATH = re.compile(r"\$\$([^$]+)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$(?!\d)")
LOOKS_MATH = re.compile(r"\\[a-zA-Z]|[=+*/^<>]|\d\s*-\s*\d")


def _wrap(s: str) -> str:
    return f"({s})" if len(s) > 1 else s


def tex(src: str) -> str:
    s = src
    for _ in range(4):                       # nested \frac{\sqrt{2}}{2}: a few passes
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"{_wrap(m[1])}⁄{_wrap(m[2])}", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\boxed\{([^{}]*)\}", r"**\1**", s)
    s = re.sub(r"\^\{?([0-9n-]+)\}?", lambda m: "".join(SUP.get(c, c) for c in m[1]), s)
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!", "", s)
    s = re.sub(r"\\([a-zA-Z]+)", lambda m: TEX.get(m[1], m[0]), s)
    return s.replace("\\%", "%").replace("{", "").replace("}", "")


def render(text: str) -> str:
    """The answer with its math turned into symbols; code spans and blocks are left alone."""
    parts = re.split(r"(```[\s\S]*?(?:```|$)|`[^`\n]+`)", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)
            continue
        def one(m):
            a, b, c, d = m.groups()
            if d is not None and not LOOKS_MATH.search(d):
                return m[0]
            return tex(a or b or c or d).strip()
        part = MATH.sub(one, part)
        out.append(re.sub(r"\\boxed\{([^{}]*)\}", r"**\1**", part))
    return "".join(out)
