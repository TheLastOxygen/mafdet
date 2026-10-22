"""TeX in the answers becomes plain symbols in the terminal, and money stays money."""
import pytest

from mafdet.tui.mathtext import render


@pytest.mark.parametrize("src,expected", [
    (r"Costo: $3 \times 4 = 12$ euro.", "Costo: 3 × 4 = 12 euro."),
    (r"$25 - 14 = 11$", "25 - 14 = 11"),
    ("costs $5 and $10 later", "costs $5 and $10 later"),
    (r"\[ \frac{2}{3} \times 432 = 288 \]", "2⁄3 × 432 = 288"),
    (r"La risposta è \boxed{11}.", "La risposta è **11**."),
    (r"$x^2 + y^{10} \leq \sqrt{16}$", "x² + y¹⁰ ≤ √(16)"),
    (r"$18 \text{ pagine/giorno} \times 12 = 216 \text{ pagine}$", "18  pagine/giorno × 12 = 216  pagine"),
    (r"$\boxed{184}$ pagine", "**184** pagine"),
    ("`$a = b$` in code", "`$a = b$` in code"),
])
def test_render(src, expected):
    assert render(src) == expected
