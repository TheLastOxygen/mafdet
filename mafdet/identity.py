"""
The MAFDET visual identity (v2): palette, symbol, command icons and wordmark.

Every shape is drawn on a 100-unit grid from straight segments only. Regular
shapes are filled with the foreground colour; accent shapes are amber, and in
monochrome the ones marked "open" disappear, so the background shows through
the cut (the eye stays open). Used by assets/build_assets.py (SVG files) and by
the chat window (mafdet watch).
"""

INK = "#0E0D0C"
PAPER = "#EEEAE2"
AMBER = "#D4892A"

# name → [(path, accent, open in mono)]. Even-odd fill: the accent shapes are
# also cut out of the main mass.
SHAPES = {
    "mask": [
        ("M10 2 L36 24 L64 24 L90 2 L96 46 L72 84 L50 98 L28 84 L4 46 Z "
         "M12 37 L44 45 L45 51 L42 84 L37 53 Z M88 37 L56 45 L55 51 L58 84 L63 53 Z", False, False),
        ("M12 37 L44 45 L45 51 L37 53 Z", True, True),
        ("M88 37 L56 45 L55 51 L63 53 Z", True, True),
    ],
    "awaken": [
        ("M2 44 L98 44 L50 90 Z M22 51 L78 51 L50 67 Z", False, False),
        ("M22 51 L78 51 L50 67 Z", True, True),
        ("M42 2 L58 2 L50 32 Z", False, False),
        ("M4 14 L17 5 L34 36 Z", False, False),
        ("M96 14 L83 5 L66 36 Z", False, False),
    ],
    "watch": [
        ("M2 50 L50 16 L98 50 L50 84 Z M24 50 L50 33 L76 50 L50 67 Z", False, False),
        ("M50 40 L60 50 L50 60 L40 50 Z", True, False),
    ],
    "strike": [
        ("M2 98 L98 2 L98 98 Z M38 90 L88 30 L88 46 L52 90 Z M66 90 L88 64 L88 78 L78 90 Z", False, False),
        ("M38 90 L88 30 L88 46 L52 90 Z", True, True),
    ],
    "hunt": [
        ("M50 2 L98 50 L50 98 L2 50 Z M46 14 L54 14 L54 36 L46 36 Z M46 64 L54 64 L54 86 L46 86 Z "
         "M14 46 L36 46 L36 54 L14 54 Z M64 46 L86 46 L86 54 L64 54 Z M50 41 L59 50 L50 59 L41 50 Z", False, False),
        ("M50 41 L59 50 L50 59 L41 50 Z", True, True),
    ],
    "race": [
        ("M2 12 L58 12 L98 50 L58 88 L2 88 L38 50 Z M28 30 L62 30 L69 37 L33 37 Z "
         "M46 46 L82 46 L86 50 L82 54 L46 54 Z M28 70 L62 70 L69 63 L33 63 Z", False, False),
        ("M46 46 L82 46 L86 50 L82 54 L46 54 Z", True, True),
    ],
    "judge": [
        ("M2 16 L98 16 L98 28 L60 28 L60 74 L50 99 L40 74 L40 28 L2 28 Z M47 34 L53 34 L53 72 L50 80 L47 72 Z",
         False, False),
        ("M47 34 L53 34 L53 72 L50 80 L47 72 Z", True, True),
        ("M2 38 L32 38 L17 62 Z", False, False),
        ("M68 38 L98 38 L83 62 Z", False, False),
        ("M50 0 L57 7 L50 14 L43 7 Z", False, False),
    ],
}

# Wordmark: cap height 100, stems of 14, letters placed at x = offset.
# The A has no crossbar, the terminals of F, E, T and the corners of D are cut at 45°.
WORDMARK_SIZE = (700, 100)
WORDMARK = [
    (0, "M0 100 L0 0 L18 0 L52 54 L86 0 L104 0 L104 100 L90 100 L90 26 L58 76 L46 76 L14 26 L14 100 Z"),
    (140, "M0 100 L39 0 L55 0 L94 100 L79 100 L47 18 L15 100 Z"),
    (270, "M0 0 L74 0 L60 14 L14 14 L14 43 L62 43 L48 57 L14 57 L14 100 L0 100 Z"),
    (380, "M0 0 L64 0 L90 26 L90 74 L64 100 L0 100 Z M14 14 L58 14 L76 32 L76 68 L58 86 L14 86 Z"),
    (506, "M0 0 L74 0 L60 14 L14 14 L14 43 L62 43 L48 57 L14 57 L14 86 L60 86 L74 100 L0 100 Z"),
    (616, "M0 0 L84 0 L70 14 L49 14 L49 100 L35 100 L35 14 L14 14 Z"),
]
