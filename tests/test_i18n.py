"""The four languages of the window have the same keys and the same placeholders."""
import json
import re
from importlib.resources import files

import pytest

LANGS = ("en", "it", "fr", "de")


def load(lang):
    return json.loads(files("mafdet").joinpath("ui", "i18n", f"{lang}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang", LANGS[1:])
def test_same_keys_and_placeholders(lang):
    en, other = load("en"), load(lang)
    assert set(other) == set(en), f"missing: {set(en) - set(other)}, extra: {set(other) - set(en)}"
    for key, text in en.items():
        assert set(re.findall(r"\{\w+\}", other[key])) == set(re.findall(r"\{\w+\}", text)), key


def test_keys_used_by_the_page_exist():
    """Every t("key") and data-i18n="key" in the page's files is in the English file."""
    en = load("en")
    ui = files("mafdet").joinpath("ui")
    used = set()
    for name in ("index.html", "app.js", "chat.js", "views.js"):
        src = ui.joinpath(name).read_text(encoding="utf-8")
        used |= set(re.findall(r"""\bt\(\s*["']([\w.]+)["']""", src))
        used |= set(re.findall(r"""data-i18n(?:-[\w-]+)?=["']([\w.]+)["']""", src))
    assert not used - set(en), sorted(used - set(en))
