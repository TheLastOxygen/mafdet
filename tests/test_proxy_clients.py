"""The browser-side clients: syntax of the user script, behaviour of mafdet-drafts.js (needs Node)."""
import shutil
import subprocess
from pathlib import Path

import pytest

CLIENTS = Path(__file__).resolve().parents[1] / "mafdet" / "proxy" / "clients"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@needs_node
@pytest.mark.parametrize("name", ["open-webui.user.js", "mafdet-drafts.js"])
def test_the_scripts_parse(name):
    subprocess.run([NODE, "--check", str(CLIENTS / name)], check=True)


@needs_node
def test_mafdet_drafts_debounces_pastes_and_stays_quiet_when_the_proxy_is_down():
    check = Path(__file__).with_name("proxy_drafts_check.js")
    out = subprocess.run([NODE, str(check), str(CLIENTS / "mafdet-drafts.js")],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr + out.stdout
    assert out.stdout.startswith("ok")
