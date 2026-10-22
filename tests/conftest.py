import pytest


@pytest.fixture(autouse=True)
def mafdet_home(tmp_path, monkeypatch):
    """Every test gets its own empty ~/.mafdet: no settings, measurements or models of the real user."""
    monkeypatch.setenv("MAFDET_HOME", str(tmp_path / "mafdet-home"))
    return tmp_path / "mafdet-home"
