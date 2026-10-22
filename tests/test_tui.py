"""The terminal interfaces, driven headless by Textual's pilot."""
import asyncio

from textual.widgets import ContentSwitcher

from mafdet.core import config
from mafdet.tui.app import MafdetTUI, Setup
from mafdet.tui.launcher import Launcher


def test_launcher_choice():
    async def go():
        app = Launcher()
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("r")
            await pilot.press("enter")
        return app.return_value
    assert asyncio.run(go()) == ("tui", True)


def test_launcher_letter():
    async def go():
        app = Launcher()
        async with app.run_test() as pilot:
            await pilot.press("w")
        return app.return_value
    assert asyncio.run(go()) == ("gui", False)


def test_first_start_shows_the_setup_and_skipping_saves_it():
    async def go():
        app = MafdetTUI()
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause(2.4)                         # the splash
            assert isinstance(app.screen, Setup)
            await pilot.click("#setup-skip")
            await pilot.pause(0.2)
            assert not isinstance(app.screen, Setup)
    asyncio.run(go())
    assert config.load()["first_run_done"] is True


def test_views_switch_with_function_keys():
    config.save({"first_run_done": True})

    async def go():
        app = MafdetTUI()
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause(2.4)
            seen = []
            for key in ("f2", "f3", "f4", "f1"):
                await pilot.press(key)
                await pilot.pause(0.3)
                seen.append(app.query_one("#views", ContentSwitcher).current)
            return seen
    assert asyncio.run(go()) == ["models", "hardware", "settings", "chat"]


def test_models_view_keys():
    """Tab moves between families, Q switches the quantization of the model under the cursor."""
    config.save({"first_run_done": True})

    async def go():
        from mafdet.tui.app import ModelsView
        app = MafdetTUI()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause(2.4)
            await pilot.press("f2")
            await pilot.pause(0.5)
            view = app.query_one(ModelsView)
            # A model with several quantizations (the first row is the one recommended for this machine).
            view.selected = "qwen3-8b"
            before = view.current()[1]["quant"]
            await pilot.press("q")
            after = view.current()[1]["quant"]
            await pilot.press("tab")
            await pilot.pause(0.2)
            family = view.family
            await pilot.press("f")
            await pilot.pause(0.2)
            fits = view.only
        app.service.close()
        return before, after, family, fits
    before, after, family, fits = asyncio.run(go())
    assert (before, after) == ("Q4_K_M", "Q5_K_M") and family == "qwen3" and fits == "fits"
