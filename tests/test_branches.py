"""--branches: when a guessed ending still agrees with the typing, and how the setting reaches mafdet strike."""
from mafdet.core import catalog, config
from mafdet.core.chat import ChatService
from mafdet.watch import Branch

BASE = "Luca ha 25 euro, compra 3 quaderni da 4 euro e una penna da 2 euro. "


def branch(guess: str, base: str = BASE) -> Branch:
    return Branch(slot=1, base=base, stream=None, state="reason", guess=guess)


def test_typing_that_follows_the_guess_agrees():
    b = branch("Quanti euro gli restano?")
    assert b.agreement(BASE, typing=True) == 1.0                       # nothing typed since the guess
    assert b.agreement(BASE + "Quanti euro gli", typing=True) == 1.0
    assert b.agreement(BASE + "Quanti euro gli resta", typing=True) == 1.0   # a word still being typed


def test_typing_that_contradicts_the_guess_does_not():
    b = branch("Quanti euro gli restano?")
    assert b.agreement(BASE + "Che ora segna l'orologio ", typing=True) < 0.5


def test_an_edit_before_the_guess_ends_it():
    b = branch("Quanti euro gli restano?")
    assert b.agreement(BASE.replace("25", "30") + "Quanti", typing=True) == 0.0


def test_full_message():
    assert branch(" Quanti euro gli restano?", BASE.rstrip() + " ").full == BASE.rstrip() + " Quanti euro gli restano?"


def args_for(settings: dict):
    service = ChatService.__new__(ChatService)
    service.settings = {**config.DEFAULTS, **settings}
    v = catalog.find("qwen3-4b")
    return service._args(v, "model.gguf")


def test_setting_turns_one_branch_on():
    assert args_for({"branches": True}).branches == 1
    assert args_for({}).branches == 0


def test_branches_need_reasoning_and_give_way_to_the_gatherer():
    assert args_for({"branches": True, "think": False}).branches == 0
    assert args_for({"branches": True, "gatherer": "some/gatherer.gguf"}).branches == 0
