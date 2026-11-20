from __future__ import annotations

from types import SimpleNamespace

from voice_core.listening_alone import MachineClaim, NobodysClaim


def _lock(*, free=True):
    calls: list[str] = []

    def acquire(timeout=None):
        calls.append(f"acquire within {timeout}")
        return free

    return SimpleNamespace(calls=calls, acquire=acquire, release=lambda: calls.append("release"),
                           close=lambda: calls.append("close"))


def test_the_claim_is_taken_without_waiting_for_a_listener_that_already_holds_it():
    lock = _lock(free=False)

    assert (MachineClaim(lock).take(), lock.calls) == (False, ["acquire within 0"])


def test_the_claim_given_back_lets_the_lock_go():
    lock = _lock()
    claim = MachineClaim(lock)
    claim.take()

    claim.give_back()

    assert lock.calls[-1] == "close"


def test_a_claim_nobody_else_holds_is_tried_and_let_go_at_once_and_one_held_is_left_alone():
    free, held = _lock(), _lock(free=False)

    assert (MachineClaim(free).held_elsewhere(), free.calls) == (False, ["acquire within 0", "release"])
    assert (MachineClaim(held).held_elsewhere(), held.calls) == (True, ["acquire within 0"])


def test_a_listener_with_no_claim_handed_in_neither_takes_one_nor_is_ever_muted_by_one():
    claim = NobodysClaim()

    assert (claim.take(), claim.held_elsewhere(), claim.give_back()) == (False, False, None)
