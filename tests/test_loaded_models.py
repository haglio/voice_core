from __future__ import annotations

import pytest

from voice_core.loaded_models import LoadedModels


def test_a_model_asked_for_twice_is_loaded_once_and_handed_to_both():
    loads = []
    models = LoadedModels()

    first = models.take("small", lambda: loads.append("small") or object())
    second = models.take("small", lambda: loads.append("small again") or object())

    assert (loads, second) == (["small"], first)


def test_models_under_different_names_are_each_loaded():
    models = LoadedModels()

    assert models.take("base", lambda: "the base model") != models.take("small", lambda: "the small model")


def test_a_model_that_failed_to_load_is_tried_again_by_the_next_to_ask():
    models = LoadedModels()

    def refuses():
        raise OSError("not on this machine")

    with pytest.raises(OSError, match="not on this machine"):
        models.take("small", refuses)

    assert models.take("small", lambda: "the small model") == "the small model"
