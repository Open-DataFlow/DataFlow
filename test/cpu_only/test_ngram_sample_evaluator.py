import pytest

from dataflow.operators.general_text.eval.ngram_sample_evaluator import (
    NgramSampleEvaluator,
)


@pytest.mark.cpu
def test_auto_mode_uses_character_ngrams_for_han_text():
    evaluator = NgramSampleEvaluator(ngrams=5, language="auto")

    assert evaluator._score_func("今天天气真不错，适合出门散步。") == 1.0


@pytest.mark.cpu
def test_auto_mode_keeps_word_ngrams_for_english_text():
    text = "test test test test test test final"
    auto_evaluator = NgramSampleEvaluator(ngrams=5, language="auto")
    en_evaluator = NgramSampleEvaluator(ngrams=5, language="en")

    assert auto_evaluator._score_func(text) == en_evaluator._score_func(text)


@pytest.mark.cpu
def test_default_and_explicit_en_modes_are_not_overridden_for_mixed_text():
    default_evaluator = NgramSampleEvaluator(ngrams=5)
    en_evaluator = NgramSampleEvaluator(ngrams=5, language="en")
    text = "test test test test test test 中文"

    assert default_evaluator._score_func(text) == pytest.approx(2 / 3)
    assert en_evaluator._score_func(text) == pytest.approx(2 / 3)


@pytest.mark.cpu
def test_auto_mode_detects_han_characters_outside_basic_block():
    evaluator = NgramSampleEvaluator(ngrams=5, language="auto")

    assert evaluator._score_func("𠀀𠀁𠀂𠀃𠀄𠀅") == 1.0


@pytest.mark.cpu
def test_rejects_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        NgramSampleEvaluator(language="cjk")
