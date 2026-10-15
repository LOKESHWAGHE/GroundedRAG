from app.services.verification.sentence_splitter import split_sentences_with_markers


def test_single_sentence_with_one_marker():
    units = split_sentences_with_markers("Product X supports PostgreSQL. [1]")
    assert len(units) == 1
    assert units[0].text == "Product X supports PostgreSQL."
    assert units[0].markers == frozenset({1})


def test_multiple_sentences_each_with_own_markers():
    units = split_sentences_with_markers("First fact. [1] Second fact. [2]")
    assert [u.text for u in units] == ["First fact.", "Second fact."]
    assert units[0].markers == frozenset({1})
    assert units[1].markers == frozenset({2})


def test_sentence_with_multiple_contiguous_markers():
    units = split_sentences_with_markers("It supports both databases. [1][3]")
    assert units[0].markers == frozenset({1, 3})


def test_sentence_with_no_markers_has_empty_marker_set():
    units = split_sentences_with_markers("This sentence cites nothing.")
    assert units[0].markers == frozenset()


def test_question_and_exclamation_marks_are_sentence_boundaries():
    units = split_sentences_with_markers("Is it supported? Yes it is! [1]")
    assert len(units) == 2
    assert units[1].markers == frozenset({1})


def test_empty_string_yields_no_units():
    assert split_sentences_with_markers("") == []
    assert split_sentences_with_markers("   ") == []


def test_trailing_text_without_terminal_punctuation_is_still_captured():
    units = split_sentences_with_markers("A complete sentence. [1] and a trailing fragment")
    assert len(units) == 2
    assert units[1].text == "and a trailing fragment"
    assert units[1].markers == frozenset()
