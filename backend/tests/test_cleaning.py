from app.services.ingestion.cleaning import clean_text


def test_dehyphenates_words_split_across_lines():
    assert clean_text("inter-\nnational trade") == "international trade"


def test_keeps_real_hyphens_before_capitals_or_digits():
    assert clean_text("GPT-\n4 model") == "GPT-\n4 model"


def test_removes_control_characters_and_soft_hyphens():
    assert clean_text("he\x00llo\x07 wo\u00adrld") == "hello world"


def test_folds_ligatures_and_nbsp():
    assert clean_text("e\ufb03cient\u00a0systems") == "efficient systems"


def test_collapses_inline_spaces_but_keeps_indentation():
    assert clean_text("a    b\n    - nested") == "a b\n    - nested".strip()


def test_preserves_paragraph_breaks_and_limits_blank_lines():
    assert clean_text("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_normalises_windows_newlines():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_is_idempotent():
    once = clean_text("Some   text\r\n\r\n\r\nwith  noise-\nlike this.  ")
    assert clean_text(once) == once
