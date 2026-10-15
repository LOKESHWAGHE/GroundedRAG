from app.services.verification.evidence_windows import build_evidence_windows


def test_short_chunk_is_one_window():
    windows = build_evidence_windows("A single short sentence.", max_windows=3)
    assert len(windows) == 1
    assert windows[0].text == "A single short sentence."


def test_chunk_split_into_sentence_pairs():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    windows = build_evidence_windows(text, max_windows=5, window_size_sentences=2)
    assert len(windows) == 2
    assert "Sentence one." in windows[0].text and "Sentence two." in windows[0].text
    assert "Sentence three." in windows[1].text and "Sentence four." in windows[1].text


def test_long_chunk_is_capped_at_max_windows():
    sentences = " ".join(f"Sentence number {i}." for i in range(20))
    windows = build_evidence_windows(sentences, max_windows=3, window_size_sentences=1)
    assert len(windows) <= 3


def test_capped_sampling_includes_first_and_last_window():
    sentences = " ".join(f"Sentence number {i}." for i in range(20))
    windows = build_evidence_windows(sentences, max_windows=3, window_size_sentences=1)
    indices = [w.window_index for w in windows]
    assert indices[0] == 0
    assert indices[-1] == 19


def test_empty_text_yields_no_windows():
    assert build_evidence_windows("", max_windows=3) == []
    assert build_evidence_windows("   ", max_windows=3) == []
