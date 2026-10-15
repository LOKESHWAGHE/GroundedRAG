from app.services.chunking.splitter import chunk_page_fixed, chunk_page_structure_aware
from app.services.chunking.tokenizer import count_words


def test_structure_aware_respects_paragraph_boundaries():
    text = "Para one is short.\n\nPara two is also short."
    chunks = chunk_page_structure_aware(text, chunk_size=100, overlap=0, min_tokens=1)
    assert len(chunks) == 1
    assert "Para one" in chunks[0].text and "Para two" in chunks[0].text


def test_structure_aware_splits_when_over_budget():
    para_a = " ".join(f"wordA{i}" for i in range(15))
    para_b = " ".join(f"wordB{i}" for i in range(15))
    chunks = chunk_page_structure_aware(f"{para_a}\n\n{para_b}", chunk_size=20, overlap=0, min_tokens=1)
    assert len(chunks) == 2
    assert "wordA0" in chunks[0].text and "wordB0" in chunks[1].text


def test_heading_becomes_section_title_and_is_not_its_own_chunk():
    text = "# Introduction\n\nThis explains the background."
    chunks = chunk_page_structure_aware(text, chunk_size=100, overlap=0, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Introduction"
    assert "# Introduction" not in chunks[0].text


def test_section_title_carries_across_multiple_chunks_under_one_heading():
    para_a = " ".join(f"a{i}" for i in range(15))
    para_b = " ".join(f"b{i}" for i in range(15))
    text = f"# Policy\n\n{para_a}\n\n{para_b}"
    chunks = chunk_page_structure_aware(text, chunk_size=20, overlap=0, min_tokens=1)
    assert len(chunks) == 2
    assert all(c.section_title == "Policy" for c in chunks)


def test_overlap_repeats_trailing_words_in_next_chunk():
    para_a = " ".join(f"a{i}" for i in range(20))
    para_b = " ".join(f"b{i}" for i in range(20))
    chunks = chunk_page_structure_aware(f"{para_a}\n\n{para_b}", chunk_size=20, overlap=5, min_tokens=1)
    assert len(chunks) == 2
    tail_of_first = chunks[0].text.split()[-5:]
    head_of_second = chunks[1].text.split()[:5]
    assert tail_of_first == head_of_second


def test_long_paragraph_is_split_on_sentence_boundaries():
    sentences = [f"This is sentence number {i}." for i in range(10)]
    para = " ".join(sentences)
    chunks = chunk_page_structure_aware(para, chunk_size=15, overlap=0, min_tokens=1)
    assert len(chunks) > 1
    for c in chunks:
        assert count_words(c.text) <= 15 + 5


def test_unbreakable_long_run_falls_back_to_hard_word_split():
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunk_page_structure_aware(text, chunk_size=10, overlap=0, min_tokens=1)
    assert len(chunks) == 5
    assert all(count_words(c.text) <= 10 for c in chunks)


def test_chunks_below_min_tokens_are_dropped():
    chunks = chunk_page_structure_aware("hi", chunk_size=100, overlap=0, min_tokens=5)
    assert chunks == []


def test_empty_page_yields_no_chunks():
    assert chunk_page_structure_aware("", 100, 20, 5) == []
    assert chunk_page_fixed("", 100, 20, 5) == []


def test_fixed_strategy_ignores_structure_and_slides_a_window():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_page_fixed(text, chunk_size=10, overlap=2, min_tokens=1)
    assert chunks[0].text.split()[:3] == ["w0", "w1", "w2"]
    assert chunks[1].text.split()[:2] == ["w8", "w9"]
    assert all(c.section_title is None for c in chunks)


def test_fixed_strategy_covers_the_whole_text_without_gaps():
    text = " ".join(f"w{i}" for i in range(37))
    chunks = chunk_page_fixed(text, chunk_size=10, overlap=3, min_tokens=1)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(text.split())
