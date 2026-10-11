import numpy as np

from fixture import KEYS, vocab_from_keys
from wordpack_compiler.build import encode_vocab


def test_every_key_finds_its_row():
    v = vocab_from_keys(KEYS)
    assert [v.row(k) for k in KEYS] == list(range(len(KEYS)))


def test_absent_key_gives_minus_one():
    v = vocab_from_keys(KEYS)
    assert v.row("zebra") == -1
    assert v.row("") == -1
    assert v.row("do ") == -1


def test_prefix_and_space_and_non_ascii_keys():
    v = vocab_from_keys(KEYS)
    assert v.row("do") == 2 and v.row("don") == 23 and v.row("don't") == 32
    assert v.row("new york") == 21
    assert v.row("café") == 22
    assert v.row("café") == -1  # lookup is exact: no normalization


def test_vocab_order_sorts_utf8_bytes():
    _, order = encode_vocab(KEYS)
    assert order.dtype == np.int32
    expected = sorted(range(len(KEYS)), key=lambda i: KEYS[i].encode("utf-8"))
    assert order.tolist() == expected


def test_sequence_in_row_order():
    v = vocab_from_keys(KEYS)
    assert len(v) == 48
    assert v[0] == "the" and v[22] == "café" and v[-1] == "b"
    assert v[1:3] == ["i", "do"]
    assert list(v) == KEYS
    assert "dog" in v and "zebra" not in v


def test_has_prefix():
    v = vocab_from_keys(KEYS)
    assert v.has_prefix("cov") and v.has_prefix("covid-1") and v.has_prefix("covid-19")
    assert v.has_prefix("u.") and v.has_prefix("and/")
    assert not v.has_prefix("covid-19,") and not v.has_prefix("zz")


def test_empty_vocab():
    v = vocab_from_keys([])
    assert len(v) == 0 and v.row("a") == -1 and not v.has_prefix("a")


def test_keys_beyond_the_top_block_and_repeated_lookups(monkeypatch):
    import wordpack_compiler.reader as reader

    monkeypatch.setattr(reader, "_TOP_KEYS", 5)
    v = vocab_from_keys(KEYS)
    for _ in range(2):  # the second pass is answered from the remembered results
        assert [v.row(k) for k in KEYS] == list(range(len(KEYS)))
        assert v.row("zebra") == -1 and v.row("") == -1
        assert v.has_prefix("cov") and not v.has_prefix("zz")
