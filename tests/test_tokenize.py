import copy
import unicodedata

import pytest

from fixture import KEYS, vocab_from_keys
from wordpack_compiler.build import char_table
from wordpack_compiler.reader import Chars, resolve, tokenize

CLITICS = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
MASKED = ["shit", "dick", "duck"]
CS_KEYS = KEYS + ["Polska", "Polish", "polish", "czytały"]  # rows 48-51


@pytest.fixture(scope="module")
def vocab():
    return vocab_from_keys(KEYS)


CHARS = Chars(char_table())


def tok(vocab, text, chars=CHARS, **kw):
    kw.setdefault("clitics", CLITICS)
    kw.setdefault("masked", MASKED)
    return tokenize(text, vocab, chars, **kw)


def check(t, sents, keys, rows, spans, sent):
    assert t.sents == sents
    assert t.keys == keys
    assert t.rows == rows
    assert t.spans == spans
    assert t.sent == sent


def test_words_clitics_and_final_period(vocab):
    t = tok(vocab, "I don't like the dog.")
    check(t, [(0, 21)],
          ["i", "do", "n't", "like", "the", "dog"], [1, 2, 3, 39, 0, 15],
          [(0, 1), (2, 4), (4, 7), (8, 12), (13, 16), (17, 20)], [0] * 6)


def test_vocabulary_match(vocab):
    t = tok(vocab, "It's covid-19, e.g., 4chan and/or it.")
    check(t, [(0, 37)],
          ["it", "'s", "covid-19", "e.g.", "4chan", "and/or", "it"],
          [12, 4, 24, 33, 26, 28, 12],
          [(0, 2), (2, 4), (5, 13), (15, 19), (21, 26), (27, 33), (34, 36)], [0] * 7)


def test_masked_words_emphasis_digits(vocab):
    t = tok(vocab, "Sh*t happens with café! *Never 2024 x")
    check(t, [(0, 23), (24, 37)],
          ["shit", "happens", "with", "café", "never", "x"], [34, 41, 42, 22, 37, 46],
          [(0, 4), (5, 12), (13, 17), (18, 22), (25, 30), (36, 37)], [0, 0, 0, 0, 1, 1])


def test_hyphens_curly_apostrophe_line_break_ambiguous_mask(vocab):
    t = tok(vocab, "Blue-green dogs can’t swim\nwell-known d*ck")
    check(t, [(0, 26), (27, 42)],
          ["blue", "green", "dogs", "ca", "n't", "swim", "well-known", "d", "ck"],
          [19, 20, 16, 10, 3, 40, 18, -1, -1],
          [(0, 4), (5, 10), (11, 15), (16, 18), (18, 21), (22, 26), (27, 37), (38, 39), (40, 42)],
          [0, 0, 0, 0, 0, 0, 1, 1, 1])


def test_nfc_offsets_and_abbreviation_split(vocab):
    raw = "cafe\u0301 Dr. Who"
    t = tok(vocab, raw)
    assert t.text == unicodedata.normalize("NFC", raw) and len(t.text) == 12
    check(t, [(0, 8), (9, 12)], ["café", "dr", "who"], [22, -1, -1],
          [(0, 4), (5, 7), (9, 12)], [0, 0, 1])


def test_skipped_spans(vocab):
    t = tok(vocab, "see https://x.org/a.b @bob mail@x.com")
    check(t, [(0, 37)], ["see", "mail", "x", "com"], [-1, -1, 46, -1],
          [(0, 3), (27, 31), (32, 33), (34, 37)], [0] * 4)


def test_abbreviation_with_periods(vocab):
    t = tok(vocab, "The U.S. is big")
    check(t, [(0, 8), (9, 15)], ["the", "u.s.", "is", "big"], [0, 27, 13, -1],
          [(0, 3), (4, 8), (9, 11), (12, 15)], [0, 0, 1, 1])


def test_longest_qualifying_match_then_rest(vocab):
    t = tok(vocab, "covid-19x")
    assert t.keys == ["covid-19", "x"] and t.spans == [(0, 8), (8, 9)]


def test_stray_clitics(vocab):
    t = tok(vocab, "the 90's covid-19's u.s.'s sh*t's 'so'")
    check(t, [(0, 38)],
          ["the", "'s", "covid-19", "'s", "u.s.", "'s", "shit", "'s", "so"],
          [0, 4, 24, 4, 27, 4, 34, 4, -1],
          [(0, 3), (6, 8), (9, 17), (17, 19), (20, 24), (24, 26), (27, 31), (31, 33), (35, 37)],
          [0] * 9)


def test_clitics_inside_hyphenated_words(vocab):
    t = tok(vocab, "don't-care well-known's")
    check(t, [(0, 23)], ["do", "n't", "care", "well-known", "'s"], [2, 3, -1, 18, 4],
          [(0, 2), (2, 5), (6, 10), (11, 21), (21, 23)], [0] * 5)


def test_clitic_on_capitals(vocab):
    t = tok(vocab, "DON'T")
    assert t.keys == ["do", "n't"] and t.spans == [(0, 2), (2, 5)]


def test_empty_masked_list(vocab):
    t = tok(vocab, "Sh*t", masked=())
    assert t.keys == ["sh", "t"] and t.rows == [-1, -1] and t.spans == [(0, 2), (3, 4)]


def test_paired_emphasis_stars_become_spaces(vocab):
    t = tok(vocab, "**sh** *the dog* s*** *f")
    assert t.text == "  sh    the dog  s*** *f"
    check(t, [(2, 24)], ["sh", "the", "dog", "shit", "f"], [-1, 0, 15, 34, -1],
          [(2, 4), (8, 11), (12, 15), (17, 21), (23, 24)], [0] * 5)


def test_emphasis_star_edges(vocab):
    t = tok(vocab, "*sh*t*")  # the middle star has a letter on each side
    assert t.text == " sh*t " and t.keys == ["shit"] and t.spans == [(1, 5)]
    assert tok(vocab, "*dog\ncat*").text == "*dog\ncat*"  # no pair across a line feed
    assert tok(vocab, "2*x dog*2").text == "2*x dog*2"  # a digit on the outer side
    assert tok(vocab, "* dog *").text == "* dog *"  # whitespace on the inner side
    assert tok(vocab, "*dog **cat** x*").text == " dog   cat   x "
    t = tok(vocab, "https://x.org/*a* @*b* *dog*")  # stars inside a skipped span are left alone
    assert t.text == "https://x.org/*a* @*b*  dog " and t.keys == ["dog"]


# Empty input, URL-only input, Windows line endings


def test_empty_and_whitespace_text(vocab):
    for text in ["", "   ", " \n \t "]:
        t = tok(vocab, text)
        assert t.sents == [] and t.keys == [] and t.spans == []


def test_url_or_handle_only(vocab):
    assert tok(vocab, "https://x.org").sents == [(0, 13)]
    assert tok(vocab, "https://x.org").keys == []
    assert tok(vocab, "@bob").keys == []


def test_windows_line_endings(vocab):
    t = tok(vocab, "dog\r\ncat")
    check(t, [(0, 3), (5, 8)], ["dog", "cat"], [15, 17], [(0, 3), (5, 8)], [0, 1])


# Character table


def test_fold(vocab):
    t = tok(vocab, "infor\u00admation dog\u200bcat don\u2018t")
    assert t.text == "information dog cat don't"
    check(t, [(0, 25)], ["information", "dog", "cat", "do", "n't"], [-1, 15, 17, 2, 3],
          [(0, 11), (12, 15), (16, 19), (20, 22), (22, 25)], [0] * 5)
    assert CHARS.fold("\u2010\u2011\u02bc") == "--'"


def test_long_whitespace_run_is_linear(vocab):
    import time

    n = 100_000
    start = time.perf_counter()
    t = tok(vocab, "dog" + " " * n + "cat")
    assert time.perf_counter() - start < 2
    assert t.keys == ["dog", "cat"]
    assert t.sents == [(0, n + 6)]
    assert t.spans == [(0, 3), (n + 3, n + 6)]


def test_long_punctuation_run_is_linear(vocab):
    import time

    n = 100_000
    start = time.perf_counter()
    t = tok(vocab, "dog" + "." * n + "cat")
    assert time.perf_counter() - start < 2
    assert t.keys == ["dog", "cat"]
    assert t.sents == [(0, n + 6)]


def test_reader_follows_the_pack_table(vocab):
    table = copy.deepcopy(char_table())
    table["letter"].append([0x2603, 0x2603])  # mark the snowman as a letter
    assert tok(vocab, "a\u2603b").keys == ["a", "b"]
    assert tok(vocab, "a\u2603b", chars=Chars(table)).keys == ["a\u2603b"]


def test_case_maps_have_no_context_rule():
    assert CHARS.lower("\u039f\u03a3") == "\u03bf\u03c3"  # not final sigma
    assert CHARS.lower("\u0130") == "i\u0307"
    assert CHARS.upper("\u00df") == "SS"


# Case-sensitive packs


def test_case_rule():
    v = vocab_from_keys(CS_KEYS)
    assert resolve("Polska", v, True, CHARS) == ("Polska", 48)
    assert resolve("polska", v, True, CHARS) == ("Polska", 48)
    assert resolve("POLSKA", v, True, CHARS) == ("Polska", 48)
    assert resolve("Polish", v, True, CHARS) == ("Polish", 49)
    assert resolve("polish", v, True, CHARS) == ("polish", 50)
    assert resolve("Czytały", v, True, CHARS) == ("czytały", 51)
    assert resolve("Zebra", v, True, CHARS) == ("zebra", -1)
    assert resolve("POLSKA", v, False, CHARS) == ("polska", -1)


def test_case_sensitive_tokenize():
    v = vocab_from_keys(CS_KEYS)
    t = tok(v, "POLSKA polish Polish. Czytały", case_sensitive=True)
    check(t, [(0, 21), (22, 29)],
          ["Polska", "polish", "Polish", "czytały"], [48, 50, 49, 51],
          [(0, 6), (7, 13), (14, 20), (22, 29)], [0, 0, 0, 1])
