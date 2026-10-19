import json
import struct
import zipfile
from collections import Counter

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

import fixture
from wordpack_compiler.build import LINE_BREAK_TEXTS, build, compute_abtt, encode_vocab, kept_rows
from wordpack_compiler.reader import dequantize

TENSORS = {
    "abtt": ("float32", (11, 16)),
    "norms": ("float32", (48,)),
    "vocab_order": ("int32", (48,)),
    "matrix": ("int8", (48, 16)),
}
BYTE_TENSORS = ["vocab", "stopwords", "masked", "chars", "selftest"]


def test_safetensors_opens_it(pack_path):
    t = load_file(pack_path)
    assert set(t) == set(TENSORS) | set(BYTE_TENSORS)
    for name, (dtype, shape) in TENSORS.items():
        assert t[name].dtype == np.dtype(dtype) and t[name].shape == shape
    for name in BYTE_TENSORS:
        assert t[name].dtype == np.uint8 and t[name].ndim == 1


def test_metadata(pack_path):
    with safe_open(pack_path, "np") as f:
        meta = f.metadata()
    assert meta == {
        "format": "wordpack",
        "format_version": "1",
        "name": "en-fixture",
        "version": "1",
        "description": "Synthetic test pack.",
        "source": json.dumps({"url": "file:///nonexistent", "license": "CC0-1.0", "citation": "none"}),
        "language": "en",
        "keys": "surface",
        "case_sensitive": "false",
        "clitics": json.dumps(fixture.CLITICS),
        "neighbor_filter": ".*\\d",
        "stopwords_source": "fixture",
        "quantization": "int8-rowmax",
        "compiler_version": "0.1.0",
    }


def test_alignment(pack_path):
    raw = pack_path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    assert (8 + size) % 8 == 0
    for name in ["abtt", "norms", "vocab_order"]:
        assert header[name]["data_offsets"][0] % 4 == 0


def test_rows_dropped_and_kept(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    summary = build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert summary["counts"] == {
        "lines": 52,
        "malformed line": 1,
        "duplicate key": 1,
        "zero or non-finite row": 1,
        "<unk> row": 1,
    }
    assert summary["N"] == 48 and summary["D"] == 16
    t = load_file(tmp_path / "p.wordpack")
    keys = bytes(t["vocab"]).decode("utf-8").split("\n")[:-1]
    assert keys == fixture.KEYS  # the NFC form of café that came first won


def test_quantized_matrix(pack_path):
    t = load_file(pack_path)
    q = t["matrix"]
    assert q.min() > -128 and (np.abs(q.astype(int)).max(axis=1) == 127).all()
    src = fixture.source_rows(fixture.KEYS)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    assert (np.sum(dequantize(q, None, 0) * src, axis=1) > 0.999).all()
    assert np.allclose(t["norms"], np.linalg.norm(fixture.source_rows(fixture.KEYS), axis=1), rtol=1e-5)


def test_abtt_and_order_tensors(pack_path):
    t = load_file(pack_path)
    assert np.array_equal(t["abtt"], compute_abtt(t["matrix"]))
    assert np.array_equal(t["vocab_order"], encode_vocab(fixture.KEYS)[1])


def test_selftest_tensor(pack_path):
    data = json.loads(bytes(load_file(pack_path)["selftest"]))
    assert [e["text"] for e in data["texts"]] == fixture.SELFTEST + LINE_BREAK_TEXTS
    assert [(v["row"], v["abtt"]) for v in data["vectors"]] == [
        (0, 0), (0, 1), (0, 3), (47, 0), (47, 1), (47, 3)
    ]
    assert data["texts"][0]["keys"] == ["i", "do", "n't", "like", "the", "dog", "it", "'s", "well-known"]


def test_line_break_texts_match_format_md():
    assert LINE_BREAK_TEXTS == [
        json.loads(s)
        for s in [
            r'"one\ntwo"',
            r'"one\r\ntwo"',
            r'"one.\n\n \t\ntwo"',
            r'"one\rtwo\u000bthree\u000cfour\u0085five\u2028six\u2029seven\u001cend"',
        ]
    ]


def test_line_break_texts_end_every_selftest(pack_path):
    sents = [e["sents"] for e in json.loads(bytes(load_file(pack_path)["selftest"]))["texts"][-4:]]
    assert sents == [[[0, 3], [4, 7]], [[0, 3], [5, 8]], [[0, 4], [9, 12]], [[0, 37]]]


def test_selftest_line_may_hold_unicode_line_separator(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    (lang / "selftest.txt").write_text("the dog\u2028the cat\n", encoding="utf-8")
    build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    texts = json.loads(bytes(load_file(tmp_path / "p.wordpack")["selftest"]))["texts"]
    assert [e["text"] for e in texts[:-4]] == ["the dog\u2028the cat"]


def test_chars_tensor(pack_path):
    table = json.loads(bytes(load_file(pack_path)["chars"]))
    assert set(table) == {"unicode_version", "letter", "digit", "space", "lower", "upper", "fold"}
    assert table["fold"]["173"] == "" and table["fold"]["8203"] == " " and table["fold"]["8217"] == "'"
    assert [65, 90] in table["letter"] and [48, 57] in table["digit"]


def test_masked_and_stopwords_tensors(pack_path):
    t = load_file(pack_path)
    assert bytes(t["masked"]).decode() == "shit\ndick\nduck\n"
    assert bytes(t["stopwords"]).decode() == "the\na\nis\ni\nand\nor\n"


def test_missing_masked_file_gives_empty_tensor(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    (lang / "do_not_read_this_file.txt").unlink()
    build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert load_file(tmp_path / "p.wordpack")["masked"].shape == (0,)


def test_repeated_masked_word_is_kept_once(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    (lang / "do_not_read_this_file.txt").write_text("shit\nShit\ndick\n", encoding="utf-8")
    build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert bytes(load_file(tmp_path / "p.wordpack")["masked"]).decode() == "shit\ndick\n"


def test_nan_and_overflowing_rows_are_dropped(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read("vectors.txt").decode()
    extra = "nanrow " + " ".join(["nan"] + ["1"] * 15) + "\nhuge " + " ".join(["3e38"] * 16) + "\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("vectors.txt", text + extra)
    summary = build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert summary["counts"]["zero or non-finite row"] == 3  # the zero row, nan and huge
    assert summary["N"] == 48
    keys = bytes(load_file(tmp_path / "p.wordpack")["vocab"]).decode().split("\n")
    assert "nanrow" not in keys and "huge" not in keys


def test_masked_word_is_lowercased_with_the_pack_table(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    (lang / "do_not_read_this_file.txt").write_text("\u039f\u03a3\n", encoding="utf-8")
    build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert bytes(load_file(tmp_path / "p.wordpack")["masked"]).decode() == "\u03bf\u03c3\n"


def test_stopword_is_folded_with_the_pack_table(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    (lang / "stopwords.txt").write_text("n't\nn’t\nthe\n", encoding="utf-8")
    build(recipe, lang, zip_path, tmp_path / "p.wordpack", 1)
    assert bytes(load_file(tmp_path / "p.wordpack")["stopwords"]).decode() == "n't\nthe\n"


def test_kept_rows_are_the_rows_a_pack_keeps(tmp_path):
    _, _, zip_path = fixture.write_source(tmp_path)
    counts = Counter()
    keys = [key for key, _, _ in kept_rows(zip_path, fixture.DIMS, counts, False)]
    assert keys == fixture.KEYS
    assert counts == Counter({
        "lines": 52,
        "malformed line": 1,
        "duplicate key": 1,
        "zero or non-finite row": 1,
        "<unk> row": 1,
    })


def test_keys_are_folded_and_the_first_of_each_is_kept(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("vectors.txt").decode().splitlines()
    row = lambda key, v: key + " " + " ".join([v] * fixture.DIMS)
    # "‍" is a format character: it folds to nothing.
    extra = [row("’s", "0.75"), row("‍", "0.5")]
    for name, merged in (("after", lines + extra), ("before", extra + lines)):
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("vectors.txt", "\n".join(merged) + "\n")
        counts = Counter()
        rows = {key: r for key, r, _ in kept_rows(zip_path, fixture.DIMS, counts, False)}
        assert counts["empty key"] == 1 and counts["duplicate key"] == 2
        assert list(rows).count("'s") == 1 and "’s" not in rows and "" not in rows
        assert (rows["'s"] == 0.75).all() == (name == "before")
        summary = build(recipe, lang, zip_path, tmp_path / f"{name}.wordpack", 1)
        keys = bytes(load_file(tmp_path / f"{name}.wordpack")["vocab"]).decode().split("\n")[:-1]
        assert keys.count("'s") == 1 and "" not in keys and summary["N"] == 48


def test_keys_are_lowercased_in_a_case_insensitive_pack(tmp_path):
    _, _, zip_path = fixture.write_source(tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("vectors.txt").decode().splitlines()
    # "Ꭲ" is a Cherokee capital whose lowercase is "ꭲ".
    extra = [key + " " + " ".join(["0.5"] * fixture.DIMS) for key in ("The", "Ꭲhe")]
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("vectors.txt", "\n".join(lines + extra) + "\n")
    insensitive = [key for key, _, _ in kept_rows(zip_path, fixture.DIMS, Counter(), False)]
    assert insensitive == fixture.KEYS + ["ꭲhe"]
    sensitive = [key for key, _, _ in kept_rows(zip_path, fixture.DIMS, Counter(), True)]
    assert sensitive == fixture.KEYS + ["The", "Ꭲhe"]
