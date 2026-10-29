import json
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from safetensors.numpy import load_file

import fixture
from wordpack_compiler.reader import WordPack, WordPackError, dequantize, read_header


def _rewrite_meta(src, dst, **changes):
    """Copy a pack with metadata values replaced; the data bytes are unchanged."""
    raw = src.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    header["__metadata__"].update(changes)
    new = json.dumps(header, separators=(",", ":")).encode()
    new += b" " * (-len(new) % 8)
    dst.write_bytes(struct.pack("<Q", len(new)) + new + raw[8 + size :])
    return dst


def test_open(pack_path):
    with WordPack(pack_path) as pack:
        assert pack.meta["name"] == "test"
        assert pack.meta["case_sensitive"] is False
        assert pack.meta["clitics"] == fixture.CLITICS
        assert pack.meta["source"]["license"] == "CC0-1.0"
        assert (pack.n, pack.dims) == (48, 16)
        assert list(pack.keys) == fixture.KEYS
        assert pack.stopwords == fixture.STOPWORDS
        assert pack.masked == fixture.MASKED
        assert pack.chars.lower("ÀB") == "àb"


def test_lookup(pack_path):
    with WordPack(pack_path) as pack:
        out = pack.lookup(["dog", "nope", "new york"])
        assert out.dtype == np.int64 and out.tolist() == [15, -1, 21]


def test_read_rows_matches_stored_matrix(pack_path):
    t = load_file(pack_path)
    with WordPack(pack_path) as pack:
        for m in (0, 1, 3, 10):
            assert np.array_equal(pack.read_rows(list(range(48)), m), dequantize(t["matrix"], t["abtt"], m))


def test_read_rows_order_repeats_and_input_types(pack_path):
    with WordPack(pack_path) as pack:
        full = pack.read_rows(range(48))  # ssdiff reads its prefix as a range
        idx = [5, 0, 5, 47, 3, 3, 20]
        # different batch sizes may differ in the last float bits, hence allclose
        assert np.allclose(pack.read_rows(idx), full[idx], atol=1e-6)
        assert np.allclose(pack.read_rows(np.array(idx, dtype=np.int32)), full[idx], atol=1e-6)
        assert pack.read_rows([]).shape == (0, 16)


def test_read_rows_errors(pack_path):
    with WordPack(pack_path) as pack:
        with pytest.raises(IndexError):
            pack.read_rows([48])
        with pytest.raises(IndexError):
            pack.read_rows([-1])
        with pytest.raises(ValueError):
            pack.read_rows([0], abtt=11)


def test_threads(pack_path):
    with WordPack(pack_path) as pack:
        full = pack.read_rows(range(48))
        rng = np.random.default_rng(0)
        jobs = [rng.integers(0, 48, size=10).tolist() for _ in range(400)]
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(pack.read_rows, jobs))
        for idx, got in zip(jobs, results):
            assert np.allclose(got, full[idx], atol=1e-6)


def test_tokenize_uses_pack_settings(pack_path):
    with WordPack(pack_path) as pack:
        t = pack.tokenize("I don't like sh*t.")
        assert t.keys == ["i", "do", "n't", "like", "shit"] and t.rows == [1, 2, 3, 39, 34]


def test_case_sensitive_pack(cased_pack_path):
    with WordPack(cased_pack_path) as pack:
        assert pack.meta["case_sensitive"] is True
        t = pack.tokenize("POLSKA polish Polish. I")
        assert t.keys == ["Polska", "polish", "Polish", "i"] and t.rows == [48, 50, 49, 1]


def test_selftest_passes(pack_path, cased_pack_path):
    for path in (pack_path, cased_pack_path):
        with WordPack(path) as pack:
            pack.selftest()


def test_selftest_fails_on_changed_row_id(pack_path):
    with WordPack(pack_path) as pack:
        pack._selftest["texts"][0]["rows"][0] += 1
        with pytest.raises(WordPackError, match=r"test-1 \(.*self-test failed"):
            pack.selftest()


def test_selftest_fails_on_changed_matrix_byte(pack_path, tmp_path):
    bad = tmp_path / "bad.wordpack"
    shutil.copy(pack_path, bad)
    with open(bad, "rb") as f:
        base, tensors, _ = read_header(f, str(bad))
    pos = base + tensors["matrix"]["data_offsets"][0]  # row 0, component 0
    raw = bytearray(bad.read_bytes())
    raw[pos] = 0 if raw[pos] != 0 else 100
    bad.write_bytes(bytes(raw))
    with WordPack(bad) as pack:
        with pytest.raises(WordPackError, match="self-test failed"):
            pack.selftest()


def test_selftest_fails_on_nan_rows(pack_path):
    with WordPack(pack_path) as pack:
        pack._abtt = np.full_like(pack._abtt, np.nan)
        with pytest.raises(WordPackError, match="self-test failed"):
            pack.selftest()


@pytest.mark.parametrize(
    "changes, message",
    [({"format": "other"}, "not a wordpack"), ({"format_version": "2"}, "format version"), ({"keys": "lemma"}, "surface")],
)
def test_refuses(pack_path, tmp_path, changes, message):
    bad = _rewrite_meta(pack_path, tmp_path / "bad.wordpack", **changes)
    with pytest.raises(WordPackError, match=message):
        WordPack(bad)


def test_truncated_file(pack_path, tmp_path):
    cut = tmp_path / "cut.wordpack"
    cut.write_bytes(pack_path.read_bytes()[:-100])
    with pytest.raises(WordPackError, match="truncated"):
        WordPack(cut)
    tiny = tmp_path / "tiny.wordpack"
    tiny.write_bytes(b"\x01")
    with pytest.raises(WordPackError):
        WordPack(tiny)
    text = tmp_path / "notes.txt"
    text.write_text("This is a text file, not a wordpack at all.\n")
    with pytest.raises(WordPackError):
        WordPack(text)


def _parts(path):
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8 : 8 + size]), raw[8 + size :]


def _set_byte(data, pos, value):
    return data[:pos] + bytes([value]) + data[pos + 1 :]


def _write(dst, header, data):
    new = json.dumps(header, separators=(",", ":")).encode()
    new += b" " * (-len(new) % 8)
    dst.write_bytes(struct.pack("<Q", len(new)) + new + data)
    return dst


def test_truncation_found_with_matrix_stored_last(pack_path, tmp_path):
    header, data = _parts(pack_path)
    start, end = header["matrix"]["data_offsets"]
    rest = data[:start] + data[end:]
    for key, entry in header.items():
        if key not in ("__metadata__", "matrix") and entry["data_offsets"][0] >= end:
            entry["data_offsets"] = [v - (end - start) for v in entry["data_offsets"]]
    header["matrix"]["data_offsets"] = [len(rest), len(rest) + end - start]
    with WordPack(_write(tmp_path / "good.wordpack", header, rest + data[start:end])) as pack:
        pack.selftest()
    cut = _write(tmp_path / "cut.wordpack", header, (rest + data[start:end])[:-100])
    with pytest.raises(WordPackError, match="file is truncated") as e:
        WordPack(cut)
    assert str(cut) in str(e.value)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda h, d: ([1, 2, 3], d),
        lambda h, d: (h["__metadata__"].update(clitics="[not json") or h, d),
        lambda h, d: (h["__metadata__"].update(source="{") or h, d),
        lambda h, d: (h["abtt"].update(dtype="F64") or h, d),
        lambda h, d: (h["__metadata__"].pop("case_sensitive") and h, d),
        lambda h, d: (h["__metadata__"].pop("clitics") and h, d),
        lambda h, d: (h["abtt"].pop("dtype") and h, d),
        lambda h, d: (h["abtt"].pop("shape") and h, d),
        lambda h, d: (h["abtt"].pop("data_offsets") and h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[0]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[0, 1, 2]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[0, "8"]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[0.0, 8]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[True, 8]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=5) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[-1, 8]) or h, d),
        lambda h, d: (h["abtt"].update(data_offsets=[8, 0]) or h, d),
        lambda h, d: (h.update(abtt=[1]) or h, d),
        lambda h, d: (h["abtt"].update(dtype=["F32"]) or h, d),
        lambda h, d: (h["abtt"].update(dtype={"a": 1}) or h, d),
        lambda h, d: (h["abtt"].update(shape="11x16") or h, d),
        lambda h, d: (h["abtt"].update(shape=[-1, 16]) or h, d),
        lambda h, d: (h["abtt"].update(shape=[True, 16]) or h, d),
        lambda h, d: (h["abtt"].update(shape=[1.5, 16]) or h, d),
        lambda h, d: (h["matrix"].update(shape=[48]) or h, d),
        lambda h, d: (h["matrix"].update(shape=[48, 16, 1]) or h, d),
        lambda h, d: (h.update(__metadata__=[1]) or h, d),
        lambda h, d: (h.update(__metadata__="wordpack") or h, d),
        lambda h, d: (h["abtt"].update(shape=[10, 16]) or h, d),
        lambda h, d: (h["matrix"].update(shape=[47, 16]) or h, d),
        lambda h, d: (h, _set_byte(d, h["vocab"]["data_offsets"][0], 0xFF)),
    ],
    ids=[
        "header-not-object", "clitics-json", "source-json", "dtype", "no-case-sensitive", "no-clitics",
        "no-dtype", "no-shape", "no-offsets", "offsets-one", "offsets-three", "offsets-str",
        "offsets-float", "offsets-bool", "offsets-int", "offsets-negative", "offsets-reversed", "entry-not-object",
        "dtype-list", "dtype-dict", "shape-str", "shape-negative", "shape-bool", "shape-float",
        "matrix-1d", "matrix-3d", "metadata-list", "metadata-str", "shape-vs-bytes", "matrix-shape-vs-bytes",
        "vocab-utf8",
    ],
)
def test_corrupt_metadata_names_path(pack_path, tmp_path, corrupt):
    header, data = _parts(pack_path)
    header, data = corrupt(header, data)
    bad = _write(tmp_path / "bad.wordpack", header, data)
    with pytest.raises(WordPackError) as e:
        WordPack(bad)
    assert str(bad) in str(e.value)


def test_vocab_count_mismatch_names_path(pack_path, tmp_path):
    header, data = _parts(pack_path)
    start, end = header["vocab_order"]["data_offsets"]
    header["vocab_order"]["shape"] = [47]
    # Keep the tensor inside the file; the last 4 bytes become unused.
    header["vocab_order"]["data_offsets"] = [start, end - 4]
    bad = _write(tmp_path / "bad.wordpack", header, data)
    with pytest.raises(WordPackError, match="vocab has 48 keys") as e:
        WordPack(bad)
    assert str(bad) in str(e.value)
