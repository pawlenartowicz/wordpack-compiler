"""Build a .wordpack from a recipe, a language folder and a source zip."""

from __future__ import annotations

import functools
import json
import os
import re
import struct
import sys
import time
import tomllib
import unicodedata
from collections import Counter
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

from wordpack_compiler.progress import Progress, format_count, format_size, format_time
from wordpack_compiler.reader import FORMAT_VERSION, Chars, Vocab, dequantize, make_selftest, tokenize
from wordpack_compiler.source import read_glove_zip


def encode_vocab(keys: list[str]) -> tuple[bytes, np.ndarray]:
    """The `vocab` bytes and `vocab_order`: row ids sorted by UTF-8 key bytes."""
    blob = "".join(k + "\n" for k in keys).encode("utf-8")
    encoded = [k.encode("utf-8") for k in keys]
    order = np.array(sorted(range(len(keys)), key=encoded.__getitem__), dtype=np.int32)
    return blob, order


N_COMPONENTS = 10
BLOCK = 100_000


def quantize(unit: np.ndarray) -> np.ndarray:
    """`int8-rowmax`: each unit row's largest |component| maps to 127; -128 is never written."""
    scale = 127.0 / np.abs(unit).max(axis=1, keepdims=True)
    return np.rint(unit * scale).astype(np.int8)


def compute_abtt(q: np.ndarray, progress: Progress | None = None) -> np.ndarray:
    """Mean and top principal components of the rows a reader sees at m = 0.

    Computed from the stored int8 rows, not the float source, so reading with
    ABTT m equals ssdiff's normalize(l2=True, abtt=m) on the m = 0 matrix.
    Accumulates in float64 in blocks, so the matrix is never held as floats.
    """
    d = q.shape[1]
    if d < N_COMPONENTS:
        raise ValueError(f"need at least {N_COMPONENTS} dimensions for ABTT, got {d}")
    total = np.zeros(d)
    cross = np.zeros((d, d))
    n_blocks = -(-len(q) // BLOCK)
    for i, start in enumerate(range(0, len(q), BLOCK), 1):
        if progress:
            progress.update(f"chunk {i} / {n_blocks}")
        x = dequantize(q[start : start + BLOCK], None, 0).astype(np.float64)
        total += x.sum(axis=0)
        cross += x.T @ x
    mean = total / len(q)
    _, vecs = np.linalg.eigh(cross / len(q) - np.outer(mean, mean))
    top = vecs[:, ::-1][:, :N_COMPONENTS].T
    return np.ascontiguousarray(np.vstack([mean, top]), dtype=np.float32)


# Appended to every pack's self-test (FORMAT.md, Self-test): only U+000A ends a
# sentence, and selftest.txt cannot hold a line feed inside a text.
LINE_BREAK_TEXTS = [
    "one\ntwo",
    "one\r\ntwo",
    "one.\n\n \t\ntwo",
    "one\rtwo\x0bthree\x0cfour\x85five\u2028six\u2029seven\x1cend",
]

_APOSTROPHES = "‘’ʼ"
_HYPHENS = "‐‑"


@functools.cache
def char_table() -> dict:
    """The `chars` table from this Python's Unicode data (FORMAT.md, Characters).

    Cached: scanning all code points takes about a second. Callers that change
    the table must copy it first.
    """
    letter = re.compile(r"[^\W\d_]")
    cps = range(sys.maxunicode + 1)
    fold = {str(cp): "" for cp in cps if unicodedata.category(chr(cp)) == "Cf"}
    fold[str(0x200B)] = " "  # ZERO WIDTH SPACE separates words, unlike the other format characters
    fold.update({str(ord(ch)): "'" for ch in _APOSTROPHES})
    fold.update({str(ord(ch)): "-" for ch in _HYPHENS})
    return {
        "unicode_version": unicodedata.unidata_version,
        "letter": _ranges(lambda ch: letter.match(ch) is not None),
        "digit": _ranges(lambda ch: unicodedata.category(ch) == "Nd"),
        "space": _ranges(str.isspace),
        "lower": {str(cp): chr(cp).lower() for cp in cps if chr(cp).lower() != chr(cp)},
        "upper": {str(cp): chr(cp).upper() for cp in cps if chr(cp).upper() != chr(cp)},
        "fold": fold,
    }


def _ranges(pred) -> list[list[int]]:
    out: list[list[int]] = []
    start = None
    for cp in range(sys.maxunicode + 1):
        if pred(chr(cp)):
            if start is None:
                start = cp
        elif start is not None:
            out.append([start, cp - 1])
            start = None
    if start is not None:
        out.append([start, sys.maxunicode])
    return out


_DTYPE_NAMES = {np.dtype("<f4"): "F32", np.dtype("<i4"): "I32", np.dtype("i1"): "I8", np.dtype("u1"): "U8"}


def write_safetensors(path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """Write a safetensors file: 8-byte LE header length, JSON header padded to 8 bytes, raw data.

    Tensors are written in dict order with no gaps; FORMAT.md puts the 4-byte
    types first so they stay aligned.
    """
    header: dict = {"__metadata__": metadata}
    arrays = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        header[name] = {
            "dtype": _DTYPE_NAMES[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
        arrays.append(arr)
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for arr in arrays:
            f.write(arr.tobytes() if arr.size == 0 else memoryview(arr).cast("B"))


def _word_list(path: Path, chars: Chars) -> list[str]:
    # Normalized as the reader normalizes text (NFC, fold, then lowercase with
    # the pack's table), so a listed word matches what the reader produces.
    # Repeats are dropped: a masked word listed twice would match twice, and a
    # masked match needs exactly one word.
    words = [chars.lower(chars.fold(unicodedata.normalize("NFC", w.strip()))) for w in path.read_text("utf-8").splitlines() if w.strip()]
    return list(dict.fromkeys(words))


def _lines_tensor(words: list[str]) -> np.ndarray:
    return np.frombuffer("".join(w + "\n" for w in words).encode("utf-8"), dtype=np.uint8)


def kept_rows(source_zip, dims: int, counts: Counter):
    """Yield (NFC key, float32 row, norm) for each source row a pack keeps, in file order.

    `build()` and `benchmarks/run.py` both read rows through this, so a pack
    and its source embedding have the same vocabulary. Dropped rows are
    counted in `counts`.
    """
    seen: set[str] = set()
    for key, row in read_glove_zip(source_zip, dims, counts):
        key = unicodedata.normalize("NFC", key)
        # GloVe's averaged unknown-word vector. No token reaches it (`<` is not a
        # letter), but it would turn up as a nearest neighbour.
        if key == "<unk>":
            counts["<unk> row"] += 1
            continue
        if key in seen:
            counts["duplicate after NFC"] += 1
            continue
        norm = float(np.sqrt(np.dot(row.astype(np.float64), row)))
        # The norm is stored as float32, so one beyond its range is as unusable as inf.
        if norm == 0 or not norm <= float(np.finfo(np.float32).max):
            counts["zero or non-finite row"] += 1
            continue
        seen.add(key)
        yield key, row, norm


def build(recipe_path, language_dir, source_zip, out_path, version: int) -> dict:
    """Build one pack (FORMAT.md) and return a summary of what was read and dropped.

    `version` is the pack's whole-number version; it is written into the metadata.
    """
    started = time.perf_counter()
    recipe = tomllib.loads(Path(recipe_path).read_text("utf-8"))
    language_dir = Path(language_dir)
    lang = tomllib.loads((language_dir / "language.toml").read_text("utf-8"))
    chars = Chars(char_table())
    stopwords = _word_list(language_dir / "stopwords.txt", chars)
    masked_path = language_dir / "do_not_read_this_file.txt"
    masked = _word_list(masked_path, chars) if masked_path.exists() else []
    lines = [s for s in (language_dir / "selftest.txt").read_text("utf-8").split("\n") if s.strip()]
    dims = int(recipe["dims"])
    case_sensitive = bool(lang["case_sensitive"])
    clitics = list(lang["clitics"])

    counts: Counter = Counter()
    keys: list[str] = []
    norms: list[float] = []
    blocks = []
    block = np.empty((BLOCK, dims), dtype=np.int8)
    k = 0
    reading = Progress("[1/4] reading rows   ")
    reported = 0
    for key, row, norm in kept_rows(source_zip, dims, counts):
        if counts["lines"] // 100_000 > reported:
            reported = counts["lines"] // 100_000
            reading.update(f"{format_count(len(keys))} rows   {format_time(reading.elapsed())}")
        keys.append(key)
        norms.append(norm)
        block[k] = quantize((row / norm)[None, :])[0]
        k += 1
        if k == BLOCK:
            blocks.append(block)
            block = np.empty_like(block)
            k = 0
    blocks.append(block[:k])
    q = np.concatenate(blocks)
    del blocks
    if len(q) == 0:
        raise ValueError(f"{source_zip}: no usable rows")

    reading.done(f"{format_count(len(q))} rows   {format_time(reading.elapsed())}")

    abtt_progress = Progress("[2/4] computing ABTT ")
    abtt = compute_abtt(q, abtt_progress)
    abtt_progress.done(format_time(abtt_progress.elapsed()))

    testing = Progress("[3/4] self-test      ")
    testing.update("")
    blob, order = encode_vocab(keys)
    vocab = Vocab(blob, order)
    table = char_table()
    selftest = make_selftest(
        lines + LINE_BREAK_TEXTS,
        lambda text: tokenize(text, vocab, chars, case_sensitive, clitics, masked),
        lambda rows, m: dequantize(q[rows], abtt, m),
        len(q),
    )
    testing.done(f"{len(selftest['texts'])} texts   {format_time(testing.elapsed())}")

    writing = Progress("[4/4] writing file   ")
    writing.update("")
    metadata = {
        "format": "wordpack",
        "format_version": str(FORMAT_VERSION),
        "name": recipe["name"],
        "version": str(version),
        "description": recipe["description"],
        "source": json.dumps(
            {"url": recipe["url"], "license": recipe["license"], "citation": recipe["citation"]},
            ensure_ascii=False,
        ),
        "language": lang["code"],
        "keys": lang["keys"],
        "case_sensitive": "true" if case_sensitive else "false",
        "clitics": json.dumps(clitics, ensure_ascii=False),
        "neighbor_filter": lang["neighbor_filter"],
        "stopwords_source": lang["stopwords_source"],
        "quantization": "int8-rowmax",
        "compiler_version": package_version("wordpack-compiler"),
    }
    tensors = {
        "abtt": abtt,
        "norms": np.asarray(norms, dtype=np.float32),
        "vocab_order": order,
        "matrix": q,
        "vocab": np.frombuffer(blob, dtype=np.uint8),
        "stopwords": _lines_tensor(stopwords),
        "masked": _lines_tensor(masked),
        "chars": np.frombuffer(json.dumps(table, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
        "selftest": np.frombuffer(json.dumps(selftest, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    write_safetensors(part, tensors, metadata)
    os.replace(part, out_path)
    writing.done(f"{format_size(out_path.stat().st_size)}   {format_time(writing.elapsed())}")
    return {
        "counts": dict(counts),
        "uppercase keys": 0 if case_sensitive else sum(key != key.lower() for key in keys),
        "N": len(q),
        "D": dims,
        "file size": out_path.stat().st_size,
        "seconds": round(time.perf_counter() - started, 1),
    }
