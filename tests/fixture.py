"""Synthetic source embedding and language folder for tests.

Run as a script, `python tests/fixture.py OUT`, to write OUT/test.wordpack.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from wordpack_compiler.build import build, encode_vocab
from wordpack_compiler.reader import Vocab

# Row order of the fixture pack. Indices are relied on by the tokenizer tests.
KEYS = [
    "the", "i", "do", "n't", "'s", "'re", "'ve", "'ll", "'d", "'m",          # 0-9
    "ca", "wo", "it", "is", "a", "dog", "dogs", "cat",                         # 10-17
    "well-known", "blue", "green", "new york", "café", "don",             # 18-23
    "covid-19", "covid", "4chan", "u.s.", "and/or", "and", "or",               # 24-30
    "it.", "don't", "e.g.", "shit", "dick", "duck", "never", "people",         # 31-38
    "like", "swim", "happens", "with", "in", "are", "they", "x", "b",          # 39-47
]


def vocab_from_keys(keys: list[str]) -> Vocab:
    return Vocab(*encode_vocab(keys))


DIMS = 16  # at least 10: the format stores 10 ABTT components
CS_EXTRA = ["Polska", "Polish", "polish", "czytały"]  # rows 48-51 in the cased pack
CLITICS = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
STOPWORDS = ["the", "a", "is", "i", "and", "or"]
MASKED = ["shit", "dick", "duck"]
SELFTEST = [
    "I don't like the dog. It's well-known!",
    "Covid-19 and 4chan, e.g., in the U.S. are blue-green?",
    "Sh*t happens *never with café https://x.org @bob 2024",
    "Dogs can't swim, won't they",
    "Another cafe\u0301 with well\u00adknown dog\u200bcat @bob\u200bdogs",
    "The well\u2010known dogs don\u02bct swim, ΟΣ",
    "I don't, I like it. Covid-19's 90's d**k",
]


def source_rows(keys: list[str]) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((len(keys), DIMS)).astype(np.float32)


def write_source(directory, case_sensitive: bool = False) -> tuple[Path, Path, Path]:
    """Write source.zip, recipe.toml and lang/ into `directory`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    keys = KEYS + (CS_EXTRA if case_sensitive else [])
    lines = [k + " " + " ".join(f"{v:.6f}" for v in r) for k, r in zip(keys, source_rows(keys))]
    lines.insert(23, "cafe\u0301 " + " ".join(["0.5"] * DIMS))  # NFC duplicate of row 22; dropped
    lines.append("broken 1 2 3")  # wrong number of fields; dropped
    lines.append("zero " + " ".join(["0"] * DIMS))  # zero row; dropped
    lines.append("<unk> " + " ".join(["0.25"] * DIMS))  # GloVe's unknown-word row; dropped
    zip_path = d / "source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("vectors.txt", "\n".join(lines) + "\n")

    lang = d / "lang"
    lang.mkdir(exist_ok=True)
    (lang / "language.toml").write_text(
        'code = "en"\nname = "Fixture"\nkeys = "surface"\n'
        f"case_sensitive = {'true' if case_sensitive else 'false'}\n"
        f"clitics = {json.dumps(CLITICS)}\n"
        "neighbor_filter = '.*\\d'\n"
        'stopwords_source = "fixture"\n',
        encoding="utf-8",
    )
    (lang / "stopwords.txt").write_text("\n".join(STOPWORDS) + "\n", encoding="utf-8")
    (lang / "do_not_read_this_file.txt").write_text("\n".join(MASKED) + "\n", encoding="utf-8")
    (lang / "selftest.txt").write_text("\n".join(SELFTEST) + "\n", encoding="utf-8")

    recipe = d / "recipe.toml"
    recipe.write_text(
        f'name = "test{"-cased" if case_sensitive else ""}"\n'
        'description = "Synthetic test pack."\nurl = "file:///nonexistent"\n'
        f'dims = {DIMS}\nlicense = "CC0-1.0"\ncitation = "none"\n',
        encoding="utf-8",
    )
    return recipe, lang, zip_path


def build_fixture(directory, case_sensitive: bool = False) -> Path:
    recipe, lang, zip_path = write_source(directory, case_sensitive)
    out = Path(directory) / "test.wordpack"
    build(recipe, lang, zip_path, out, 1)
    return out


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        shutil.move(build_fixture(work), out_dir / "test.wordpack")
    print(out_dir / "test.wordpack")
