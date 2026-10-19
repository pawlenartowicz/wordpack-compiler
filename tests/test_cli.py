import json
import shutil
from pathlib import Path

import pytest

import fixture
from wordpack_compiler import cli
from wordpack_compiler.build import build
from wordpack_compiler.cli import format_summary, main
from wordpack_compiler.reader import WordPack, read_header

REPO = Path(__file__).resolve().parents[1]


def test_info_prints_and_passes(pack_path, capsys):
    assert main(["info", str(pack_path)]) == 0
    out = capsys.readouterr().out
    assert "en-fixture" in out and "matrix" in out and "self-test: passed" in out


def test_info_fails_on_corrupt_pack(pack_path, tmp_path, capsys):
    bad = tmp_path / "bad.wordpack"
    shutil.copy(pack_path, bad)
    with open(bad, "rb") as f:
        base, tensors, _ = read_header(f, str(bad))
    raw = bytearray(bad.read_bytes())
    pos = base + tensors["matrix"]["data_offsets"][0]
    raw[pos] = 0 if raw[pos] != 0 else 100
    bad.write_bytes(bytes(raw))
    assert main(["info", str(bad)]) == 1
    assert "self-test failed" in capsys.readouterr().out


def test_english_language_folder_builds(tmp_path):
    recipe, _, zip_path = fixture.write_source(tmp_path)
    out = tmp_path / "en.wordpack"
    build(recipe, REPO / "languages" / "en", zip_path, out, 1)
    lines = [s for s in (REPO / "languages/en/selftest.txt").read_text("utf-8").split("\n") if s.strip()]
    with WordPack(out) as pack:
        pack.selftest()
        assert len(pack._selftest["texts"]) == len(lines) + 4
        assert pack.masked == ["fuck", "fucking", "shit", "bitch", "dick", "ass", "fucks", "motherfucker", "nigga", "pussy"]
        assert pack.meta["stopwords_source"].startswith("spaCy 3.8.13")
        assert pack.meta["clitics"] == ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
    assert "café" in (REPO / "languages/en/selftest.txt").read_text("utf-8")


def test_info_on_truncated_file_prints_error(pack_path, tmp_path, capsys):
    cut = tmp_path / "cut.wordpack"
    cut.write_bytes(pack_path.read_bytes()[:-100])
    assert main(["info", str(cut)]) == 1
    assert "truncated" in capsys.readouterr().out


def test_info_on_a_missing_path_prints_an_error(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing.wordpack")]) == 1
    assert "missing.wordpack" in capsys.readouterr().out


def test_build_of_a_missing_recipe_prints_an_error(capsys):
    assert main(["build", "en/no-such-recipe"]) == 1
    assert "no-such-recipe.toml not found" in capsys.readouterr().out


def test_build_argument_without_slash_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        main(["build", "en"])
    assert e.value.code == 2
    assert "<lang>/<recipe>" in capsys.readouterr().err


def _repo(directory, packs: dict) -> Path:
    """A stand-in repository: languages/en/fix.toml, its source already downloaded, and index.json."""
    recipe, lang, zip_path = fixture.write_source(directory / "src")
    en = directory / "languages" / "en"
    shutil.copytree(lang, en)
    shutil.copy(recipe, en / "fix.toml")
    sources = directory / "build" / "sources"
    sources.mkdir(parents=True)
    # The recipe's URL ends in "nonexistent"; `build` skips a download whose file is already there.
    shutil.copy(zip_path, sources / "nonexistent")
    (directory / "index.json").write_text(json.dumps({"index_version": 1, "packs": packs}), "utf-8")
    return directory


@pytest.mark.parametrize(
    "packs, expected",
    [
        ({"pl-other": {"versions": {"7": {}}}}, 1),
        ({"en-fixture": {"versions": {"1": {}, "2": {}, "10": {}}}}, 11),
    ],
)
def test_build_takes_the_next_version_from_the_index(tmp_path, monkeypatch, packs, expected):
    monkeypatch.setattr(cli, "REPO", _repo(tmp_path, packs))
    assert main(["build", "en/fix"]) == 0
    with WordPack(tmp_path / "build" / f"en-fixture-{expected}.wordpack") as pack:
        assert pack.meta["version"] == str(expected)


def test_build_refuses_a_recipe_name_that_reads_as_a_full_name(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, {})
    recipe = repo / "languages" / "en" / "fix.toml"
    recipe.write_text(recipe.read_text("utf-8").replace('"en-fixture"', '"en-fixture-2"'), "utf-8")
    monkeypatch.setattr(cli, "REPO", repo)
    assert main(["build", "en/fix"]) == 1
    assert "en-fixture-2" in capsys.readouterr().out
    assert not list((repo / "build").glob("*.wordpack"))


def test_build_names_an_index_that_is_not_valid_json(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, {})
    (repo / "index.json").write_text("{", "utf-8")
    monkeypatch.setattr(cli, "REPO", repo)
    assert main(["build", "en/fix"]) == 1
    assert "index.json" in capsys.readouterr().out


def test_format_summary_aligns_numbers_and_shortens_reasons():
    summary = {
        "counts": {
            "lines": 1291147,
            "malformed line": 3,
            "duplicate key": 8,
            "zero or non-finite row": 1,
            "empty key": 0,
            "odd new reason": 2,
        },
        "N": 1291133,
        "D": 300,
        "file size": 412_000_000,
        "seconds": 161.0,
    }
    assert format_summary("build/x.wordpack", summary) == (
        "built build/x.wordpack\n"
        "  rows read                   1,291,147\n"
        "  rows dropped                       14\n"
        "    malformed line                    3\n"
        "    duplicate key                     8\n"
        "    zero or non-finite                1\n"
        "    odd new reason                    2\n"
        "  N × D                 1,291,133 × 300\n"
        "  file size                      412 MB\n"
        "  time                             2:41"
    )
