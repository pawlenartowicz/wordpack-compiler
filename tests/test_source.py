import io
import zipfile
from collections import Counter

import numpy as np
import pytest

from wordpack_compiler.source import download, read_glove_zip


def _zip(tmp_path, data: bytes, members=("v.txt",)):
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, data)
    return path


def test_reads_rows_and_counts_bad_lines(tmp_path):
    data = (
        b"the 1 2 3\n"
        b"new york 4 5 6\n"
        b"short 1 2\n"
        b" 7 8 9\n"
        b"\xff\xfe 1 2 3\n"
        b"bad 1 x 3\n"
        b"last 0.5 -0.5 1e-3 \n"
    )
    counts = Counter()
    rows = list(read_glove_zip(_zip(tmp_path, data), 3, counts))
    assert [k for k, _ in rows] == ["the", "new york", "last"]
    assert rows[1][1].dtype == np.float32
    assert rows[2][1].tolist() == pytest.approx([0.5, -0.5, 1e-3])
    assert counts == Counter({"lines": 7, "malformed line": 2, "empty key": 1, "invalid UTF-8 key": 1})


def test_zip_must_hold_one_file(tmp_path):
    with pytest.raises(ValueError, match="one file"):
        list(read_glove_zip(_zip(tmp_path, b"a 1\n", members=("a.txt", "b.txt")), 1, Counter()))


def test_download_skips_existing_file(tmp_path):
    dest = tmp_path / "x.zip"
    dest.write_bytes(b"already here")
    download("http://invalid.invalid/x.zip", dest)
    assert dest.read_bytes() == b"already here"


def test_download_from_file_url(tmp_path, capsys):
    src = tmp_path / "remote.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "sources" / "remote.bin"
    download(src.as_uri(), dest)
    assert dest.read_bytes() == b"payload"
    assert not dest.with_name("remote.bin.part").exists()
    last = capsys.readouterr().err.splitlines()[-1]
    assert last.startswith("download  remote.bin  ")


def test_download_refuses_a_cut_download(tmp_path, monkeypatch):
    class Cut(io.BytesIO):
        headers = {"Content-Length": "1000"}

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: Cut(b"x" * 100))
    dest = tmp_path / "x.zip"
    with pytest.raises(OSError, match="100 of 1,000 bytes"):
        download("http://invalid.invalid/x.zip", dest)
    assert not dest.exists()


def test_dims_must_match_the_first_line(tmp_path):
    with pytest.raises(ValueError, match="dims = 2"):
        list(read_glove_zip(_zip(tmp_path, b"the 1 2 3\n"), 2, Counter()))
