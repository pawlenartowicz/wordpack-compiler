import hashlib
import json
import struct
import types

import pytest

import fixture
from wordpack_compiler import cli
from wordpack_compiler.build import build
from wordpack_compiler.cli import main
from wordpack_compiler.publish import PublishError, publish
from wordpack_compiler.reader import WordPackError, read_header

PACK_KEY = "wordpacks/en-fixture-1.wordpack"
INDEX_KEY = "wordpacks/index.json"
OTHER = {
    "language": "pl",
    "description": "Another pack.",
    "license": "CC0-1.0",
    "versions": {"3": {"file": "pl-other-3.wordpack", "size": 5, "sha256": "ab", "format_version": 1}},
}


class FakeS3:
    """Stands in for the boto3 S3 client: keeps objects in a dict and records uploads in order."""

    class ClientError(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    def __init__(self):
        self.exceptions = types.SimpleNamespace(ClientError=self.ClientError)
        self.objects = {}  # key -> (bytes, metadata)
        self.puts = []

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.ClientError("404")
        body, metadata = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": metadata}

    def put_object(self, Bucket, Key, Body, ContentType, Metadata=None):
        self.objects[Key] = (Body if isinstance(Body, bytes) else Body.read(), Metadata or {})
        self.puts.append(Key)


def write_repo(directory, packs=None):
    (directory / "index.json").write_text(json.dumps({"index_version": 1, "packs": packs or {}}), "utf-8")
    return directory


def version_entry(pack_path) -> dict:
    data = pack_path.read_bytes()
    return {
        "file": "en-fixture-1.wordpack",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format_version": 1,
    }


def fixture_pack(pack_path) -> dict:
    return {
        "language": "en",
        "description": "Synthetic test pack.",
        "license": "CC0-1.0",
        "versions": {"1": version_entry(pack_path)},
    }


def test_publish_uploads_the_pack_then_the_index(pack_path, tmp_path):
    repo = write_repo(tmp_path, packs={"pl-other": OTHER})
    s3 = FakeS3()
    publish(pack_path, s3, "bucket", repo)
    assert s3.puts == [PACK_KEY, INDEX_KEY]
    assert s3.objects[PACK_KEY] == (pack_path.read_bytes(), {"sha256": version_entry(pack_path)["sha256"]})
    assert json.loads((repo / "index.json").read_text("utf-8")) == {
        "index_version": 1,
        "packs": {"pl-other": OTHER, "en-fixture": fixture_pack(pack_path)},
    }
    assert s3.objects[INDEX_KEY][0] == (repo / "index.json").read_bytes()


def test_publish_skips_the_upload_when_the_same_pack_is_in_the_bucket(pack_path, tmp_path):
    repo = write_repo(tmp_path)
    s3 = FakeS3()
    s3.objects[PACK_KEY] = (pack_path.read_bytes(), {"sha256": version_entry(pack_path)["sha256"]})
    publish(pack_path, s3, "bucket", repo)
    assert s3.puts == [INDEX_KEY]
    assert json.loads((repo / "index.json").read_text("utf-8"))["packs"] == {"en-fixture": fixture_pack(pack_path)}


def test_publish_refuses_a_pack_that_fails_its_self_test(pack_path, tmp_path):
    with open(pack_path, "rb") as f:
        base, tensors, _ = read_header(f, str(pack_path))
    raw = bytearray(pack_path.read_bytes())
    pos = base + tensors["matrix"]["data_offsets"][0]  # row 0, component 0
    raw[pos] = 0 if raw[pos] != 0 else 100
    bad = tmp_path / "bad.wordpack"
    bad.write_bytes(bytes(raw))
    repo = write_repo(tmp_path)
    before = (repo / "index.json").read_bytes()
    s3 = FakeS3()
    with pytest.raises(WordPackError, match="self-test failed"):
        publish(bad, s3, "bucket", repo)
    assert s3.puts == []
    assert (repo / "index.json").read_bytes() == before


@pytest.mark.parametrize("stored", ["other size", "same size, other sha256", "same size, no sha256"])
def test_publish_never_replaces_a_published_file(pack_path, tmp_path, stored):
    data = pack_path.read_bytes()
    s3 = FakeS3()
    s3.objects[PACK_KEY] = {
        "other size": (data + b"x", {"sha256": version_entry(pack_path)["sha256"]}),
        "same size, other sha256": (data, {"sha256": "0" * 64}),
        "same size, no sha256": (data, {}),
    }[stored]
    repo = write_repo(tmp_path)
    before = (repo / "index.json").read_bytes()
    with pytest.raises(PublishError, match="never replaced"):
        publish(pack_path, s3, "bucket", repo)
    assert s3.puts == []
    assert (repo / "index.json").read_bytes() == before


def test_publish_with_no_file_uploads_only_the_index(pack_path, tmp_path):
    repo = write_repo(tmp_path, packs={"en-fixture": fixture_pack(pack_path)})
    before = (repo / "index.json").read_bytes()
    s3 = FakeS3()
    publish(None, s3, "bucket", repo)
    assert s3.puts == [INDEX_KEY]
    assert (repo / "index.json").read_bytes() == before


def test_publish_without_its_config_file_names_the_file_and_the_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "REPO", write_repo(tmp_path))
    assert main(["publish"]) == 1
    out = capsys.readouterr().out
    assert ".publish.toml" in out
    assert all(key in out for key in ["endpoint", "bucket", "access_key_id", "secret_access_key"])


def test_publish_with_an_incomplete_config_names_the_missing_key_and_not_the_secret(tmp_path, monkeypatch, capsys):
    (tmp_path / ".publish.toml").write_text(
        'endpoint = "https://x.example"\naccess_key_id = "id"\nsecret_access_key = "s3cr3t-value"\n', "utf-8"
    )
    monkeypatch.setattr(cli, "REPO", write_repo(tmp_path))
    assert main(["publish"]) == 1
    captured = capsys.readouterr()
    assert "bucket" in captured.out
    assert "s3cr3t-value" not in captured.out + captured.err


class ForbiddenS3(FakeS3):
    def head_object(self, Bucket, Key):
        raise self.ClientError("403")


class UnreachableS3(FakeS3):
    def head_object(self, Bucket, Key):
        from botocore.exceptions import EndpointConnectionError

        raise EndpointConnectionError(endpoint_url="https://x.example")


def publish_from(tmp_path, monkeypatch, s3) -> None:
    """Make `main(["publish", ...])` publish from a repo in `tmp_path` through `s3`."""
    # The command catches botocore's errors, which the stand-in client does not bring.
    pytest.importorskip("botocore")
    (tmp_path / ".publish.toml").write_text(
        'endpoint = "https://x.example"\nbucket = "b"\naccess_key_id = "id"\nsecret_access_key = "s"\n', "utf-8"
    )
    monkeypatch.setattr(cli, "REPO", write_repo(tmp_path))
    monkeypatch.setattr(cli, "make_client", lambda config: s3)


@pytest.mark.parametrize(
    "file, s3_class",
    [
        ("missing.wordpack", FakeS3),
        ("index.json", FakeS3),
        ("other.safetensors", FakeS3),
        (None, ForbiddenS3),
        (None, UnreachableS3),
    ],
)
def test_publish_prints_an_ordinary_mistake_and_uploads_nothing(
    pack_path, tmp_path, monkeypatch, capsys, file, s3_class
):
    """A missing path, a file that is not safetensors, a safetensors file that is not a wordpack,
    a bucket that answers 403, and a bucket that cannot be reached."""
    s3 = s3_class()
    publish_from(tmp_path, monkeypatch, s3)
    # A safetensors file with an empty header.
    (tmp_path / "other.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    assert main(["publish", str(tmp_path / file) if file else str(pack_path)]) == 1
    assert capsys.readouterr().out.strip()
    assert s3.puts == []


def test_publish_names_an_index_that_is_not_valid_json(tmp_path, monkeypatch, capsys):
    s3 = FakeS3()
    publish_from(tmp_path, monkeypatch, s3)
    (tmp_path / "index.json").write_text("{", "utf-8")
    assert main(["publish"]) == 1
    assert "index.json" in capsys.readouterr().out
    assert s3.puts == []


def test_publish_refuses_a_pack_whose_version_is_not_a_whole_number(tmp_path):
    recipe, lang, zip_path = fixture.write_source(tmp_path / "src")
    pack = tmp_path / "old.wordpack"
    build(recipe, lang, zip_path, pack, "0.1.0")
    repo = write_repo(tmp_path)
    before = (repo / "index.json").read_bytes()
    s3 = FakeS3()
    with pytest.raises(PublishError, match=r"version '0\.1\.0'"):
        publish(pack, s3, "bucket", repo)
    assert s3.puts == []
    assert (repo / "index.json").read_bytes() == before
