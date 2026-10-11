"""`index.json`, `tags.json` and publishing packs to the bucket."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

from wordpack_compiler.progress import format_size
from wordpack_compiler.reader import WordPack, read_header

CONFIG_KEYS = ("endpoint", "bucket", "access_key_id", "secret_access_key")
PREFIX = "wordpacks/"


class PublishError(Exception):
    """`publish` stopped; the message says why."""


def is_name(text: str) -> bool:
    """Whether `text` can be a pack name or a short tag.

    Lowercase letters, digits and hyphens, not ending in a hyphen followed
    only by digits: that ending marks a full name, `<name>-<version>`.
    """
    return re.fullmatch(r"[a-z0-9-]+", text) is not None and re.search(r"-[0-9]+$", text) is None


def next_version(index: dict, name: str) -> int:
    """One more than the highest version `index` lists for `name`, or 1 when it lists none."""
    versions = index["packs"].get(name, {}).get("versions", {})
    return max(map(int, versions), default=0) + 1


def load_config(repo: Path) -> dict:
    """The bucket settings from `<repo>/.publish.toml`. The file holds a secret and is never printed."""
    path = repo / ".publish.toml"
    if not path.exists():
        raise PublishError(f"{path} not found. It must set: {', '.join(CONFIG_KEYS)}")
    config = tomllib.loads(path.read_text("utf-8"))
    missing = [key for key in CONFIG_KEYS if not config.get(key)]
    if missing:
        raise PublishError(f"{path} does not set: {', '.join(missing)}")
    return config


def make_client(config: dict):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise PublishError("publishing needs boto3: pip install -e .[publish]") from e
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key_id"],
        aws_secret_access_key=config["secret_access_key"],
        region_name="auto",
        # Send each upload as one plain PUT. By default boto3 adds a checksum
        # trailer in a chunked request encoding, which an S3-compatible store
        # may not accept.
        config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required"),
    )


def check_tags(tags: dict, index: dict) -> list[str]:
    """What is wrong with `tags`: a tag that breaks the naming rule, or one whose full name `index` does not list."""
    listed = {f"{name}-{version}" for name, pack in index["packs"].items() for version in pack["versions"]}
    problems = [f"tag {tag!r} breaks the naming rule" for tag in tags if not is_name(tag)]
    problems += [
        f"tag {tag!r} stands for {full!r}, which index.json does not list"
        for tag, full in tags.items()
        if full not in listed
    ]
    return problems


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except ValueError as e:
        raise PublishError(f"{path} is not valid JSON: {e}") from e


def _publish_pack(path: Path, client, bucket: str, index: dict) -> None:
    """Upload the pack unless the bucket already holds it, and add its version to `index`."""
    with open(path, "rb") as f:
        _, _, meta = read_header(f, str(path))
        if meta.get("format") != "wordpack":
            raise PublishError(f"{path} is not a wordpack")
        f.seek(0)
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    size = path.stat().st_size
    name, version = meta["name"], meta["version"]
    if re.fullmatch(r"[0-9]+", version) is None:
        raise PublishError(f"{path}: version {version!r} is not a whole number. Build the pack again.")
    # A published file is never replaced, so a pack that fails its self-test stops here.
    with WordPack(path) as pack:
        pack.selftest()
    file = f"{name}-{version}.wordpack"
    key = PREFIX + file
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise
        head = None
    if head is None:
        print(f"uploading {file} ({format_size(size)})", file=sys.stderr)
        with open(path, "rb") as f:
            # One request, so the bucket stores the metadata with the object:
            # the SHA-256 is how a later run tells this file from another of the same size.
            client.put_object(
                Bucket=bucket, Key=key, Body=f, ContentType="application/octet-stream", Metadata={"sha256": sha256}
            )
    elif head["ContentLength"] == size and head.get("Metadata", {}).get("sha256") == sha256:
        print(f"{file} is already in the bucket; upload skipped", file=sys.stderr)
    else:
        raise PublishError(
            f"{key} is already in the bucket with other contents. "
            "A published file is never replaced: build a new version."
        )
    versions = index["packs"].get(name, {}).get("versions", {})
    index["packs"][name] = {
        "language": meta["language"],
        "description": meta["description"],
        "license": json.loads(meta["source"])["license"],
        "versions": {
            **versions,
            version: {"file": file, "size": size, "sha256": sha256, "format_version": int(meta["format_version"])},
        },
    }


def publish(pack_path: Path | None, client, bucket: str, repo: Path) -> None:
    """Upload `pack_path`, then `index.json` and `tags.json` from `repo`; with no pack, only the two JSON files.

    The pack goes first and the tags last, so the published index never lists
    a file that is not there, and no published tag stands for a pack the
    index does not list.
    """
    index_path = repo / "index.json"
    index = _read_json(index_path)
    if pack_path is not None:
        _publish_pack(Path(pack_path), client, bucket, index)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", "utf-8")
    problems = check_tags(_read_json(repo / "tags.json"), index)
    if problems:
        raise PublishError("tags.json: " + "; ".join(problems))
    for name in ("index.json", "tags.json"):
        client.put_object(
            Bucket=bucket, Key=PREFIX + name, Body=(repo / name).read_bytes(), ContentType="application/json"
        )
        print(f"uploaded {name}", file=sys.stderr)
