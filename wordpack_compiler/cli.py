"""Command line: `wordpack-compiler build <lang>/<recipe>`, `info <file>` and `publish [file]`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

from wordpack_compiler.build import build
from wordpack_compiler.progress import format_count, format_size, format_time
from wordpack_compiler.publish import PublishError, is_name, load_config, make_client, next_version, publish
from wordpack_compiler.reader import WordPack, WordPackError, read_header
from wordpack_compiler.source import download

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wordpack-compiler")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="build a pack from languages/<lang>/<recipe>.toml")
    p_build.add_argument("pack", help="<lang>/<recipe>, e.g. en/glove2024-dolma")
    p_info = sub.add_parser("info", help="print a pack's contents and run its self-test")
    p_info.add_argument("file")
    p_publish = sub.add_parser(
        "publish", help="upload a pack, then index.json and tags.json; with no file, only the two JSON files"
    )
    p_publish.add_argument("file", nargs="?")
    args = parser.parse_args(argv)
    if args.command == "build":
        if "/" not in args.pack:
            parser.error("pack must be <lang>/<recipe>, e.g. en/glove2024-dolma")
        return _build(args.pack)
    if args.command == "publish":
        return _publish(args.file)
    return _info(args.file)


def _build(pack: str) -> int:
    lang, recipe_name = pack.split("/", 1)
    language_dir = REPO / "languages" / lang
    recipe_path = language_dir / f"{recipe_name}.toml"
    if not recipe_path.exists():
        print(f"{recipe_path} not found")
        return 1
    recipe = tomllib.loads(recipe_path.read_text("utf-8"))
    name = recipe["name"]
    if not is_name(name):
        print(
            f"{recipe_path}: name {name!r} must be made of lowercase letters, digits and hyphens, "
            "and must not end in a hyphen followed only by digits"
        )
        return 1
    try:
        index = json.loads((REPO / "index.json").read_text("utf-8"))
    except (OSError, ValueError) as e:
        print(f"{REPO / 'index.json'}: {e}")
        return 1
    version = next_version(index, name)
    source = REPO / "build" / "sources" / recipe["url"].rsplit("/", 1)[-1]
    print(f"{name}-{version}", file=sys.stderr)
    download(recipe["url"], source)
    out = REPO / "build" / f"{name}-{version}.wordpack"
    summary = build(recipe_path, language_dir, source, out, version)
    print(file=sys.stderr)
    print(format_summary(out, summary))
    return 0


def _publish(file: str | None) -> int:
    try:
        config = load_config(REPO)
        client = make_client(config)
    except PublishError as e:
        print(e)
        return 1
    # make_client stops the command when boto3, and with it botocore, is not installed.
    from botocore.exceptions import BotoCoreError

    try:
        publish(file, client, config["bucket"], REPO)
    # BotoCoreError: the bucket gave no answer. ClientError: its answer is an error.
    except (PublishError, WordPackError, OSError, BotoCoreError, client.exceptions.ClientError) as e:
        print(e)
        return 1
    return 0


_REASON_NAMES = {"duplicate after NFC": "NFC duplicate", "zero or non-finite row": "zero or non-finite"}


def format_summary(out_path, summary: dict) -> str:
    """The text printed after a build: counts, size and time, numbers right-aligned in one column."""
    counts = summary["counts"]
    dropped = {k: v for k, v in counts.items() if k != "lines"}
    rows = [
        (2, "rows read", format_count(counts["lines"])),
        (2, "rows dropped", format_count(sum(dropped.values()))),
        *[(4, _REASON_NAMES.get(k, k), format_count(v)) for k, v in dropped.items() if v > 0],
        (2, "uppercase keys", format_count(summary["uppercase keys"])),
        (2, "N × D", f"{format_count(summary['N'])} × {summary['D']}"),
        (2, "file size", format_size(summary["file size"])),
        (2, "time", format_time(summary["seconds"])),
    ]
    label_w = max(indent + len(label) for indent, label, _ in rows)
    value_w = max(len(value) for _, _, value in rows)
    lines = [f"built {out_path}"]
    lines += [f"{' ' * indent}{label}".ljust(label_w) + "  " + value.rjust(value_w) for indent, label, value in rows]
    return "\n".join(lines)


def _info(path: str) -> int:
    try:
        with open(path, "rb") as f:
            _, tensors, _ = read_header(f, path)
        pack = WordPack(path)
    except (WordPackError, OSError) as e:
        print(e)
        return 1
    with pack:
        for key, value in pack.meta.items():
            print(f"{key}: {value}")
        for name, entry in tensors.items():
            print(f"tensor {name}: {entry['dtype']} {entry['shape']}")
        print(f"N: {pack.n}  D: {pack.dims}  file size: {os.path.getsize(path):,} bytes")
        try:
            pack.selftest()
        except WordPackError as e:
            print(e)
            return 1
    print("self-test: passed")
    return 0
