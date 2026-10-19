"""Download a source embedding and stream GloVe text rows out of its zip."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from wordpack_compiler.progress import Progress, format_size, format_time

_NO_LENGTH_MILESTONE = 500_000_000  # bytes between log lines when the size is unknown


def _format_speed(bytes_per_second: float) -> str:
    mb = bytes_per_second / 1e6
    return f"{mb:.1f} MB/s" if mb < 10 else f"{mb:,.0f} MB/s"


def download(url: str, dest: Path) -> None:
    """Fetch `url` to `dest` through `dest.part`; skip when `dest` exists. No checksum."""
    dest = Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    progress = Progress(f"download  {dest.name}  ")
    done, milestones = 0, 0
    # urllib applies the timeout to each blocking operation (connect, each read),
    # not to the whole download, so a slow but live download is not cut off.
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
        try:
            total = int(response.headers.get("Content-Length") or 0)
        except ValueError:
            total = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            speed = done / max(progress.elapsed(), 1e-9)
            if total:
                done_text, total_text = format_size(done), format_size(total)
                if done_text[-2:] == total_text[-2:]:
                    done_text = done_text[:-3]
                text = (
                    f"{done_text} / {total_text}  {done / total:.0%}  "
                    f"{_format_speed(speed)}  {format_time(max(0, (total - done) / speed))} left"
                )
                reached = done * 10 // total
            else:
                text = f"{format_size(done)}  {_format_speed(speed)}"
                reached = done // _NO_LENGTH_MILESTONE
            progress.update(text, milestone=reached > milestones)
            milestones = reached
    # A dropped connection ends the read loop the same way the end of the file does.
    if total and done != total:
        raise OSError(f"{url}: download stopped at {done:,} of {total:,} bytes")
    progress.done(f"{format_size(done)}  {format_time(progress.elapsed())}")
    os.replace(part, dest)


def read_glove_zip(path, dims: int, counts: Counter):
    """Yield (key, float32 row) from the one text file in a GloVe zip, in file order.

    Lines split from the right, so a key containing a space survives. The first
    line must hold `dims` numbers. Skipped lines are counted in `counts`.
    """
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one file in the zip, found {len(members)}")
        with zf.open(members[0]) as fh:
            for raw in fh:
                counts["lines"] += 1
                # GloVe's first key has no space, so its line shows the file's dims.
                if counts["lines"] == 1 and raw.rstrip().count(b" ") != dims:
                    raise ValueError(f"{path}: the first line has {raw.rstrip().count(b' ')} numbers; the recipe says dims = {dims}")
                parts = raw.rstrip().rsplit(b" ", dims)
                if len(parts) != dims + 1:
                    counts["malformed line"] += 1
                    continue
                if not parts[0].strip():
                    counts["empty key"] += 1
                    continue
                try:
                    key = parts[0].decode("utf-8")
                except UnicodeDecodeError:
                    counts["invalid UTF-8 key"] += 1
                    continue
                try:
                    row = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    counts["malformed line"] += 1
                    continue
                yield key, row
