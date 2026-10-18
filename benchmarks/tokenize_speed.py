"""Wall-clock time of tokenizing a corpus with a wordpack, natively or under Pyodide.

Two stages, each on a freshly opened pack (the key-to-row cache starts empty):
`tokenize`, the reader alone (`pack.tokenize` per text), and `corpus`,
ssdiff's `Corpus(texts, embeddings=emb)`, which adds stopwords, per-sentence
lists and flat docs. The first run is discarded; report the minimum of the rest.

    python benchmarks/tokenize_speed.py build/en-glove2024-wikigiga-1.wordpack

The texts are the 39,565 comments of `measuring_hate_speech`, read from the
parquet once and cached as JSON, so a Pyodide run (no pyarrow) reads the cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXTS = REPO / "build" / "bench" / "cache" / "hate_speech_texts.json"


def load_texts() -> list[str]:
    if not TEXTS.exists():
        sys.path.insert(0, str(REPO / "benchmarks"))
        from run import load_hate_speech

        TEXTS.parent.mkdir(parents=True, exist_ok=True)
        TEXTS.write_text(json.dumps(load_hate_speech()[0]), encoding="utf-8")
    return json.loads(TEXTS.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--runs", type=int, default=4, help="runs per stage; the first is discarded")
    ap.add_argument("--out", default=str(REPO / "build" / "bench" / "tokenize"))
    a = ap.parse_args()

    import ssdiff
    from ssdiff import Corpus, Embeddings
    from ssdiff.wordpack.reader import WordPack

    texts = load_texts()
    n_chars = sum(map(len, texts))

    def stage_tokenize():
        with WordPack(a.pack) as pack:
            t = time.perf_counter()
            toks = [pack.tokenize(s) for s in texts]
            return time.perf_counter() - t, sum(len(x.keys) for x in toks)

    def stage_corpus():
        emb = Embeddings.load(a.pack)
        t = time.perf_counter()
        corpus = Corpus(texts, embeddings=emb)
        return time.perf_counter() - t, sum(map(len, corpus.docs))

    rows = []
    for name, fn in (("tokenize", stage_tokenize), ("corpus", stage_corpus)):
        for run in range(a.runs):
            secs, n_tokens = fn()
            rows.append({"stage": name, "run": run, "seconds": secs, "n_tokens": n_tokens})
            print(f"{name} run {run}: {secs:.2f} s, {n_tokens:,} tokens", file=sys.stderr, flush=True)

    meta = {
        "date": date.today().isoformat(),
        "platform": sys.platform,
        "python": platform.python_version(),
        "ssdiff": ssdiff.__version__,
        "pack": Path(a.pack).name,
        "n_texts": len(texts),
        "n_chars": n_chars,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{meta['date']}-{meta['platform']}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*meta, *rows[0]])
        w.writeheader()
        for r in rows:
            w.writerow({**meta, **r})

    for name in ("tokenize", "corpus"):
        kept = [r for r in rows if r["stage"] == name][1:]
        best = min(kept, key=lambda r: r["seconds"])
        # n_tokens is every key for `tokenize` but only the kept tokens for
        # `corpus` (stopwords dropped), so the rates compare texts and chars.
        print(f"{name}: min {best['seconds']:.2f} s over runs 1..{a.runs - 1}, "
              f"{len(texts) / best['seconds']:,.0f} texts/s, "
              f"{n_chars / best['seconds'] / 1e6:.2f} M chars/s, {best['n_tokens']:,} tokens")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
