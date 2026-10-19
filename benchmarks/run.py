"""Coverage and cross-validated R² for each English wordpack next to its source embedding.

Each pack is compared with its own GloVe source run through ssdiff's spaCy
route (en_core_web_lg lemmas, ssdiff's stopwords), so a difference between the
two comes from the pack pipeline alone.

    python benchmarks/run.py warriner hate_speech
"""

from __future__ import annotations

import argparse
import functools
import gc
import os
import pickle
import re
import sys
import tomllib
from collections import Counter
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from ssdiff import SSD, Corpus, Embeddings
from ssdiff.lang_config import lang_to_model

from wordpack_compiler import build as build_module, source as source_module
from wordpack_compiler.build import kept_rows

REPO = Path(__file__).resolve().parents[1]
SEED = 2137
N_FOLDS = 10
SPACY_MODEL = lang_to_model("en")

MODELS = {
    "pls-auto": lambda ssd: ssd.fit_pls(k="auto", test=False),
    "pls-1": lambda ssd: ssd.fit_pls(k=1, test=False),
    "pls-3": lambda ssd: ssd.fit_pls(k=3, test=False),
    "ols-auto": lambda ssd: ssd.fit_ols(),
    "ols-20": lambda ssd: ssd.fit_ols(fixed_k=20),
    "ols-40": lambda ssd: ssd.fit_ols(fixed_k=40),
}
# The auto models are left out unless named with --models. Each call is
# hundreds of PLS fits (pls-auto's split search) or 60 PCA K values with
# k-means per side (ols-auto's sweep), so they take most of a run's time.
DEFAULT_MODELS = ["pls-1", "pls-3", "ols-20", "ols-40"]

RESULT_COLUMNS = [
    "benchmark", "outcome", "arm", "model", "fold", "k",
    "n_train", "n_test_own", "n_test_shared", "r2_own", "r2_shared",
]
COVERAGE_COLUMNS = [
    "benchmark", "arm", "n_items", "n_items_with_vector", "item_share",
    "n_tokens", "n_tokens_with_row", "token_share",
]


class Arm(NamedTuple):
    name: str
    load: Callable[[], Embeddings]
    tokens: Callable[[str, list[str], Embeddings], list[list[str]]]
    baseline: str | None = None
    # Files the arm's rows come from. A saved fold older than one of them is refitted.
    inputs: tuple[Path, ...] = ()


def pack_tokens(dataset: str, texts: list[str], emb: Embeddings) -> list[list[str]]:
    return Corpus(texts, embeddings=emb).docs


def load_warriner() -> tuple[list[str], dict[str, np.ndarray]]:
    outcomes = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
    # keep_default_na=False: the word "null" is a word, not a missing value.
    df = pd.read_csv(
        REPO / "testpack" / "data" / "warriner2013" / "BRM-emot-submit.csv",
        usecols=["Word", *outcomes], keep_default_na=False,
    )
    assert len(df) == 13915
    return df["Word"].astype(str).tolist(), {c: df[c].to_numpy(np.float64) for c in outcomes}


def load_hate_speech() -> tuple[list[str], dict[str, np.ndarray]]:
    df = pd.read_parquet(
        REPO.parent / "Corpuses" / "measuring_hate_speech.parquet",
        columns=["comment_id", "text", "hate_speech_score"],
    )
    # One row per comment: the parquet has one row per annotation.
    g = df.groupby("comment_id")
    texts = g["text"].first()
    score = g["hate_speech_score"].mean()
    assert len(texts) == 39565
    return texts.astype(str).tolist(), {"hate_speech_score": score.to_numpy(np.float64)}


DATASETS = {"warriner": load_warriner, "hate_speech": load_hate_speech}


def source_rows(zip_path: Path, dims: int) -> tuple[list[str], np.ndarray]:
    """(keys, rows) of a GloVe zip that its pack keeps, before normalization.

    Saved once as a pickle of (keys, rows) at
    `<zip stem>.pkl` next to the zip, and rebuilt when `build.py` or
    `source.py`, which choose the rows, is newer than it. Reading the 5 GB zip
    takes minutes; loading the pickle takes seconds. ssdiff's own formats do not fit:
    .ssdembed does not load in ssdiff 3.2, and word2vec .bin ends a key at its first
    space, while GloVe keys may contain one.
    """
    cached = zip_path.with_suffix(".pkl")
    if not cached.exists() or cached.stat().st_mtime < max(Path(m.__file__).stat().st_mtime for m in (build_module, source_module)):
        print(f"reading {zip_path.name} (once, then cached as {cached.name}) ...", file=sys.stderr, flush=True)
        keys, rows = [], []
        for key, row, _ in kept_rows(zip_path, dims, Counter(), case_sensitive=False):  # the English packs are case-insensitive
            keys.append(key)
            rows.append(row)
        rows = np.stack(rows)
        # Written through .part, so a run killed mid-write leaves no cut-off cache.
        part = cached.with_name(cached.name + ".part")
        with open(part, "wb") as f:
            pickle.dump((keys, rows), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(part, cached)
    else:
        with open(cached, "rb") as f:
            keys, rows = pickle.load(f)
    return keys, rows


def load_source(zip_path: Path, dims: int) -> Embeddings:
    """The rows of a GloVe zip that its pack keeps, at L2 + ABTT 1."""
    return Embeddings(*source_rows(zip_path, dims)).normalize(l2=True, abtt=1)


def spacy_tokens(dataset: str, texts: list[str], emb: Embeddings) -> list[list[str]]:
    """ssdiff's spaCy route, cached per dataset and shared by both spaCy arms.

    The versions are in the file name, so a release that changes spaCy's
    tokenizer or stopwords, or ssdiff's token filter, never reads an old
    cache. An edit to an ssdiff checkout that keeps its version does; delete
    the cache by hand then.
    """
    try:
        model_version = version(SPACY_MODEL)
    except PackageNotFoundError:
        # Corpus offers ssdiff's download; the cache name needs the model's version, so this run is not cached.
        return Corpus(texts, lang="en").docs
    cache = REPO / "build" / "bench" / "cache" / (
        f"{dataset}-{SPACY_MODEL}-{model_version}-spacy{version('spacy')}-ssdiff{version('ssdiff')}.pkl"
    )
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    print(f"spaCy over {len(texts):,} {dataset} items (once, then cached) ...", file=sys.stderr, flush=True)
    docs = Corpus(texts, lang="en").docs
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Through .part, as in source_rows.
    part = cache.with_name(cache.name + ".part")
    part.write_bytes(pickle.dumps(docs, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(part, cache)
    return docs


def english_arms(recipe_names: list[str]) -> list[Arm]:
    arms = []
    for recipe_name in recipe_names:
        recipe = tomllib.loads((REPO / "languages" / "en" / f"{recipe_name}.toml").read_text("utf-8"))
        zip_path = REPO / "build" / "sources" / recipe["url"].rsplit("/", 1)[-1]
        # `build` puts the version in the file name; the benchmark runs the newest build.
        # The glob also matches other files, and packs whose name starts with this one.
        builds = sorted(
            (
                p
                for p in (REPO / "build").glob(f"{recipe['name']}-*.wordpack")
                if re.fullmatch(re.escape(recipe["name"]) + "-[0-9]+", p.stem)
            ),
            key=lambda p: int(p.stem.rsplit("-", 1)[1]),
        )
        if not builds:
            raise FileNotFoundError(f"no build of {recipe['name']} in build/: run `wordpack-compiler build en/{recipe_name}`")
        pack_path = builds[-1]
        short = recipe_name.removeprefix("glove2024-")
        arms.append(Arm(f"spacy-{short}", functools.partial(load_source, zip_path, int(recipe["dims"])), spacy_tokens,
                        inputs=(zip_path.with_suffix(".pkl"),)))
        arms.append(Arm(f"pack-{short}", functools.partial(Embeddings.load, str(pack_path)), pack_tokens,
                        baseline=f"spacy-{short}", inputs=(pack_path,)))
    return arms


def _corpus(docs: list[list[str]], idx: np.ndarray) -> Corpus:
    # Corpus cannot take a subset. Without lang, fit_ols()'s PCA sweep raises on
    # embeddings that are not a wordpack.
    return Corpus([docs[i] for i in idx], pretokenized=True, lang="en")


def run_arm(arm: Arm, datasets: dict, folds: dict, out_dir: Path) -> tuple:
    """Fit every model in every fold of every dataset on one arm.

    Returns (pack, coverage rows, has-vector masks, predictions, fit rows).
    None of these holds the embeddings or a fit result.
    """
    # The previous arm's fit results reference themselves and hold its
    # embeddings, so only the cycle collector frees them. Collect before the
    # load, so two source embeddings are never in memory at once.
    gc.collect()
    emb = arm.load()
    pack = f"{emb.name}-{emb.version}" if emb.name else None
    coverage, has_vec, preds, fits = [], {}, {}, []
    for b, (texts, outcomes) in datasets.items():
        docs = arm.tokens(b, texts, emb)
        hits = [[t in emb for t in doc] for doc in docs]
        hv = np.array([any(h) for h in hits])
        has_vec[b] = hv
        n_tokens, n_hits = sum(map(len, hits)), sum(map(sum, hits))
        coverage.append({
            "benchmark": b, "arm": arm.name,
            "n_items": len(docs), "n_items_with_vector": int(hv.sum()), "item_share": hv.mean(),
            "n_tokens": n_tokens, "n_tokens_with_row": n_hits,
            "token_share": n_hits / n_tokens if n_tokens else float("nan"),
        })
        for outcome, y in outcomes.items():
            for f, test in enumerate(folds[b]):
                train = np.setdiff1d(np.arange(len(docs)), test)
                train = train[hv[train]]
                # Each fold's (k, test predictions) per model is saved as soon
                # as the fold ends, so a stopped run restarted into the same
                # out_dir fits only the folds and models it has not saved. A
                # fold saved before one of the arm's inputs was last written
                # is refitted.
                saved = out_dir / "folds" / f"{arm.name}-{b}-{outcome}-{f}.pkl"
                fresh = saved.exists() and all(p.stat().st_mtime < saved.stat().st_mtime for p in arm.inputs)
                done = pickle.loads(saved.read_bytes()) if fresh else {}
                todo = [m for m in MODELS if m not in done]
                if todo:
                    print(f"{arm.name}  {b}  {outcome}  fold {f + 1}/{len(folds[b])}", file=sys.stderr, flush=True)
                    ssd = SSD(emb, _corpus(docs, train), y[train], use_full_doc=True, sif_a=None)
                    test_corpus = _corpus(docs, test)
                for model in todo:
                    try:
                        result = MODELS[model](ssd)
                    except ValueError as e:
                        if "found no component" not in str(e):
                            raise
                        done[model] = (0, None)
                    else:
                        k = result.pca_k if model.startswith("ols") else result.n_components
                        p = result.predict(test_corpus)
                        # Shared R² is taken over the has-vector masks, so they
                        # must be exactly the items predict() scores.
                        if not np.array_equal(np.isfinite(p), hv[test]):
                            raise RuntimeError(f"{arm.name} {b} {outcome} {model} fold {f}: predict() and the coverage count disagree")
                        done[model] = (k, p)
                if todo:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    # Through .part, as in source_rows.
                    part = saved.with_name(saved.name + ".part")
                    part.write_bytes(pickle.dumps(done, protocol=pickle.HIGHEST_PROTOCOL))
                    os.replace(part, saved)
                for model in MODELS:
                    k, p = done[model]
                    pred = preds.setdefault((b, outcome, model), np.full(len(docs), np.nan))
                    if p is not None:
                        pred[test] = p
                    fits.append({"benchmark": b, "outcome": outcome, "arm": arm.name,
                                 "model": model, "fold": f, "k": k, "n_train": len(train)})
    return pack, coverage, has_vec, preds, fits


def _r2(y: np.ndarray, pred: np.ndarray, y_bar: float) -> float:
    if len(y) == 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y_bar) ** 2))


def run(arms: list[Arm], datasets: dict, n_folds: int, out_dir) -> None:
    """Run every arm on every dataset and write results.csv, coverage.csv and summary.md to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = {
        b: np.array_split(np.random.default_rng(SEED).permutation(len(texts)), n_folds)
        for b, (texts, _) in datasets.items()
    }
    packs, coverage, has_vec, preds, fits = {}, [], {}, {}, []
    for arm in arms:
        pack, cov, hv, pr, ft = run_arm(arm, datasets, folds, out_dir)
        if pack:
            packs[pack] = None
        coverage += cov
        fits += ft
        has_vec.update({(b, arm.name): m for b, m in hv.items()})
        preds.update({(b, o, arm.name, m): p for (b, o, m), p in pr.items()})

    shared = {b: np.logical_and.reduce([has_vec[b, a.name] for a in arms]) for b in datasets}
    rows = []
    for fit in fits:
        b, outcome, arm = fit["benchmark"], fit["outcome"], fit["arm"]
        test = folds[b][fit["fold"]]
        y = datasets[b][1][outcome]
        y_bar = np.delete(y, test).mean()  # all training items, the same for every arm
        pred = preds[b, outcome, arm, fit["model"]]
        own = test[has_vec[b, arm][test]]
        both = test[shared[b][test]]
        rows.append({
            **fit,
            "n_test_own": len(own), "n_test_shared": len(both),
            "r2_own": _r2(y[own], pred[own], y_bar),
            "r2_shared": _r2(y[both], pred[both], y_bar),
        })

    res = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    res.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(coverage, columns=COVERAGE_COLUMNS).to_csv(out_dir / "coverage.csv", index=False)
    (out_dir / "summary.md").write_text(_summary(res, arms, list(packs), n_folds), "utf-8")


def _version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return "not installed"


def _summary(res: pd.DataFrame, arms: list[Arm], packs: list[str], n_folds: int) -> str:
    lines = [
        "# Benchmark",
        "",
        f"ssdiff {_version('ssdiff')} · plskit {_version('plskit')} · spaCy {_version('spacy')} · "
        f"{SPACY_MODEL} {_version(SPACY_MODEL)}",
        f"packs: {', '.join(packs)}",
        f"seed {SEED} · {n_folds} folds",
        "",
        "Means and SDs are over the folds with a finite R²; `folds` counts them.",
    ]
    for (b, outcome), cell in res.groupby(["benchmark", "outcome"], sort=False):
        lines += ["", f"## {b} · {outcome}", "", _md(_arm_table(cell))]
        pairs = _pair_table(cell, arms)
        if len(pairs):
            lines += ["", _md(pairs)]
    return "\n".join(lines) + "\n"


def _arm_table(cell: pd.DataFrame) -> pd.DataFrame:
    return cell.groupby(["model", "arm"], sort=False).agg(
        own_mean=("r2_own", "mean"), own_sd=("r2_own", "std"),
        shared_mean=("r2_shared", "mean"), shared_sd=("r2_shared", "std"),
        folds=("r2_own", "count"),
    ).reset_index()


def _pair_table(cell: pd.DataFrame, arms: list[Arm]) -> pd.DataFrame:
    """Per-fold pack − baseline R² differences within each model."""
    out = []
    cols = ["r2_own", "r2_shared"]
    for arm in arms:
        if arm.baseline is None:
            continue
        pack = cell[cell.arm == arm.name].set_index(["model", "fold"])[cols]
        base = cell[cell.arm == arm.baseline].set_index(["model", "fold"])[cols]
        for model, d in (pack - base).groupby(level="model", sort=False):
            row = {"model": model, "pack": arm.name, "baseline": arm.baseline}
            for col in cols:
                s = col.removeprefix("r2_")
                row[f"{s}_diff_mean"] = d[col].mean()
                row[f"{s}_diff_sd"] = d[col].std()
                row[f"{s}_pack_higher"] = int((d[col] > 0).sum())
                row[f"{s}_folds"] = int(d[col].count())
                row[f"{s}_diff_per_fold"] = " ".join(f"{x:+.4f}" for x in d[col])
            out.append(row)
    return pd.DataFrame(out)


def _md(df: pd.DataFrame) -> str:
    cell = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wordpack benchmark: coverage and CV R² against the spaCy route.")
    parser.add_argument("benchmarks", nargs="+", choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=DEFAULT_MODELS,
                        help=f"models to fit (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--packs", nargs="+", choices=["dolma", "wikigiga"], default=["dolma"],
                        help="English packs to run, each with its spaCy baseline (default: dolma)")
    args = parser.parse_args(argv)
    for model in list(MODELS):
        if model not in args.models:
            del MODELS[model]
    datasets = {b: DATASETS[b]() for b in args.benchmarks}
    out_dir = REPO / "build" / "bench" / date.today().isoformat()
    run(english_arms([f"glove2024-{p}" for p in args.packs]), datasets, N_FOLDS, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
