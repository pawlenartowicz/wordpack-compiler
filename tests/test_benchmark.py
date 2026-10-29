import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ssdiff")
pd = pytest.importorskip("pandas")

from ssdiff import Embeddings  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_run", Path(__file__).parents[1] / "benchmarks" / "run.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

RESULT_COLUMNS = [
    "benchmark", "outcome", "arm", "model", "fold", "k",
    "n_train", "n_test_own", "n_test_shared", "r2_own", "r2_shared",
]
COVERAGE_COLUMNS = [
    "benchmark", "arm", "n_items", "n_items_with_vector", "item_share",
    "n_tokens", "n_tokens_with_row", "token_share",
]
WORDS = ["dog", "cat", "blue", "green", "people", "like", "swim", "happens", "never", "covid"]


def tiny_dataset():
    rng = np.random.default_rng(0)
    texts = [" ".join(rng.choice(WORDS, 4)) for _ in range(79)]
    texts.append("the a is")  # only stopwords: no vector in any arm
    return {"tiny": (texts, {"y": rng.normal(size=80)})}


def pack_arms(pack_path):
    # Two arms on one pack, so shared R² is taken over more than one arm and
    # the pack-against-baseline table has a row.
    load = lambda: Embeddings.load(str(pack_path))  # noqa: E731
    return [
        bench.Arm("pack-a", load, bench.pack_tokens),
        bench.Arm("pack-b", load, bench.pack_tokens, baseline="pack-a"),
    ]


def test_run_writes_three_files(pack_path, tmp_path):
    bench.run(pack_arms(pack_path), tiny_dataset(), 2, tmp_path)

    res = pd.read_csv(tmp_path / "results.csv")
    assert list(res.columns) == RESULT_COLUMNS
    assert len(res) == 2 * len(bench.MODELS) * 2  # arms × models × folds
    # The stopword-only item is in no fold's test count.
    one_model = res[(res.arm == "pack-a") & (res.model == "pls-1")]
    assert one_model.n_test_own.sum() == 79 and one_model.n_test_shared.sum() == 79

    cov = pd.read_csv(tmp_path / "coverage.csv")
    assert list(cov.columns) == COVERAGE_COLUMNS
    assert list(cov.n_items) == [80, 80] and list(cov.n_items_with_vector) == [79, 79]

    summary = (tmp_path / "summary.md").read_text("utf-8")
    assert summary.startswith("# Benchmark")
    assert "test-1" in summary and "seed 2137" in summary


def test_no_component_is_recorded(pack_path, tmp_path, monkeypatch):
    def no_component(ssd, y):
        raise ValueError("fit_pls(k='auto'): plskit found no component (k_star = 0); y shows no signal in X.")

    monkeypatch.setitem(bench.MODELS, "pls-auto", no_component)
    bench.run(pack_arms(pack_path), tiny_dataset(), 2, tmp_path)
    res = pd.read_csv(tmp_path / "results.csv")
    auto = res[res.model == "pls-auto"]
    assert (auto.k == 0).all() and auto.r2_own.isna().all() and auto.r2_shared.isna().all()
    assert res[res.model == "pls-1"].r2_own.notna().all()


def test_other_errors_stop_the_run(pack_path, tmp_path, monkeypatch):
    def broken(ssd, y):
        raise ValueError("something else")

    monkeypatch.setitem(bench.MODELS, "pls-auto", broken)
    with pytest.raises(ValueError, match="something else"):
        bench.run(pack_arms(pack_path), tiny_dataset(), 2, tmp_path)


def test_restart_refits_only_unsaved_folds(pack_path, tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    bench.run(pack_arms(pack_path), tiny_dataset(), 2, fresh)

    real = bench.MODELS["pls-1"]
    calls = []

    def stop_at_third(ssd, y):
        calls.append(1)
        if len(calls) == 3:  # pack-a's two folds are done; pack-b's first is not
            raise KeyboardInterrupt
        return real(ssd, y)

    out = tmp_path / "resumed"
    monkeypatch.setitem(bench.MODELS, "pls-1", stop_at_third)
    with pytest.raises(KeyboardInterrupt):
        bench.run(pack_arms(pack_path), tiny_dataset(), 2, out)

    calls.clear()
    monkeypatch.setitem(bench.MODELS, "pls-1", lambda ssd, y: (calls.append(1), real(ssd, y))[1])
    bench.run(pack_arms(pack_path), tiny_dataset(), 2, out)
    assert len(calls) == 2  # pack-b's two folds only
    pd.testing.assert_frame_equal(pd.read_csv(out / "results.csv"), pd.read_csv(fresh / "results.csv"))


def test_warriner_keeps_null():
    texts, outcomes = bench.load_warriner()
    assert len(texts) == 13915 and "null" in texts
    assert list(outcomes) == ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
    assert all(len(y) == 13915 and np.isfinite(y).all() for y in outcomes.values())


def test_one_arm_in_memory_at_a_time(pack_path, tmp_path):
    # Plain Embeddings, not a pack: their fit results hold the
    # embeddings inside reference cycles. Automatic collection is off, so only
    # run()'s own cleanup can free one arm before the next loads.
    import gc
    import weakref

    pack = Embeddings.load(str(pack_path))
    keys, rows = list(pack.index_to_key), np.array(pack.vectors)
    loaded = []

    def load():
        assert all(ref() is None for ref in loaded), "the previous arm's embeddings are still alive"
        emb = Embeddings(keys, rows.copy()).normalize(l2=True, abtt=1)
        loaded.append(weakref.ref(emb))
        return emb

    def tokens(dataset, texts, emb):
        return [[t for t in text.split() if t in WORDS] for text in texts]

    arms = [bench.Arm("plain-a", load, tokens), bench.Arm("plain-b", load, tokens)]
    gc.disable()
    try:
        bench.run(arms, tiny_dataset(), 2, tmp_path)
    finally:
        gc.enable()
