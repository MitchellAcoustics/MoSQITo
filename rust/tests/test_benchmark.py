"""Wall-time comparison between `mosqito` (pure Python) and `mosqito_rs`
(this port) on identical inputs, via `pytest-benchmark`.

Not run as part of the routine conformance/differential suite — select
these explicitly::

    pytest tests/ -m benchmark --benchmark-only --benchmark-group-by=name

Needs `mosqito` installed (`pip install mosqito`); skipped otherwise. See
`crates/mosqito-core/benches/metrics.rs` for the Rust-only Criterion
benchmarks (no Python/mosqito comparison, but covering `roughness_ecma` and
longer signals too).
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs

FS = 48000.0


def _pink_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.uniform(-1.0, 1.0, size=n)
    b0 = b1 = b2 = 0.0
    out = np.empty(n)
    for i in range(n):
        w = white[i]
        b0 = 0.99765 * b0 + w * 0.0990460
        b1 = 0.96300 * b1 + w * 0.2965164
        b2 = 0.57000 * b2 + w * 1.0526913
        out[i] = (b0 + b1 + b2 + w * 0.1848) * 0.05
    return out


def _tone_1khz(n: int) -> np.ndarray:
    t = np.arange(n) / FS
    return 0.05 * np.sin(2 * np.pi * 1000 * t)


@pytest.fixture(scope="module")
def pink_1s():
    return _pink_noise(int(FS), seed=1)


@pytest.fixture(scope="module")
def tone_1s():
    return _tone_1khz(int(FS))


@pytest.fixture(scope="module")
def mosqito():
    return pytest.importorskip("mosqito")


@pytest.mark.benchmark
def test_loudness_zwst_rust(benchmark, pink_1s):
    benchmark(mosqito_rs.loudness_zwst, pink_1s, FS)


@pytest.mark.benchmark
def test_loudness_zwst_python(benchmark, pink_1s, mosqito):
    benchmark(mosqito.sq_metrics.loudness_zwst, pink_1s, FS)


@pytest.mark.benchmark
def test_loudness_zwtv_rust(benchmark, pink_1s):
    benchmark(mosqito_rs.loudness_zwtv, pink_1s, FS)


@pytest.mark.benchmark
def test_loudness_zwtv_python(benchmark, pink_1s, mosqito):
    benchmark(mosqito.sq_metrics.loudness_zwtv, pink_1s, FS)


@pytest.mark.benchmark
def test_sharpness_din_st_rust(benchmark, pink_1s):
    benchmark(mosqito_rs.sharpness_din_st, pink_1s, FS)


@pytest.mark.benchmark
def test_sharpness_din_st_python(benchmark, pink_1s, mosqito):
    benchmark(mosqito.sq_metrics.sharpness_din_st, pink_1s, FS)


@pytest.mark.benchmark
def test_loudness_ecma_rust(benchmark, tone_1s):
    benchmark(mosqito_rs.loudness_ecma, tone_1s, FS)


@pytest.mark.benchmark
def test_loudness_ecma_python(benchmark, tone_1s, mosqito):
    benchmark(mosqito.sq_metrics.loudness_ecma, tone_1s, FS)


@pytest.mark.benchmark
def test_roughness_ecma_rust(benchmark, tone_1s):
    benchmark(mosqito_rs.roughness_ecma, tone_1s, FS)


@pytest.mark.benchmark
def test_roughness_ecma_python(benchmark, tone_1s, mosqito):
    # MoSQITo's own roughness_ecma is slow (a full Python nested loop over
    # 53 bands x segments): pytest-benchmark's default is many rounds, so
    # cap it explicitly to keep this comparison tractable.
    benchmark.pedantic(
        mosqito.sq_metrics.roughness_ecma, args=(tone_1s, FS), rounds=2, iterations=1
    )
