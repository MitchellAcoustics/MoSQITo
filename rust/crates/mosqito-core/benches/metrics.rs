//! Wall-time benchmarks for the ported metrics, on representative signals.
//!
//! Signals: 1 s and 10 s of synthetic pink noise, a 1 s 1 kHz tone, and (for
//! the fixed-duration groups) the real ISO 532-1 Annex B.5 propeller-airplane
//! recording — the same broadband technical signal the plan names, loaded
//! directly from the checked-in corpus.
//!
//! `roughness_ecma` alone gets a reduced sample size: its per-call cost
//! (tens to hundreds of ms even on short signals, from its 53-band Hilbert
//! transform and per-block spectral analysis) makes Criterion's default 100
//! samples impractically slow.
//!
//! Run from `rust/`: `cargo bench -p mosqito-core`. HTML reports land under
//! `target/criterion/`.

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use mosqito_core::loudness::ecma::loudness_ecma;
use mosqito_core::loudness::zwst::{loudness_zwst, FieldType};
use mosqito_core::loudness::zwtv::loudness_zwtv;
use mosqito_core::roughness::ecma::roughness_ecma;
use mosqito_core::sharpness::din::{sharpness_din_st, Weighting};
use mosqito_core::slm::noct::noct_spectrum;

const FS: f64 = 48000.0;

/// Deterministic xorshift64 PRNG, matching the noise generator used
/// throughout this crate's own tests — no external `rand` dependency needed.
fn xorshift64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Map to [-1, 1).
    ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Approximate pink (1/f) noise via the Paul Kellet "economy" recursive
/// filter applied to xorshift white noise — adequate for benchmarking
/// (realistic spectral shape, not intended as a conformance signal).
fn pink_noise(n_samples: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let (mut b0, mut b1, mut b2) = (0.0f64, 0.0f64, 0.0f64);
    (0..n_samples)
        .map(|_| {
            let white = xorshift64(&mut state);
            b0 = 0.99765 * b0 + white * 0.0990460;
            b1 = 0.96300 * b1 + white * 0.2965164;
            b2 = 0.57000 * b2 + white * 1.0526913;
            let pink = b0 + b1 + b2 + white * 0.1848;
            pink * 0.05 // scale to a plausible Pa-level amplitude
        })
        .collect()
}

fn tone_1khz(n_samples: usize) -> Vec<f64> {
    let amplitude = 0.05;
    (0..n_samples)
        .map(|i| amplitude * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / FS).sin())
        .collect()
}

fn annex_b5_propeller_airplane() -> Option<Vec<f64>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("validations/sq_metrics/loudness_zwtv/input/ISO_532-1/Annex B.5/Test signal 14 (propeller-driven airplane).wav");
    let mut reader = hound::WavReader::open(&path).ok()?;
    let wav_calib = 2.0 * 2f64.sqrt();
    Some(
        reader
            .samples::<i16>()
            .map(|s| wav_calib * s.unwrap() as f64 / (2f64.powi(15) - 1.0))
            .collect(),
    )
}

fn bench_loudness_zwst(c: &mut Criterion) {
    let mut group = c.benchmark_group("loudness_zwst");
    for (label, sig) in [
        ("pink_1s", pink_noise(FS as usize, 1)),
        ("pink_10s", pink_noise(10 * FS as usize, 2)),
        ("tone_1khz_1s", tone_1khz(FS as usize)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &sig, |b, sig| {
            b.iter(|| loudness_zwst(black_box(sig), FS, FieldType::Free));
        });
    }
    group.finish();
}

fn bench_loudness_zwtv(c: &mut Criterion) {
    let mut group = c.benchmark_group("loudness_zwtv");
    for (label, sig) in [
        ("pink_1s", pink_noise(FS as usize, 3)),
        ("pink_10s", pink_noise(10 * FS as usize, 4)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &sig, |b, sig| {
            b.iter(|| loudness_zwtv(black_box(sig), FS, FieldType::Free).unwrap());
        });
    }
    group.finish();
}

fn bench_sharpness_din_st(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpness_din_st");
    for (label, sig) in [
        ("pink_1s", pink_noise(FS as usize, 5)),
        ("tone_1khz_1s", tone_1khz(FS as usize)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &sig, |b, sig| {
            b.iter(|| sharpness_din_st(black_box(sig), FS, Weighting::Din, FieldType::Free));
        });
    }
    group.finish();
}

fn bench_loudness_ecma(c: &mut Criterion) {
    let mut group = c.benchmark_group("loudness_ecma");
    for (label, sig) in [
        ("pink_1s", pink_noise(FS as usize, 6)),
        ("tone_1khz_1s", tone_1khz(FS as usize)),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &sig, |b, sig| {
            b.iter(|| loudness_ecma(black_box(sig), FS, 2048, 1024));
        });
    }
    group.finish();
}

fn bench_noct_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("noct_spectrum");
    for (label, sig) in [
        ("pink_1s", pink_noise(FS as usize, 7)),
        ("pink_10s", pink_noise(10 * FS as usize, 8)),
    ] {
        let sig2d = ndarray::Array2::from_shape_vec((sig.len(), 1), sig).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(label), &sig2d, |b, sig2d| {
            b.iter(|| {
                noct_spectrum(black_box(sig2d.view()), FS, 24.0, 12600.0, 3, 10, 1000.0).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_roughness_ecma(c: &mut Criterion) {
    let mut group = c.benchmark_group("roughness_ecma");
    group.sample_size(10);

    let mut signals = vec![
        ("pink_1s".to_string(), pink_noise(FS as usize, 9)),
        ("tone_1khz_1s".to_string(), tone_1khz(FS as usize)),
    ];
    if let Some(propeller) = annex_b5_propeller_airplane() {
        signals.push(("annex_b5_propeller_airplane".to_string(), propeller));
    }

    for (label, sig) in &signals {
        group.bench_with_input(BenchmarkId::from_parameter(label), sig, |b, sig| {
            b.iter(|| roughness_ecma(black_box(sig), FS));
        });
    }
    group.finish();
}

/// Scaling with rayon thread count, on `roughness_ecma` — the metric with
/// the most parallel work per call (53 bands x (time, band) pairs).
fn bench_roughness_ecma_thread_scaling(c: &mut Criterion) {
    let sig = pink_noise(FS as usize, 10);
    let mut group = c.benchmark_group("roughness_ecma_thread_scaling");
    group.sample_size(10);

    let mut thread_counts = vec![1, 2, 4, rayon::current_num_threads()];
    thread_counts.sort_unstable();
    thread_counts.dedup();

    for threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool builds");
        group.bench_with_input(BenchmarkId::from_parameter(threads), &sig, |b, sig| {
            b.iter(|| pool.install(|| roughness_ecma(black_box(sig), FS)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_loudness_zwst,
    bench_loudness_zwtv,
    bench_sharpness_din_st,
    bench_loudness_ecma,
    bench_noct_spectrum,
    bench_roughness_ecma,
    bench_roughness_ecma_thread_scaling,
);
criterion_main!(benches);
