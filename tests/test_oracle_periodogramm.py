"""Orakel: Kernroutinen gegen einen unabhängigen Rechenweg (scipy, Aufzählung, geschlossene Formeln; DEMO-PLAYBOOK §12)."""

import math

import numpy as np
import pytest

import pgm_constants as C
import pgm_evaluation as ev
import pgm_spectrum as S
import signal_generator as G

_trapz = getattr(np, "trapezoid", None) or np.trapz
sps = pytest.importorskip("scipy.signal")
stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize("n", [16, 17, 100, 255, 256])
def test_periodogram_equals_scipy_two_sided_density(n):
    x = G.SplitMix64(n).normal(n)
    f, p = sps.periodogram(x, fs=250.0, window="boxcar", detrend=False, return_onesided=False, scaling="density")
    assert np.allclose(S.periodogram(x, 250.0), p[: n // 2 + 1], rtol=1e-10, atol=1e-15)


def test_ar_noise_equals_the_recursive_filter_applied_to_the_same_innovations():
    poles = ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9))
    a = G.ar_from_poles(poles)
    sg = G.ar_unit_sigma(a)
    n = 700
    sig = G.make_signal(n, 1000.0, [], poles, sg, 0.0, G.SplitMix64(21))
    length = G.ar_length(a)
    e = sg * G.SplitMix64(21).normal(n + length)
    ref = sps.lfilter([1.0], np.r_[1.0, -a], e)[length:]
    assert np.allclose(sig.x, ref, atol=1e-8)


def test_autocovariance_equals_numerical_integration_of_the_density():
    poles = ((0.2, 0.97),)
    a = G.ar_from_poles(poles)
    sg = 1.0
    f = np.linspace(-0.5, 0.5, 400001)
    s = G.true_psd(f, 1.0, poles, sg, 0.0)
    r = G.ar_autocov(a, sg ** 2, 6)
    for tau in range(6):
        ref = _trapz(s * np.cos(2 * np.pi * f * tau), f)
        assert r[tau] == pytest.approx(ref, rel=1e-4, abs=1e-6)


def test_the_periodogram_values_are_exponentially_distributed():
    x = G.make_noise_batch(60, 1024, (), 0.0, 1.0, G.SplitMix64(1))
    r = S.periodogram(x, 1.0)[:, 1:-1] * 1.0                    # I / S mit S = sigma^2 / fs = 1
    assert stats.kstest(r.ravel()[:30000], "expon").pvalue > 0.001
    assert r.mean() == pytest.approx(1.0, abs=0.02) and r.std() == pytest.approx(1.0, abs=0.02)


def test_expected_periodogram_agrees_with_simulation_within_standard_error():
    poles, sg, wh = ev.noise_spec("Steil")
    n, runs = 256, 1200
    x = G.make_noise_batch(runs, n, poles, sg, wh, G.SplitMix64(77))
    I = S.periodogram(x, C.FS_HI)
    r = G.noise_autocov(poles, sg, wh, n)
    ks = [5, 26, 51, 77, 102, 126]
    e = G.expected_periodogram(np.array(ks) * C.FS_HI / n, C.FS_HI, r)
    for k, ek in zip(ks, e):
        z = (I[:, k].mean() - ek) / (I[:, k].std() / math.sqrt(runs))
        assert abs(z) < 4.0, (k, z)


def test_leakage_bias_in_the_valley_falls_with_length():
    rows = ev.sweep_bias(ns=(128, 512), runs=20)
    valley = {r["n"]: r["bias_db"] for r in rows if r["frac"] == 0.49}
    assert valley[128] > valley[512] > 3.0


def test_a_line_between_two_bins_loses_3_92_db():
    row = ev.sweep_scalloping(n=1024, offsets=(0.0, 0.5), runs=30)[-1]
    assert row["loss_db"] == pytest.approx(10 * math.log10(np.sinc(0.5) ** 2), abs=0.1)


def test_aliased_noise_matches_the_folded_density():
    rows = ev.sweep_sampling(qs=(4,), n=256, runs=60)
    r = rows[0]
    assert r["raw_db"] == pytest.approx(r["folded_db"], abs=0.35)
    assert r["filtered_db"] == pytest.approx(r["filtered_oracle_db"], abs=0.35)
    assert r["raw_db"] > r["true_db"] + 1.0


def test_expected_sampled_periodogram_equals_simulation_for_the_filtered_case():
    poles, sg, wh = ev.noise_spec("Tiefpass")
    n, q, runs = 128, 4, 600
    h = S.antialias_taps(q)
    x_hi = G.make_noise_batch(runs, S.hi_length(n, q, h), poles, sg, wh, G.SplitMix64(9))
    y = np.stack([S.acquire(x_hi[i], n, q, h) for i in range(runs)])
    I = S.periodogram(y, C.FS_HI / q)
    f = S.bin_freqs(n, C.FS_HI / q)
    e = S.expected_sampled_periodogram(f, C.FS_HI / q, n, q, h, poles, sg, wh)
    for k in (4, 20, 40, 60):
        z = (I[:, k].mean() - e[k]) / (I[:, k].std() / math.sqrt(runs))
        assert abs(z) < 4.0, (k, z)
