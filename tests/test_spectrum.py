"""Kern: Periodogramm, Abtastung, scheinbare Frequenz, Tiefpass, gefaltete Dichte, Operationszählung (Einheiten gegen Handrechnung)."""

import math

import numpy as np
import pytest

import pgm_spectrum as S
import signal_generator as G


def test_mini_instance_periodogram_by_hand(mini):
    I = S.periodogram(mini, 1.0)
    assert I.shape == (5,)
    assert I[2] == pytest.approx(2.0, abs=1e-12)
    assert np.allclose(np.delete(I, 2), 0.0, atol=1e-12)


def test_parseval_on_the_mini_instance(mini):
    I = S.periodogram(mini, 1.0)
    full = I[0] + 2 * I[1:4].sum() + I[4]               # beide Hälften des zweiseitigen Spektrums
    assert full == pytest.approx((mini ** 2).sum() / 1.0, rel=1e-12)


def test_parseval_on_random_even_and_odd_lengths():
    rng = G.SplitMix64(5)
    for n in (16, 17, 100, 255):
        x = rng.normal(n)
        I = S.periodogram(x, 3.0)
        full = I[0] + 2 * I[1:(n + 1) // 2].sum() + (I[n // 2] if n % 2 == 0 else 0.0)
        assert full == pytest.approx((x ** 2).sum() / 3.0, rel=1e-10)


def test_bin_frequencies():
    assert S.bin_freqs(8, 8.0).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert S.bin_freqs(7, 7.0).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_aliasing_by_hand_seven_hertz_looks_like_one_hertz():
    t = np.arange(32) / 8.0
    assert np.allclose(np.cos(2 * np.pi * 7.0 * t), np.cos(2 * np.pi * 1.0 * t), atol=1e-12)
    assert S.apparent_frequency(7.0, 8.0) == pytest.approx(1.0)


@pytest.mark.parametrize("f,fs,expected", [(3.0, 10.0, 3.0), (7.0, 10.0, 3.0), (10.0, 10.0, 0.0), (12.0, 10.0, 2.0), (5.0, 10.0, 5.0), (27.0, 10.0, 3.0)])
def test_apparent_frequency_table(f, fs, expected):
    assert float(S.apparent_frequency(f, fs)) == pytest.approx(expected)


def test_apparent_frequency_is_where_the_periodogram_peaks():
    rng = G.SplitMix64(31)
    fs, n = 10.0, 1024
    for f in 0.2 * fs + 6.0 * fs * rng.uniform(80):
        x = np.cos(2 * np.pi * f * np.arange(n) / fs + 1.0)
        peak = S.bin_freqs(n, fs)[int(np.argmax(S.periodogram(x, fs)))]
        assert abs(peak - float(S.apparent_frequency(f, fs))) <= fs / n


def test_antialias_taps_are_symmetric_normalised_and_cut_at_the_given_edge():
    h = S.antialias_taps(4)
    assert len(h) == S.TAPS and h.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(h, h[::-1], atol=1e-15)
    fs_lo = 250.0
    gain = lambda f: float(S.filter_response([f], 1000.0, h)[0])
    assert gain(0.0) == pytest.approx(1.0)
    assert gain(0.9 * fs_lo / 2) == pytest.approx(0.25, abs=0.03)       # -6 dB bei 0,9 Nyquist
    assert gain(1.1 * fs_lo / 2) < 1e-2                                 # Sperrbereich


def test_antialias_taps_match_scipy_firwin():
    sps = pytest.importorskip("scipy.signal")
    for q in (2, 4, 8):
        ref = sps.firwin(S.TAPS, S.CUTOFF * (1000.0 / q) / 2, window=("kaiser", S.KAISER_BETA), fs=1000.0)
        assert np.allclose(S.antialias_taps(q), ref, atol=1e-12)


def test_acquire_without_filter_takes_every_qth_value():
    x = np.arange(40.0)
    assert S.acquire(x, 10, 4).tolist() == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0]
    assert S.hi_length(10, 4, None) == 40


def test_acquire_with_filter_equals_filter_output_every_qth_value():
    h = S.antialias_taps(2)
    x = G.SplitMix64(2).normal(S.hi_length(50, 2, h))
    y = S.acquire(x, 50, 2, h)
    full = np.array([np.dot(h[::-1], x[i:i + len(h)]) for i in range(len(x) - len(h) + 1)])
    assert np.allclose(y, full[: 49 * 2 + 1 : 2], atol=1e-12)


def test_acquire_rejects_a_short_sequence():
    with pytest.raises(ValueError):
        S.acquire(np.zeros(10), 10, 2)


def test_filter_response_without_filter_is_one():
    assert S.filter_response([0.0, 100.0], 1000.0, None).tolist() == [1.0, 1.0]


def test_sampled_density_of_white_noise_folds_to_the_new_rate():
    # q Abbilder der Dichte sigma^2 / fs_hi addieren sich zu sigma^2 / fs_lo
    for q in (1, 2, 4, 8):
        d = S.sampled_psd([0.0, 10.0, 50.0], 1000.0, q, None, (), 0.0, 1.0)
        assert np.allclose(d, 1.0 / (1000.0 / q))


def test_sampled_density_with_filter_removes_the_folded_copies():
    q = 4
    h = S.antialias_taps(q)
    f = np.array([10.0, 50.0])
    d = S.sampled_psd(f, 1000.0, q, h, (), 0.0, 1.0)
    assert np.allclose(d, 1.0 / 1000.0, rtol=0.02)                     # nur das eigene Band bleibt: Dichte wie vor der Abtastung, nicht q-fach


def test_sampled_autocov_unfiltered_is_every_qth_lag():
    poles = ((0.0, 0.9),)
    sg = G.ar_unit_sigma(G.ar_from_poles(poles))
    r_hi = G.noise_autocov(poles, sg, 0.0, 60)
    assert np.allclose(S.sampled_autocov(poles, sg, 0.0, 10, 3, None), r_hi[::3][:10])


def test_sampled_autocov_filtered_matches_the_direct_filter_of_the_autocovariance():
    poles = ((0.1, 0.8),)
    sg = G.ar_unit_sigma(G.ar_from_poles(poles))
    h = S.antialias_taps(2)
    n, q = 6, 2
    r = G.noise_autocov(poles, sg, 0.0, (n - 1) * q + len(h))
    m = len(h)
    expected = []
    for tau in range(n):
        total = 0.0
        for j in range(m):
            for l in range(m):
                total += h[j] * h[l] * r[abs(tau * q + j - l)]
        expected.append(total)
    assert np.allclose(S.sampled_autocov(poles, sg, 0.0, n, q, h), expected, rtol=1e-10, atol=1e-14)


def test_log_stats_of_unit_exponential_known_values():
    rng = G.SplitMix64(9)
    ratio = -np.log1p(-rng.uniform(400000))
    mean_db, std_db = S.log_stats(ratio)
    assert mean_db == pytest.approx(-10 * math.log10(math.e) * 0.5772156649, abs=0.02)
    assert std_db == pytest.approx(10 * math.log10(math.e) * math.pi / math.sqrt(6), abs=0.02)


def test_operation_counts():
    assert S.ops_direct(1024) == 1048576
    assert S.ops_radix2(1024) == 5120 and S.ops_radix2(65536) == 524288
    assert S.ops_radix2(1000) is None and S.ops_radix2(1) is None
    assert S.ops_bluestein(1000) == 3 * 1024 * 11 + 2 * 1000 + 2048
    assert S.ops_bluestein(1025) > S.ops_radix2(1024)


def test_inner_bins_drop_dc_and_the_nyquist_bin_only_for_even_lengths():
    even = np.arange(5)                      # n = 8: Bins 0 .. 4, Nyquist ist Bin 4
    assert even[S.inner_bins(8)].tolist() == [1, 2, 3]
    odd = np.arange(5)                       # n = 9: Bins 0 .. 4, kein Nyquist-Bin
    assert odd[S.inner_bins(9)].tolist() == [1, 2, 3, 4]


def test_hi_length_is_exactly_the_length_the_valid_filter_output_needs():
    h = S.antialias_taps(4)
    assert S.hi_length(10, 4, None) == 40
    assert S.hi_length(10, 4, h) == 40 + len(h) - 1
    S.acquire(np.zeros(S.hi_length(10, 4, h)), 10, 4, h)                 # genau so lang genügt
    with pytest.raises(ValueError):
        S.acquire(np.zeros(S.hi_length(10, 4, h) - 1), 10, 4, h)


def test_operation_counts_at_the_boundaries():
    assert S.ops_radix2(2) == 1 and S.ops_radix2(4) == 4 and S.ops_radix2(3) is None
    assert S.ops_bluestein(1024) == 3 * 1024 * 11 + 2 * 1024 + 2048                  # 2n - 1 = 2047 passt in M = 2048
    assert S.ops_bluestein(1025) == 3 * 2048 * 12 + 2 * 1025 + 4096                  # 2n - 1 = 2049 braucht M = 4096
