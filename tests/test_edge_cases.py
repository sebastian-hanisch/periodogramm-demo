"""Randfälle: kleinste und größte Länge, alle Abtastungen, alle Hintergründe, Linie an den Rändern, ungerade Längen."""

import numpy as np
import pytest

import pgm_constants as C
import pgm_evaluation as ev
import pgm_spectrum as S
import signal_generator as G


@pytest.mark.parametrize("sampling", C.SAMPLING)
@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_combination_runs_at_the_smallest_length(sampling, background):
    a = ev.analyse(ev.Settings(n=C.N_MIN, sampling=sampling, background=background))
    assert len(a.I) == C.N_MIN // 2 + 1 and np.isfinite(a.I).all() and np.isfinite(a.expected).all() and (a.expected > 0).all()


def test_largest_length_with_the_strongest_downsampling_runs():
    a = ev.analyse(ev.Settings(n=C.N_MAX, sampling="jeder 8. mit Tiefpass", background="Steil"))
    assert len(a.x) == C.N_MAX and np.isfinite(a.cv)


@pytest.mark.parametrize("line", [C.LINE_MIN, C.LINE_MAX])
def test_line_at_the_slider_limits(line):
    a = ev.analyse(ev.Settings(line=line))
    assert a.apparent == pytest.approx(line)


def test_odd_length_periodogram_has_no_nyquist_bin():
    x = G.SplitMix64(1).normal(65)
    assert S.periodogram(x, 1.0).shape == (33,) and S.bin_freqs(65, 1.0)[-1] == pytest.approx(32 / 65)


def test_expected_periodogram_with_zero_noise_is_the_line_only():
    n = 8
    s = np.cos(np.pi * np.arange(n) / 2)
    e = G.expected_periodogram(np.array([0.0, 0.25, 0.5]), 1.0, np.zeros(n), s)     # f = 0,25 ist Bin 2 von 8
    assert e[1] == pytest.approx(2.0) and e[0] == pytest.approx(0.0, abs=1e-12)


def test_same_seed_same_analysis_and_different_seed_differs():
    s = ev.Settings(seed=5)
    assert np.array_equal(ev.analyse(s).x, ev.analyse(s).x)
    assert not np.array_equal(ev.analyse(s).x, ev.analyse(ev.Settings(seed=6)).x)


def test_snr_extremes_change_the_line_height_monotonically():
    low = ev.analyse(ev.Settings(snr=C.SNR_MIN)).line_peak
    high = ev.analyse(ev.Settings(snr=C.SNR_MAX)).line_peak
    assert high > 100 * low


def test_cost_table_has_consistent_numbers():
    rows = ev.cost_table((1024,))
    assert rows[0]["direct"] == 1048576 and rows[0]["radix2"] == 5120 and rows[0]["n_odd"] == 1025 and rows[0]["bluestein_odd"] > 5120
