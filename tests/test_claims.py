"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (C.SWEEP_SEED und folgende), nie über einen einzelnen Lauf. Toleranzen sind größer als der
Standardfehler der jeweiligen Läufe, damit Bibliotheksversionen (die CI installiert ungepinnt) nichts kippen.
"""

import math

import numpy as np
import pytest

import pgm_constants as C
import pgm_evaluation as ev
import pgm_spectrum as S
import signal_generator as G


@pytest.fixture(scope="module")
def variation():
    return {r["n"]: r for r in ev.sweep_variation()}


@pytest.fixture(scope="module")
def line():
    return {r["n"]: r for r in ev.sweep_line()}


@pytest.fixture(scope="module")
def bias():
    return ev.sweep_bias()


@pytest.fixture(scope="module")
def scallop():
    return {r["offset"]: r for r in ev.sweep_scalloping()}


@pytest.fixture(scope="module")
def sampling():
    return {r["q"]: r for r in ev.sweep_sampling()}


def test_experiment_streams_are_the_documented_ones():
    assert C.SWEEP_SEED == 100000 and C.EXP_N_SWEEP == (64, 256, 1024, 4096) and C.EXP_LINE_SNR == -10
    assert C.EXP_RUNS == {64: 800, 256: 400, 1024: 200, 4096: 100}


@pytest.mark.parametrize("n", C.EXP_N_SWEEP)
def test_spread_per_bin_equals_the_mean_for_every_length(variation, n):
    assert variation[n]["cv"] == pytest.approx(1.00, abs=0.03)
    assert variation[n]["mean"] == pytest.approx(1.00, abs=0.03)
    assert 100 * variation[n]["p_gt3"] == pytest.approx(5.0, abs=0.7)                  # exp(-3) = 4,98 %


@pytest.mark.parametrize("n", C.EXP_N_SWEEP)
def test_log_periodogram_bias_and_spread(variation, n):
    assert variation[n]["db_bias"] == pytest.approx(-2.5, abs=0.1)
    assert variation[n]["db_std"] == pytest.approx(5.6, abs=0.15)


def test_log_bias_and_spread_have_the_closed_forms_quoted_in_the_text():
    assert -10 * math.log10(math.e) * 0.5772156649 == pytest.approx(-2.51, abs=0.005)
    assert 10 * math.log10(math.e) * math.pi / math.sqrt(6) == pytest.approx(5.57, abs=0.005)


@pytest.mark.parametrize("n,db", [(64, 6.0), (256, 10.3), (1024, 16.1), (4096, 22.0)])
def test_line_above_the_floor(line, n, db):
    assert line[n]["over_floor_db"] == pytest.approx(db, abs=0.7)


def test_line_gain_is_ten_log_n_for_large_n(line):
    assert line[4096]["over_floor_db"] - line[1024]["over_floor_db"] == pytest.approx(10 * math.log10(4), abs=0.7)


@pytest.mark.parametrize("n,p", [(64, 45), (256, 90), (1024, 100), (4096, 100)])
def test_probability_that_the_line_is_the_highest_value(line, n, p):
    assert 100 * line[n]["p_max"] == pytest.approx(p, abs=9)


def test_valley_bias_by_length(bias):
    valley = {r["n"]: r["bias_db"] for r in bias if r["frac"] == 0.49}
    assert valley[128] == pytest.approx(17.9, abs=0.15)
    assert valley[512] == pytest.approx(11.9, abs=0.15)
    assert valley[2048] == pytest.approx(6.7, abs=0.15)


def test_bias_near_the_peak_is_small_after_128(bias):
    peak = [r for r in bias if r["frac"] == 0.2]
    assert max(abs(r["bias_db"]) for r in peak if r["n"] >= 512) < 1.0


def test_dynamic_range_of_the_steep_background_is_57_db():
    poles, sg, wh = ev.noise_spec("Steil")
    f = np.linspace(0.0, 500.0, 20001)
    s = G.true_psd(f, C.FS_HI, poles, sg, wh)
    assert 10 * math.log10(s.max() / s.min()) == pytest.approx(56.7, abs=0.3)


def test_scalloping_loss(scallop):
    assert scallop[0.25]["loss_db"] == pytest.approx(-0.9, abs=0.15)
    assert scallop[0.5]["loss_db"] == pytest.approx(-3.9, abs=0.15)


@pytest.mark.parametrize("q,excess", [(2, 2.2), (4, 2.8), (8, 3.1)])
def test_folded_noise_raises_the_upper_band(sampling, q, excess):
    r = sampling[q]
    assert r["raw_db"] - r["true_db"] == pytest.approx(excess, abs=0.4)
    assert r["raw_db"] == pytest.approx(r["folded_db"], abs=0.35)
    assert r["filtered_db"] < r["true_db"] - 1.0                                       # der Tiefpass kostet Nutzband am Rand


def test_antialias_filter_levels_quoted_in_the_text():
    def gain(q, frac):
        fs_lo = C.FS_HI / q
        return 10 * math.log10(float(S.filter_response([frac * fs_lo / 2], C.FS_HI, S.antialias_taps(q))[0]))

    for q in (2, 4, 8):
        assert gain(q, 0.9) == pytest.approx(-6.0, abs=0.3)                              # Entwurfsvorgabe: -6 dB bei 0,9 Nyquist
    assert gain(8, 1.0) == pytest.approx(-14.5, abs=0.5)                                 # feste 129 Taps: der Übergang wird mit q breiter
    assert gain(4, 1.0) == pytest.approx(-29.9, abs=0.5)


def test_operation_counts_in_the_text():
    assert S.ops_direct(1024) == 1_048_576 and S.ops_radix2(1024) == 5_120
    assert S.ops_direct(65536) == 4_294_967_296 and S.ops_radix2(65536) == 524_288


def test_line_preset_numbers_quoted_in_the_help_texts():
    assert C.PRESETS["Zu langsam abgetastet"]["line"] == 300.0 and C.FS_HI / 2 / C.SAMPLING_Q["jeder 2."][0] == 250.0
    assert float(S.apparent_frequency(300.0, 500.0)) == pytest.approx(200.0)
    assert C.PRESETS["Mehr Daten"]["n"] == 4096 and C.PRESETS["Eine Linie im Rauschen"]["n"] == 512
