"""Experimente und Analyse: feste Zufallsströme, festgeschriebene Werte (Regression) und eine unabhängige Nachrechnung der Analyse.

Die Festwerte stammen aus einem Lauf mit den festen Strömen (SplitMix64, FFT); Toleranz 1e-6 relativ gegen Plattformunterschiede der C-Bibliothek.
Sie fangen Änderungen an Vorzeichen, Fenstergrenzen, Läufen und Bandkanten der Experimentfunktionen, die die weiten Tolerzanzen der Befund-Tests
(`test_claims.py`) nicht sehen."""

import numpy as np
import pytest

import pgm_evaluation as ev
import pgm_spectrum as S


def approx(x):
    return pytest.approx(x, rel=1e-6, abs=1e-9)


def test_sweep_line_fixed_values_for_even_and_odd_length():
    rows = ev.sweep_line(ns=(64, 65), runs=12)
    assert rows[0] == {"n": 64, "runs": 12, "over_floor_db": approx(4.291369305060848), "p_max": approx(1 / 6), "higher": approx(3.1666666666666665)}
    assert rows[1] == {"n": 65, "runs": 12, "over_floor_db": approx(6.596994802847771), "p_max": approx(2 / 3), "higher": approx(1.25)}


def test_sweep_sampling_fixed_values():
    r = ev.sweep_sampling(qs=(2,), n=64, runs=8)[0]
    assert r == {"q": 2, "true_db": approx(-38.824294847169554), "folded_db": approx(-36.58913022757732), "raw_db": approx(-36.80326863234158),
                 "filtered_oracle_db": approx(-41.088358162491836), "filtered_db": approx(-41.2437921769593)}


def test_sweep_scalloping_fixed_values():
    rows = ev.sweep_scalloping(n=256, offsets=(0.0, 0.25, 0.5), runs=6)
    assert [r["loss_db"] for r in rows] == [approx(0.0), approx(-0.9240860278835304), approx(-3.861044876328616)]


def test_sweep_bias_fixed_values_and_standard_scores():
    rows = ev.sweep_bias(ns=(128,), runs=12, probes=(0.1, 0.49))
    assert [r["sim_vs_formula_db"] for r in rows] == [approx(-2.339873946694369), approx(-1.595066853361172)]
    assert [r["z"] for r in rows] == [approx(-2.5201647302423678), approx(-1.2164097862118572)]
    assert [r["bias_db"] for r in rows] == [approx(2.6369237207604366), approx(17.880333227029833)]


def test_standard_scores_of_the_bias_sweep_are_small_in_the_default_experiment():
    assert max(abs(r["z"]) for r in ev.sweep_bias()) < 4.0


@pytest.mark.parametrize("n,seed", [(64, 3), (65, 4), (128, 5)])
def test_analysis_matches_an_independent_recomputation_with_a_buried_line(n, seed):
    s = ev.Settings(n=64 if n == 65 else n, snr=-25, seed=seed, line=100.0)
    a = ev.analyse(s)
    bin_hz = a.fs / s.n
    inner = np.arange(1, s.n // 2)                                   # gerades n: ohne Bin 0 und Nyquist-Bin
    near = [k for k in inner if abs(a.f[k] - a.apparent) <= 1.5 * bin_hz]
    far = [k for k in inner if abs(a.f[k] - a.apparent) > 3.0 * bin_hz]
    peak = max(a.I[k] for k in near)
    assert a.line_peak == pytest.approx(peak, rel=1e-12)
    assert a.floor == pytest.approx(np.mean([a.I[k] for k in far]), rel=1e-12)
    assert a.higher == sum(a.I[k] > peak for k in far)
    top = max(inner, key=lambda k: a.I[k])
    assert a.peak_f == pytest.approx(a.f[top]) and a.line_is_max == (abs(a.f[top] - a.apparent) <= 1.5 * bin_hz)


def test_a_buried_line_is_not_always_the_maximum_and_the_peak_is_taken_next_to_the_line():
    results = [ev.analyse(ev.Settings(n=64, snr=-25, seed=sd, line=100.0)) for sd in range(1, 9)]
    assert any(not a.line_is_max for a in results)
    assert all(a.line_peak <= a.I[1:-1].max() + 1e-18 for a in results)


def test_inner_bins_of_the_analysis_follow_the_helper():
    assert S.inner_bins(64) == slice(1, 32) and S.inner_bins(65) == slice(1, 33)
