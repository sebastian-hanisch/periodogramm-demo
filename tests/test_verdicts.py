"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import pgm_evaluation as ev


def _analysis(cv: float, n: int = 512):
    s = ev.Settings(n=n)
    z = np.zeros(n // 2 + 1)
    return ev.Analysis(s, 1000.0, z, np.zeros(n), z, z, z, z, 100.0, 100.0, True, 1.0, 1.0, 0, cv)


def test_spread_verdict_is_a_warning_near_one_and_names_the_number_and_the_bins():
    kind, text = ev.verdict_spread(_analysis(1.02, 512))
    assert kind == "warning" and "Std/Mittel **1,02**" in text and "(257 Bins)" in text


def test_spread_verdict_threshold_at_0_8():
    assert ev.verdict_spread(_analysis(0.81))[0] == "warning"
    assert ev.verdict_spread(_analysis(0.8))[0] == "info"
    assert ev.verdict_spread(_analysis(0.5))[0] == "info"


def test_variation_verdict_names_first_and_last_length():
    rows = [{"n": 64, "cv": 1.003, "db_std": 5.61, "db_bias": -2.50}, {"n": 4096, "cv": 1.000, "db_std": 5.57, "db_bias": -2.50}]
    kind, text = ev.verdict_variation(rows)
    assert kind == "warning"
    assert "1,00 bei n = 64 und 1,00 bei n = 4096" in text and "5,6 dB" in text and "2,5 dB" in text


def test_line_verdict_names_gain_and_probability():
    rows = [{"n": 64, "over_floor_db": 5.9, "p_max": 0.45}, {"n": 4096, "over_floor_db": 22.0, "p_max": 1.0}]
    kind, text = ev.verdict_line(rows)
    assert kind == "info"
    assert "bei n = 64 um 5,9 dB" in text and "bei n = 4096 um 22,0 dB" in text and "45 % bzw. 100 %" in text


def test_bias_verdict_takes_the_valley_row_and_the_half_bin_loss():
    rows = [{"n": 128, "frac": 0.49, "bias_db": 17.9}, {"n": 128, "frac": 0.2, "bias_db": 1.8}, {"n": 2048, "frac": 0.49, "bias_db": 6.7}]
    scallop = [{"offset": 0.0, "loss_db": 0.0}, {"offset": 0.5, "loss_db": -3.9}]
    kind, text = ev.verdict_bias(rows, scallop)
    assert kind == "info"
    assert "17,9 dB (n = 128)" in text and "6,7 dB (n = 2048)" in text and "verliert 3,9 dB" in text


def test_sampling_verdict_reports_excess_and_filtered_level():
    rows = [{"q": 2, "true_db": -38.8, "raw_db": -36.6, "folded_db": -36.6, "filtered_db": -40.9},
            {"q": 8, "true_db": -27.8, "raw_db": -24.7, "folded_db": -24.8, "filtered_db": -32.0}]
    kind, text = ev.verdict_sampling(rows)
    assert kind == "warning"
    assert "jedem 8. Wert" in text and "3,1 dB zu hoch" in text and "-24,7 dB" in text and "-27,8 dB" in text and "-24,8 dB" in text and "-32,0 dB" in text


def test_decimal_comma_helper():
    assert ev.de(3.14159, 2) == "3,14" and ev.de(-2.5, 1) == "-2,5" and ev.de(7.0, 0) == "7"
