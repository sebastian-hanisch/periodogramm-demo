"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Presets zeigen ihren Effekt (feste Seeds)."""

import numpy as np
import pytest

import pgm_constants as C
import pgm_evaluation as ev
import pgm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and p["n"] % C.N_STEP == 0, name
        assert C.LINE_MIN <= p["line"] <= C.LINE_MAX and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX
        assert p["sampling"] in C.SAMPLING and p["background"] in C.BACKGROUNDS


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Eine Linie im Rauschen"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_names_are_short_enough_for_the_buttons():
    assert max(len(n) for n in C.PRESETS) <= 24


def test_line_preset_finds_the_line_at_its_true_frequency():
    a = ev.analyse(ev.Settings(**C.PRESETS["Eine Linie im Rauschen"]))
    assert a.line_is_max and abs(a.apparent - 123.5) < 1e-9


def test_colored_preset_shows_leakage_in_the_valley():
    a = ev.analyse(ev.Settings(**C.PRESETS["Farbiger Hintergrund"]))
    top = a.f > 0.9 * a.fs / 2
    gap_db = 10 * np.log10(a.expected[top].mean() / a.true_density[top].mean())
    assert gap_db > 5.0


def test_aliasing_preset_moves_the_line_and_folds_the_background():
    p = C.PRESETS["Zu langsam abgetastet"]
    a = ev.analyse(ev.Settings(**p))
    assert a.apparent == pytest.approx(200.0) and a.line_is_max
    assert a.settings.q == 2 and not a.settings.antialias
    top = a.f > 0.8 * a.fs / 2
    assert a.sampled_density[top].mean() > 1.5 * a.true_density[top].mean()


def test_more_data_preset_raises_the_line_but_not_the_scatter():
    few = ev.analyse(ev.Settings(**C.PRESETS["Eine Linie im Rauschen"]))
    many = ev.analyse(ev.Settings(**C.PRESETS["Mehr Daten"]))
    over = lambda a: 10 * np.log10(a.line_peak / a.floor)
    assert over(many) > over(few) + 6.0
    assert abs(many.cv - few.cv) < 0.15 and 0.85 < many.cv < 1.15


def test_sampling_choices_map_to_factor_and_filter():
    assert ev.Settings(sampling="jeder 4. mit Tiefpass").q == 4 and ev.Settings(sampling="jeder 4. mit Tiefpass").antialias
    assert ev.Settings(sampling="jeder Wert").q == 1 and not ev.Settings(sampling="jeder Wert").antialias
