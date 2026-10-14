"""AppTest: Auswahlfelder, Permalink, Experimente auf Klick (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import pgm_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 240


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.mark.parametrize("sampling", C.SAMPLING)
def test_every_sampling_choice_runs(sampling):
    at = _run()
    at.selectbox(key="sampling_select").set_value(sampling).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("background", list(C.BACKGROUNDS))
def test_every_background_runs_together_with_the_strongest_downsampling(background):
    at = _run()
    at.selectbox(key="background_select").set_value(background).run(timeout=TIMEOUT)
    at.selectbox(key="sampling_select").set_value("jeder 8. mit Tiefpass").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_is_applied_and_clipped():
    at = _run({"n": "99999", "smp": "jeder 4.", "f": "3", "snr": "-99", "bg": "Steil", "seed": "42"})
    assert at.slider(key="n_slider").value == C.N_MAX
    assert at.selectbox(key="sampling_select").value == "jeder 4."
    assert at.slider(key="line_slider").value == C.LINE_MIN
    assert at.slider(key="snr_slider").value == C.SNR_MIN
    assert at.selectbox(key="background_select").value == "Steil"


def test_permalink_ignores_unknown_choices():
    at = _run({"smp": "unsinn", "bg": "auch unsinn"})
    assert at.selectbox(key="sampling_select").value == C.DEFAULT_SAMPLING
    assert at.selectbox(key="background_select").value == C.DEFAULT_BACKGROUND


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Zu langsam abgetastet").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="sampling_select").value == "jeder 2."
    assert at.slider(key="line_slider").value == 300.0


def test_experiments_run_on_click_and_show_four_verdicts():
    at = _run()
    assert not [m for m in at.markdown if "Mehr Daten, gleiche Streuung" in m.value]
    at.button(key="run_experiments").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    shown = [m.value for m in at.markdown]
    assert any("Mehr Daten, gleiche Streuung" in v for v in shown) and any("Abtastung: Faltung des Rauschens" in v for v in shown)
    assert len(at.warning) >= 3          # Kernfrage, Streuung, Abtastung (Verdikt-Arten warning/info)
