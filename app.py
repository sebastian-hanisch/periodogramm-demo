"""Das Periodogramm: je mehr Daten, desto feiner, aber nicht ruhiger - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Reihe „Spektral- und Wavelet-Analyse“. Ein simuliertes Signal mit bekanntem Spektrum (Linie plus farbiger Hintergrund) wird abgetastet,
das Periodogramm berechnet und gegen den exakten Erwartungswert gelegt. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import pgm_constants as C
import pgm_evaluation as ev
from pgm_evaluation import Settings, analyse, de
from pgm_pdf_export import build_pdf
from pgm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pgm_visualization import build_bias, build_line, build_periodogram, build_sampling, build_signal, build_variation

st.set_page_config(page_title="Das Periodogramm – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _experiments():
    return {"variation": ev.sweep_variation(), "line": ev.sweep_line(), "bias": ev.sweep_bias(), "scallop": ev.sweep_scalloping(),
            "sampling": ev.sweep_sampling()}


st.title("📈 Das Periodogramm: je mehr Daten, desto feiner, aber nicht ruhiger")
st.markdown(
    """
Das **Periodogramm** ist die direkte Antwort der Fourier-Transformation auf die Frage, bei welcher **Frequenz** wie viel **Leistung** steckt. Es ist
fast **erwartungstreu**, aber nicht **konsistent**: Jeder Wert streut um seinen Erwartungswert so stark wie dessen Betrag, bei 64 Messwerten wie bei
4096. Mehr Daten machen die Frequenzachse feiner und **Linien** höher, den **Untergrund** aber nicht ruhiger.

Dies ist die erste Demo der Reihe „Spektral- und Wavelet-Analyse“ und hat nichts mit „Spectral Clustering“ zu tun (dort sind es Eigenwerte eines
Graphen, hier Frequenzen eines Signals). Alles ist **simuliert**: Das wahre Spektrum ist bekannt, deshalb lässt sich jede Abweichung messen. Weiter unten:
**🔬 Experimente**, **⚙️ Rechenaufwand**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Periodogramm", expanded=True):
    st.markdown(
        """
1. **Messwerte in Frequenzen zerlegen:** Die schnelle Fourier-Transformation (FFT) berechnet aus den n Messwerten die Werte $X_k$ bei den Frequenzen $k \\cdot f_s / n$.
2. **Leistung je Hz:** $I(f) = |X(f)|^2 / (n \\cdot f_s)$, das Periodogramm. Eine Linie erscheint als Gipfel, Rauschen als Zackenteppich.
3. **Warum es nicht ruhiger wird:** Bei Rauschen ist $X(f)$ komplex normalverteilt, $|X|^2$ also exponentialverteilt mit **Streuung gleich Mittelwert**.
   Doppelt so viele Messwerte ergeben doppelt so viele Bins, aber keine Mittelung je Bin.
4. **Abtastung:** Wer langsamer abtastet, als es das Signal verlangt, bekommt Frequenzen oberhalb der halben Abtastrate an anderer Stelle zurück
   (**Aliasing**), und Rauschen faltet sich in das Band. Ein Tiefpass vorher verhindert das, kostet aber Nutzband.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.slider("Länge n der Folge", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Anzahl der Messwerte, aus denen das Periodogramm entsteht.")
    line = st.slider("Linienfrequenz (Hz)", *bounds("line_slider"), step=0.5, key="line_slider",
                     help="Frequenz der Linie vor der Abtastung (die Messrate vorher beträgt 1000 Hz, Nyquist 500 Hz).")
    snr = st.slider("Linienstärke (dB)", *bounds("snr_slider"), key="snr_slider",
                    help="Leistung der Linie gegen die Rauschvarianz 1, je Abtastwert.")
    background = st.selectbox("Hintergrund", list(C.BACKGROUNDS), key="background_select",
                              help="Weiß: flach. Tiefpass, Resonanz, Steil: farbiges Rauschen mit exakt bekannter Dichte.")
    st.markdown("**Abtastung**")
    sampling = st.selectbox("Welche Werte bleiben", list(C.SAMPLING), key="sampling_select",
                            help="Aus den 1000 Hz werden die angegebenen Werte genommen; mit Tiefpass wird vorher gefiltert.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "sampling_select": sampling, "line_slider": float(line), "snr_slider": int(snr),
                   "background_select": background, "seed_input": int(seed)})

settings = Settings(int(n), sampling, float(line), int(snr), background, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scheinbare Frequenz der Linie", f"{de(a.apparent, 1)} Hz", delta="wahre Frequenz" if a.apparent == settings.line else f"wahre: {de(settings.line, 1)} Hz",
          delta_color="off")
m2.metric("Höchster Wert bei", f"{de(a.peak_f, 1)} Hz", delta="die Linie" if a.line_is_max else "nicht die Linie", delta_color="off")
m3.metric("Linie über dem Untergrund", f"{de(10 * math.log10(a.line_peak / a.floor), 1)} dB", delta=f"{a.higher} Bins darüber", delta_color="off")
m4.metric("Std / Mittel (gegen Erwartung)", de(a.cv, 2), delta="Exponentialverteilung: 1,00", delta_color="off")
st.plotly_chart(build_periodogram(a), width="stretch", key="periodogram_chart")
st.caption("Blau: das Periodogramm dieser Folge. Orange: sein exakter Erwartungswert. Grün gepunktet: die wahre Dichte des Hintergrunds. "
           "Rot gestrichelt: die scheinbare Frequenz der Linie. Der Zackenteppich liegt im Mittel 2,5 dB unter dem Erwartungswert und streut um 5,6 dB (Logarithmus).")
st.plotly_chart(build_signal(a), width="stretch", key="signal_chart")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="periodogramm.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wird das Periodogramm mit mehr Daten ruhiger?")
kind, text = ev.verdict_spread(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Je Experiment feste Zufallsströme (Seed 100000 und folgende), unabhängig von den Reglern. Die Rechnung dauert einige Sekunden und läuft erst auf Klick.")
if st.button("🔬 Experimente rechnen", key="run_experiments"):
    st.session_state["experiments_ready"] = True
if st.session_state.get("experiments_ready"):
    with st.spinner("Rechne die Experimente ..."):
        exp = _experiments()

    st.markdown("#### Mehr Daten, gleiche Streuung")
    st.caption("Weißes Rauschen, je Länge 800 / 400 / 200 / 100 Läufe; ohne Bin 0 und Nyquist-Bin.")
    st.plotly_chart(build_variation(exp["variation"]), width="stretch", key="exp_variation")
    st.dataframe(pd.DataFrame([{"n": r["n"], "Läufe": r["runs"], "Mittel I/S": de(r["mean"], 2), "Std/Mittel": de(r["cv"], 2), "P(I > 3 S)": de(100 * r["p_gt3"], 1) + " %",
                                "Bias des Logarithmus (dB)": de(r["db_bias"], 2), "Std des Logarithmus (dB)": de(r["db_std"], 2)} for r in exp["variation"]]),
                 hide_index=True, width="stretch")
    kind, text = ev.verdict_variation(exp["variation"])
    getattr(st, kind)(text)

    st.markdown("#### Was mit n wächst: die Linie, nicht der Untergrund")
    st.caption(f"Weißes Rauschen, Linie mit {C.EXP_LINE_SNR} dB je Abtastwert, zufällige Frequenz und Phase, je Länge {C.EXP_LINE_RUNS} Läufe.")
    st.plotly_chart(build_line(exp["line"]), width="stretch", key="exp_line")
    kind, text = ev.verdict_line(exp["line"])
    getattr(st, kind)(text)

    st.markdown("#### Wo der Mittelwert falsch liegt")
    st.caption(f"Hintergrund „Steil“ (56,7 dB Spannweite), Rechteckfenster, je Länge {C.EXP_BIAS_RUNS} Läufe; Simulation gegen die exakte Formel.")
    st.plotly_chart(build_bias(exp["bias"]), width="stretch", key="exp_bias")
    st.dataframe(pd.DataFrame([{"Linie ... Bins neben dem Gitterpunkt": de(r["offset"], 2), "Verlust des Gipfels (dB)": de(r["loss_db"], 2)} for r in exp["scallop"]]),
                 hide_index=True, width="stretch")
    kind, text = ev.verdict_bias(exp["bias"], exp["scallop"])
    getattr(st, kind)(text)

    st.markdown("#### Abtastung: Faltung des Rauschens")
    st.caption(f"Hintergrund „Tiefpass“, {C.EXP_SAMPLING_N} Werte, je Faktor {C.EXP_SAMPLING_RUNS} Läufe, oberes Band "
               f"({de(100 * C.EXP_SAMPLING_BAND[0], 0)} bis {de(100 * C.EXP_SAMPLING_BAND[1], 0)} % der neuen Nyquist-Frequenz).")
    st.plotly_chart(build_sampling(exp["sampling"]), width="stretch", key="exp_sampling")
    st.dataframe(pd.DataFrame([{"Faktor": r["q"], "wahr (dB)": de(r["true_db"], 1), "gefaltet, exakt (dB)": de(r["folded_db"], 1), "gemessen ohne Tiefpass (dB)": de(r["raw_db"], 1),
                                "mit Tiefpass, exakt (dB)": de(r["filtered_oracle_db"], 1), "gemessen mit Tiefpass (dB)": de(r["filtered_db"], 1)} for r in exp["sampling"]]),
                 hide_index=True, width="stretch")
    kind, text = ev.verdict_sampling(exp["sampling"])
    getattr(st, kind)(text)

st.markdown("---")

# --- Rechenaufwand -------------------------------------------------------------------------------------------------------------------------

st.subheader("⚙️ Rechenaufwand")
st.markdown(
    "Die FFT braucht für **jede** Länge n log n Schritte, nicht nur für Zweierpotenzen; bei Längen mit großem Primfaktor wächst nur die Konstante. "
    "Gezählt werden komplexe Multiplikationen, keine Sekunden (die hängen am Rechner)."
)
rows = ev.cost_table()
st.dataframe(pd.DataFrame([{"Länge n": f"{r['n']:,}".replace(",", "."), "direkte DFT (n²)": f"{r['direct']:,}".replace(",", "."),
                            "Radix-2-FFT": f"{r['radix2']:,}".replace(",", "."),
                            f"Bluestein bei n + 1 = {r['n_odd']:,}".replace(",", "."): f"{r['bluestein_odd']:,}".replace(",", ".")} for r in rows]),
             hide_index=True, width="stretch")
st.caption("Bluestein ist eine Näherung nach dem Lehrbuch: drei Radix-2-FFTs der nächsten Zweierpotenz ab 2n − 1 plus 2n + M Multiplikationen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Stationär**: Das Spektrum ändert sich über die Aufzeichnung nicht | Eine wandernde Frequenz hinterlässt eine Spur statt eines Gipfels, das Periodogramm sagt nicht, wann (in dieser Demo nicht gemessen). | Spektrogramm |
| **Nur Rechteckfenster** | Leckage hebt die Täler: im Hintergrund „Steil“ liegt der Mittelwert bei n = 128 um 17,9 dB über der wahren Dichte. | Gitter, Fenster, Linienlage |
| **Gleichmäßige Abtastzeitpunkte** | Bei Lücken oder unregelmäßigen Zeiten gilt die FFT-Formel nicht. | Lomb-Scargle |
| **Eine Aufzeichnung** | Die Streuung je Bin bleibt; Mitteln über Segmente oder Taper senkt sie, kostet aber Auflösung. | Bartlett und Welch, Multitaper |
| **Ein Generator** | Linien plus AR-Hintergrund mit Gauß- oder gleichverteiltem Rauschen. Reale Spektren haben Linienscharen, Drift und Ausreißer. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Periodogramm.** Für $x[0], \dots, x[n-1]$ mit Abtastrate $f_s$ ist
$I(f) = \dfrac{1}{n f_s}\Big|\sum_{m=0}^{n-1} x[m]\,e^{-2\pi i f m / f_s}\Big|^2$ an den Bins $f_k = k f_s / n$.

**Erwartungswert (exakt).** Mit Autokovarianz $r(\tau)$ des Rauschens und der deterministischen Folge $s$ der Linien:
$E[I(f)] = \dfrac{1}{n f_s}\Big(|S(f)|^2 + \sum_{|\tau| < n} (n - |\tau|)\, r(\tau)\cos(2\pi f \tau / f_s)\Big)$, $S(f) = \sum s[m] e^{-2\pi i f m / f_s}$.
Das ist die Faltung der wahren Dichte mit dem Fejér-Kern: Leckage ist Bias.

**Verteilung.** Bei Gauß'schem Rauschen und $f$ im Inneren ist $I(f)/S(f)$ exponentialverteilt mit Mittelwert 1 und Standardabweichung 1;
$10\log_{10}$ davon hat den Mittelwert $-10\log_{10}(e)\,\gamma = -2{,}51$ dB und die Standardabweichung $10\log_{10}(e)\,\pi/\sqrt{6} = 5{,}57$ dB.

**Aliasing.** Nimmt man jeden $q$-ten Wert (nach einem Filter $H$), ist die Dichte der Folge
$S_y(f) = \sum_{k} |H(f + k f_s/q)|^2\, S(f + k f_s/q)$, summiert über alle Spiegelbilder innerhalb der alten Nyquist-Grenze;
eine Linie bei $f$ erscheint bei $|f - f_s\,\mathrm{round}(f/f_s)|$.

**Quellen.** Lehrbuchwissen (u. a. Percival und Walden, *Spectral Analysis for Physical Applications*, 1993). Die Formeln hier sind per Simulation und
gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
