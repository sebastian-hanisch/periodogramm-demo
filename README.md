# Das Periodogramm: je mehr Daten, desto feiner, aber nicht ruhiger – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-periodogramm-demo.streamlit.app/)** -->

Wurzel der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Ein simuliertes Signal mit **bekanntem Spektrum** (eine Linie plus farbiger Hintergrund) wird abgetastet, sein Periodogramm berechnet und gegen den
**exakten Erwartungswert** gelegt. Weil die Wahrheit bekannt ist, lässt sich jede Abweichung messen.

**Namensfalle:** Mit „Spectral Clustering“ (`spectral-demo`, Eigenwerte eines Graphen) hat diese Demo nichts zu tun. Hier geht es um Frequenzen eines Signals.

## Warum dieses Problem

Das Periodogramm ist die direkte Antwort der Fourier-Transformation auf „bei welcher Frequenz steckt wie viel Leistung“. Man erwartet, dass es mit
mehr Messwerten ruhiger wird. Es wird feiner, aber nicht ruhiger: **Jeder Wert streut um seinen Erwartungswert so stark wie dessen Betrag**, bei 64
Messwerten wie bei 4096. Daran hängt die ganze Reihe: Mitteln über Segmente (Bartlett und Welch), Taper (Multitaper) und parametrische Modelle
(AR-Spektrum) senken die Streuung, jedes auf eigene Kosten.

## Modell

Messwerte $x[m] = \sum_j A_j \cos(2\pi f_j m / f_s + \varphi_j) + \text{Rauschen}$; das Rauschen ist weiß oder ein AR-Prozess mit Polen in Frequenz und Radius
(Varianz auf 1 normiert). Die Rate vor der Abtastung ist fest 1000 Hz. Das **Periodogramm** ist $I(f) = |\sum x[m]e^{-2\pi i f m/f_s}|^2 / (n f_s)$ mit
Rechteckfenster; sein Erwartungswert ist über die Autokovarianz exakt berechenbar (Fejér-Faltung). Formeln im 📐-Expander der App.

## Methodik

Ein Signal, vier Hintergründe (weiß, Tiefpass, Resonanz, Steil), sieben Abtastungen (jeder Wert, jeder 2./4./8., je mit oder ohne Tiefpass vorher).
Verglichen wird das Periodogramm mit seinem exakten Erwartungswert, mit der wahren Dichte vor der Abtastung und mit der Dichte der gewonnenen Folge
(gefaltet, gegebenenfalls gefiltert). Vier Experimente laufen auf Klick über feste Zufallsströme.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`.

| Frage | Befund |
|---|---|
| Wird die Streuung je Bin mit mehr Daten kleiner? (weißes Rauschen, n = 64 bis 4096) | **Nein:** Std/Mittel von I/S ist 1,00 bei jeder Länge (Exponentialverteilung); P(I > 3 S) liegt bei 5,0 %. |
| Wie liegt das Periodogramm im Logarithmus? | Mittel −2,5 dB unter dem Erwartungswert, Streuung 5,6 dB (Theorie: −2,51 und 5,57 dB), unabhängig von n. |
| Was wächst mit n? | Die Linie: 6 / 10 / 16 / 22 dB über dem Untergrund bei n = 64 / 256 / 1024 / 4096 (−10 dB je Abtastwert), also +10·log₁₀ n; sie ist in rund 45 / 90 / 100 / 100 % der Läufe der höchste Wert. |
| Wo liegt der Mittelwert falsch? (Hintergrund „Steil“, 57 dB Spannweite) | Im Tal bei 0,49 f_s um 17,9 / 11,9 / 6,7 dB über der wahren Dichte (n = 128 / 512 / 2048): Leckage des Rechteckfensters, sie fällt mit n. Am Peak (0,2 f_s) liegt der Fehler ab n = 512 unter 1 dB. |
| Wie viel verliert eine Linie zwischen zwei Bins? | ¼ Bin neben dem Gitterpunkt −0,9 dB, ½ Bin −3,9 dB (sinc²). |
| Was tut zu langsames Abtasten mit dem Rauschen? (Hintergrund „Tiefpass“, oberes Band) | Es liegt nach jedem 2. / 4. / 8. Wert um 2,2 / 2,8 / 3,1 dB zu hoch; das gemessene Niveau stimmt mit der gefalteten Dichte überein (innerhalb 0,35 dB). |
| Was kostet der Tiefpass? | −6 dB bei 0,9 der neuen Nyquist-Frequenz (Vorgabe); bei der Nyquist-Frequenz selbst −14,5 dB (Faktor 8) bzw. −29,9 dB (Faktor 4), weil die Länge fest 129 Taps ist. |
| Rechenaufwand | n = 1024: 1.048.576 komplexe Multiplikationen direkt, 5.120 mit Radix-2-FFT. |

Eine Beobachtung ohne Test (plattformabhängig): Auf dem Entwicklungsrechner braucht numpy für eine Primzahllänge um 65 000 etwa das Sechsfache einer
Zweierpotenz-Länge, bleibt aber bei n log n.

## Ehrliche Grenzen

- **Nur Rechteckfenster.** Die Leckage ist hier ein Befund, kein Gegenmittel; das Fenster ist das Thema der Folgedemo „Gitter, Fenster, Linienlage“.
- **Ein Generator.** Linie plus AR- oder weißer Hintergrund mit normalverteiltem Rauschen. Reale Spektren haben Linienscharen, Drift und Ausreißer.
- **Stationär, gleichmäßig abgetastet.** Wandernde Frequenzen (Spektrogramm) und unregelmäßige Zeitpunkte (Lomb-Scargle) kommen in Folgedemos.
- **„Wahr“ heißt: wahr im Modell.** Die Dichte ist die des Generators.
- **Der Standardfehler der Experimente** liegt bei einigen Prozent (Läufe 100 bis 800 je Länge); kleine Unterschiede sind nicht belastbar.

## Verifikation

- **Kern gegen Handrechnung:** Mini-Instanz N = 8 (Cosinus bei f_s/4: I₂ = 2, Parseval), Aliasing 7 Hz gegen 1 Hz bei 8 Hz Abtastrate, Operationszählung.
- **Orakel (`tests/test_oracle_periodogramm.py`):** Periodogramm gegen `scipy.signal.periodogram` (gerade und ungerade Längen); AR-Rauschen gegen das
  rekursive Filter `scipy.signal.lfilter` auf denselben Innovationen; Autokovarianz gegen numerische Integration der Dichte; Verteilung (Kolmogorov-Smirnov
  gegen Exp(1)); exakter Erwartungswert gegen Simulation (z-Werte), mit und ohne Tiefpass.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Pole und Koeffizienten von Hand, Dichte integriert zur Varianz,
  Erwartungswert gegen die Doppelsummen-Definition.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an handgesetzten Werten (`tests/test_verdicts.py`).
- **Presets:** vollständig, Werte in den Grenzen, jedes zeigt seinen Effekt (`tests/test_presets.py`); Randfälle (`tests/test_edge_cases.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Auswahlfelder, Permalink, Experimente auf Klick, Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (Strom, Pole, AR-Filter, exakte Dichte und Erwartungswert) |
| `pgm_spectrum.py` | Kern: Periodogramm, Abtastung, Tiefpass, gefaltete Dichte, Operationszählung |
| `pgm_constants.py` | Regler-Grenzen, feste Annahmen, Hintergründe, Presets |
| `pgm_evaluation.py` | Analyse, Kennzahlen, Experimente, Verdikte |
| `pgm_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `pgm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `pgm_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Generator, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Fenster, Mittelung, Taper:** Gegenmittel gegen Leckage und Streuung sind die Themen der Folgedemos der Reihe, nicht dieser.
- **Unregelmäßige Zeitpunkte, Zeit-Frequenz-Darstellungen, Mehrkanal:** gleichfalls Folgedemos.
- **Messung der Rechenzeit in der App:** Sekunden hängen am Rechner; die App zählt Operationen.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
