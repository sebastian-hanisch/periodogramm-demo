"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets (Kern: pgm_spectrum.py, Generator: signal_generator.py)."""

FS_HI = 1000.0                      # Abtastrate des Messsignals vor einer Unterabtastung in Hz (feste Annahme; Nyquist 500 Hz)
N_MIN, N_MAX, N_STEP, DEFAULT_N = 64, 4096, 64, 512      # Länge der gewonnenen Folge; Kosten: O(n * q) Erzeugung, O(n log n) FFT
LINE_MIN, LINE_MAX, DEFAULT_LINE = 10.0, 480.0, 123.5     # Linienfrequenz in Hz (vor der Abtastung, unterhalb von FS_HI / 2)
SNR_MIN, SNR_MAX, DEFAULT_SNR = -25, 10, -10              # Leistung der Linie A^2/2 gegen die Rauschvarianz 1, in dB, je Abtastwert
SEED_MAX, DEFAULT_SEED = 999999, 35

SAMPLING = ("jeder Wert", "jeder 2.", "jeder 4.", "jeder 8.", "jeder 2. mit Tiefpass", "jeder 4. mit Tiefpass", "jeder 8. mit Tiefpass")
DEFAULT_SAMPLING = SAMPLING[0]
SAMPLING_Q = {"jeder Wert": (1, False), "jeder 2.": (2, False), "jeder 4.": (4, False), "jeder 8.": (8, False),
              "jeder 2. mit Tiefpass": (2, True), "jeder 4. mit Tiefpass": (4, True), "jeder 8. mit Tiefpass": (8, True)}

# Hintergrund: Name -> Pole (Frequenz als Bruchteil der Abtastrate FS_HI, Radius); alle auf Varianz 1 normiert (weiß: sigma_w = 1)
BACKGROUNDS = {
    "weiß": (),
    "Tiefpass": ((0.0, 0.9),),
    "Resonanz": ((0.2, 0.97),),
    "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9)),     # 56,7 dB Dynamik, die Leckage des Rechteckfensters sichtbar macht
}
DEFAULT_BACKGROUND = "weiß"

# Experimente (feste Ströme, nie ein einzelner Lauf als Befund)
SWEEP_SEED = 100000
EXP_N_SWEEP = (64, 256, 1024, 4096)
EXP_RUNS = {64: 800, 256: 400, 1024: 200, 4096: 100}     # Läufe je Länge (weißes Rauschen)
EXP_LINE_SNR = -10                                       # dB je Abtastwert
EXP_LINE_RUNS = 300
EXP_BIAS_N = (128, 512, 2048)
EXP_BIAS_RUNS = 400
EXP_BIAS_PROBES = (0.02, 0.1, 0.2, 0.3, 0.4, 0.49)       # Bruchteile der Abtastrate
EXP_SAMPLING_N = 512
EXP_SAMPLING_RUNS = 100
EXP_SAMPLING_Q = (2, 4, 8)
EXP_SAMPLING_BAND = (0.8, 0.95)                          # oberes Band, in Bruchteilen der Nyquist-Frequenz der neuen Rate

COLOR_I = "rgba(76,120,168,0.75)"
COLOR_EXPECTED = "#E8833A"
COLOR_TRUE = "#2F6B65"
COLOR_LINE = "#B03A2E"

PRESETS = {
    "Eine Linie im Rauschen": {"n": DEFAULT_N, "sampling": "jeder Wert", "line": DEFAULT_LINE, "snr": -10, "background": "weiß", "seed": DEFAULT_SEED},
    "Farbiger Hintergrund": {"n": 1024, "sampling": "jeder Wert", "line": 300.0, "snr": -5, "background": "Steil", "seed": DEFAULT_SEED},
    "Zu langsam abgetastet": {"n": DEFAULT_N, "sampling": "jeder 2.", "line": 300.0, "snr": 0, "background": "Tiefpass", "seed": DEFAULT_SEED},
    "Mehr Daten": {"n": N_MAX, "sampling": "jeder Wert", "line": DEFAULT_LINE, "snr": -10, "background": "weiß", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Eine Linie im Rauschen": "512 Werte, eine Linie bei 123,5 Hz mit -10 dB gegen weißes Rauschen: das Periodogramm zeigt sie, aber auch viele Zacken, die nur Rauschen sind.",
    "Farbiger Hintergrund": "Steil abfallender Hintergrund mit 57 dB Spannweite: Im Tal liegt das Periodogramm weit über der wahren Dichte (Leckage des Rechteckfensters).",
    "Zu langsam abgetastet": "Jeder 2. Wert ohne Tiefpass: Die Linie bei 300 Hz liegt über der neuen Nyquist-Frequenz (250 Hz) und erscheint bei 200 Hz; das Rauschen faltet sich ins Band.",
    "Mehr Daten": "4096 statt 512 Werte: Die Linie ragt höher heraus, der Untergrund ist aber so zackig wie zuvor.",
}
