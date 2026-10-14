"""Kennzahlen, Verdikt-Logik und Experimente (Sweeps über feste Ströme) der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import pgm_constants as C
import pgm_spectrum as S
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    sampling: str = C.DEFAULT_SAMPLING
    line: float = C.DEFAULT_LINE
    snr: int = C.DEFAULT_SNR
    background: str = C.DEFAULT_BACKGROUND
    seed: int = C.DEFAULT_SEED

    @property
    def q(self) -> int:
        return C.SAMPLING_Q[self.sampling][0]

    @property
    def antialias(self) -> bool:
        return C.SAMPLING_Q[self.sampling][1]


def noise_spec(background: str):
    """(Pole, Innovations-Sigma, weißes Sigma) eines Hintergrunds, auf Varianz 1 normiert."""
    poles = C.BACKGROUNDS[background]
    if poles:
        return poles, G.ar_unit_sigma(G.ar_from_poles(poles)), 0.0
    return (), 0.0, 1.0


def line_amplitude(snr_db: float) -> float:
    """Amplitude A mit Linienleistung A^2 / 2 = 10^(snr/10) gegen Rauschvarianz 1."""
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    fs: float
    f: np.ndarray
    x: np.ndarray
    I: np.ndarray
    expected: np.ndarray        # E[I] exakt (Rauschen über die Autokovarianz, Linie über die gewonnene Folge)
    true_density: np.ndarray    # wahre Dichte des Hintergrunds bei der Rate vor der Abtastung
    sampled_density: np.ndarray  # Dichte der gewonnenen Folge (gefaltet und gefiltert)
    apparent: float             # scheinbare Frequenz der Linie in Hz
    peak_f: float               # Frequenz des höchsten Periodogrammwerts
    line_is_max: bool
    line_peak: float            # höchster Wert innerhalb eines Bins um die scheinbare Frequenz
    floor: float                # Mittel der übrigen Bins
    higher: int                 # Bins ohne Linie, die höher liegen als der Linienwert
    cv: float                   # Std / Mittel von I / E[I] über die inneren Bins


def acquire_signal(s: Settings):
    """(x, clean) der gewonnenen Folge; Zufallsverbrauch: Phase (1 Gleichverteilter), dann Generator."""
    rng = G.SplitMix64(s.seed)
    phase = 2.0 * math.pi * float(rng.uniform(1)[0])
    poles, ar_sigma, white = noise_spec(s.background)
    h = S.antialias_taps(s.q) if s.antialias else None
    n_hi = S.hi_length(s.n, s.q, h)
    sig = G.make_signal(n_hi, C.FS_HI, [(s.line, line_amplitude(s.snr), phase)], poles, ar_sigma, white, rng)
    return S.acquire(sig.x, s.n, s.q, h), S.acquire(sig.clean, s.n, s.q, h), h


def analyse(s: Settings) -> Analysis:
    x, clean, h = acquire_signal(s)
    poles, ar_sigma, white = noise_spec(s.background)
    fs = C.FS_HI / s.q
    f = S.bin_freqs(s.n, fs)
    I = S.periodogram(x, fs)
    expected = S.expected_sampled_periodogram(f, fs, s.n, s.q, h, poles, ar_sigma, white, clean)
    true_density = G.true_psd(f, C.FS_HI, poles, ar_sigma, white)
    sampled_density = S.sampled_psd(f, C.FS_HI, s.q, h, poles, ar_sigma, white)
    apparent = float(S.apparent_frequency(s.line, fs))
    inner = S.inner_bins(s.n)
    ratio = I[inner] / expected[inner]
    cv = float(ratio.std() / ratio.mean())
    bin_hz = fs / s.n
    near = np.abs(f - apparent) <= 1.5 * bin_hz
    near_i = near.copy()
    near_i[0] = False
    idx = np.arange(len(f))
    inner_mask = np.zeros(len(f), dtype=bool)
    inner_mask[inner] = True
    peak_idx = int(np.argmax(np.where(inner_mask, I, -np.inf)))
    line_peak = float(I[near_i & inner_mask].max()) if (near_i & inner_mask).any() else float(I[inner_mask].max())
    rest = inner_mask & (np.abs(f - apparent) > 3.0 * bin_hz)
    floor = float(I[rest].mean()) if rest.any() else float(I[inner_mask].mean())
    higher = int((I[rest] > line_peak).sum())
    return Analysis(s, fs, f, x, I, expected, true_density, sampled_density, apparent, float(f[peak_idx]), bool(abs(f[peak_idx] - apparent) <= 1.5 * bin_hz),
                    line_peak, floor, higher, cv)


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _inner(I: np.ndarray, n: int) -> np.ndarray:
    return I[..., S.inner_bins(n)]


def sweep_variation(ns=C.EXP_N_SWEEP) -> list[dict]:
    """Weißes Rauschen: Verteilung von I / S je Bin gegen die Länge n (S = 1 / fs)."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED)
    for n in ns:
        runs = C.EXP_RUNS[n]
        x = G.make_noise_batch(runs, n, (), 0.0, 1.0, rng)
        r = _inner(S.periodogram(x, C.FS_HI), n) * C.FS_HI          # I / S mit S = 1 / fs
        mean_db, std_db = S.log_stats(r)
        rows.append({"n": n, "runs": runs, "mean": float(r.mean()), "cv": float(r.std() / r.mean()), "p_gt3": float((r > 3.0).mean()),
                     "db_bias": mean_db, "db_std": std_db})
    return rows


def sweep_line(ns=C.EXP_N_SWEEP, snr_db: float = C.EXP_LINE_SNR, runs: int = C.EXP_LINE_RUNS) -> list[dict]:
    """Linie (zufällige Frequenz und Phase) in weißem Rauschen: Linie gegen Untergrund und Anteil der Läufe, in denen sie das Maximum ist."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED + 1)
    amp = line_amplitude(snr_db)
    for n in ns:
        f0 = (0.05 + 0.4 * rng.uniform(runs)) * C.FS_HI
        phase = 2.0 * math.pi * rng.uniform(runs)
        t = np.arange(n) / C.FS_HI
        x = amp * np.cos(2.0 * np.pi * f0[:, None] * t[None, :] + phase[:, None]) + G.make_noise_batch(runs, n, (), 0.0, 1.0, rng)
        I = S.periodogram(x, C.FS_HI)
        k0 = np.round(f0 / (C.FS_HI / n)).astype(int)
        peaks, floors, is_max, higher = [], [], [], []
        for i in range(runs):
            lo, hi = max(k0[i] - 1, 1), k0[i] + 2
            pk = float(I[i, lo:hi].max())
            mask = np.ones(I.shape[1], dtype=bool)
            mask[0] = False
            mask[-1] = n % 2 == 1
            mask[max(k0[i] - 2, 0):k0[i] + 3] = False
            rest = I[i, mask]
            peaks.append(pk)
            floors.append(float(rest.mean()))
            is_max.append(pk > float(rest.max()))
            higher.append(int((rest > pk).sum()))
        rows.append({"n": n, "runs": runs, "over_floor_db": 10.0 * math.log10(np.mean(peaks) / np.mean(floors)), "p_max": float(np.mean(is_max)),
                     "higher": float(np.mean(higher))})
    return rows


def sweep_bias(ns=C.EXP_BIAS_N, background: str = "Steil", runs: int = C.EXP_BIAS_RUNS, probes=C.EXP_BIAS_PROBES) -> list[dict]:
    """Erwartungswert des Periodogramms gegen wahre Dichte an festen Frequenzen; Simulation gegen die exakte Formel."""
    rows = []
    rng = G.SplitMix64(C.SWEEP_SEED + 2)
    poles, ar_sigma, white = noise_spec(background)
    for n in ns:
        x = G.make_noise_batch(runs, n, poles, ar_sigma, white, rng)
        I = S.periodogram(x, C.FS_HI)
        r = G.noise_autocov(poles, ar_sigma, white, n)
        for p in probes:
            k = int(round(p * n))
            f = k * C.FS_HI / n
            e = float(G.expected_periodogram([f], C.FS_HI, r)[0])
            s_true = float(G.true_psd([f], C.FS_HI, poles, ar_sigma, white)[0])
            sim = I[:, k]
            rows.append({"n": n, "frac": p, "f": f, "sim_vs_formula_db": 10.0 * math.log10(sim.mean() / e),
                         "z": float((sim.mean() - e) / (sim.std() / math.sqrt(runs))), "bias_db": 10.0 * math.log10(e / s_true)})
    return rows


def sweep_scalloping(n: int = 1024, offsets=(0.0, 0.25, 0.5), runs: int = 100) -> list[dict]:
    """Höhe des Periodogrammgipfels einer reinen Linie, die um `offset` Bins neben einem Gitterpunkt liegt, gegen die Linie auf dem Gitterpunkt."""
    rng = G.SplitMix64(C.SWEEP_SEED + 3)
    phase = 2.0 * math.pi * rng.uniform(runs)
    t = np.arange(n) / C.FS_HI
    peaks = []
    for off in offsets:
        f0 = (100 + off) * C.FS_HI / n
        x = np.cos(2.0 * np.pi * f0 * t[None, :] + phase[:, None])
        peaks.append(float(S.periodogram(x, C.FS_HI)[:, 98:104].max(axis=1).mean()))
    return [{"offset": off, "loss_db": 10.0 * math.log10(pk / peaks[0])} for off, pk in zip(offsets, peaks)]


def sweep_sampling(qs=C.EXP_SAMPLING_Q, n: int = C.EXP_SAMPLING_N, runs: int = C.EXP_SAMPLING_RUNS, background: str = "Tiefpass",
                   band=C.EXP_SAMPLING_BAND) -> list[dict]:
    """Rauschfaltung im oberen Band der neuen Rate: wahr, gefaltet (exakt), gemessen ohne und mit Tiefpass."""
    poles, ar_sigma, white = noise_spec(background)
    rng = G.SplitMix64(C.SWEEP_SEED + 4)
    rows = []
    for q in qs:
        fs = C.FS_HI / q
        f = S.bin_freqs(n, fs)
        sel = (f >= band[0] * fs / 2) & (f <= band[1] * fs / 2)
        h = S.antialias_taps(q)
        n_hi = S.hi_length(n, q, h)
        x_hi = G.make_noise_batch(runs, n_hi, poles, ar_sigma, white, rng)
        raw = np.stack([S.acquire(x_hi[i], n, q, None) for i in range(runs)])
        flt = np.stack([S.acquire(x_hi[i], n, q, h) for i in range(runs)])
        true = float(G.true_psd(f[sel], C.FS_HI, poles, ar_sigma, white).mean())
        folded = float(S.sampled_psd(f[sel], C.FS_HI, q, None, poles, ar_sigma, white).mean())
        filtered = float(S.sampled_psd(f[sel], C.FS_HI, q, h, poles, ar_sigma, white).mean())
        rows.append({"q": q, "true_db": 10.0 * math.log10(true), "folded_db": 10.0 * math.log10(folded),
                     "raw_db": 10.0 * math.log10(S.periodogram(raw, fs)[:, sel].mean()), "filtered_oracle_db": 10.0 * math.log10(filtered),
                     "filtered_db": 10.0 * math.log10(S.periodogram(flt, fs)[:, sel].mean())})
    return rows


def cost_table(ns=(256, 1024, 4096, 65536)) -> list[dict]:
    """Gezählte komplexe Multiplikationen: direkt, Radix-2 (Zweierpotenzen) und Bluestein für die nächste größere ungerade Länge."""
    rows = []
    for n in ns:
        odd = n + 1
        rows.append({"n": n, "direct": S.ops_direct(n), "radix2": S.ops_radix2(n), "n_odd": odd, "bluestein_odd": S.ops_bluestein(odd)})
    return rows


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_spread(a: Analysis) -> tuple[str, str]:
    kind = "warning" if a.cv > 0.8 else "info"
    return kind, (f"Die Werte streuen um ihren Erwartungswert mit Std/Mittel **{de(a.cv, 2)}**; ein Exponentialverteilter hat genau 1,00. "
                  f"Mehr Werte (n) ändern daran nichts, sie machen nur die Frequenzachse feiner ({a.settings.n // 2 + 1} Bins).")


def verdict_variation(rows: list[dict]) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "warning", (f"Std/Mittel je Bin ist {de(lo['cv'], 2)} bei n = {lo['n']} und {de(hi['cv'], 2)} bei n = {hi['n']}: **keine Beruhigung**. "
                       f"Der Logarithmus streut um {de(hi['db_std'], 1)} dB (Theorie 5,6 dB) und liegt im Mittel {de(abs(hi['db_bias']), 1)} dB unter dem Erwartungswert.")


def verdict_line(rows: list[dict]) -> tuple[str, str]:
    lo, hi = rows[0], rows[-1]
    return "info", (f"Die Linie ragt bei n = {lo['n']} um {de(lo['over_floor_db'], 1)} dB über den Untergrund, bei n = {hi['n']} um {de(hi['over_floor_db'], 1)} dB "
                    f"(+10 log10 n). Sie ist in {de(100 * lo['p_max'], 0)} % bzw. {de(100 * hi['p_max'], 0)} % der Läufe der höchste Wert: "
                    "**Linien gewinnen mit n, der Untergrund nicht.**")


def verdict_bias(rows: list[dict], scallop: list[dict]) -> tuple[str, str]:
    deep = [r for r in rows if r["frac"] == max(C.EXP_BIAS_PROBES)]
    small, large = deep[0], deep[-1]
    half = scallop[-1]
    return "info", (f"Im Tal des Spektrums liegt der Mittelwert um {de(small['bias_db'], 1)} dB (n = {small['n']}) bzw. {de(large['bias_db'], 1)} dB (n = {large['n']}) "
                    f"über der wahren Dichte: die Leckage des Rechteckfensters, sie fällt mit n. Eine Linie ein halbes Bin neben dem Gitterpunkt verliert "
                    f"{de(abs(half['loss_db']), 1)} dB. Stück 2 zeigt, was man dagegen tut.")


def verdict_sampling(rows: list[dict]) -> tuple[str, str]:
    r = rows[-1]
    return "warning", (f"Bei jedem {r['q']}. Wert liegt das obere Band nach der Abtastung um {de(r['raw_db'] - r['true_db'], 1)} dB zu hoch (gemessen {de(r['raw_db'], 1)} dB, "
                       f"wahr {de(r['true_db'], 1)} dB, gefaltet erwartet {de(r['folded_db'], 1)} dB). Ein Tiefpass vorher senkt es auf {de(r['filtered_db'], 1)} dB, "
                       "kostet aber Nutzband am Rand.")
