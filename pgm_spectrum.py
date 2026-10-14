"""Kern: Periodogramm, Abtastung (Unterabtastung mit und ohne Tiefpass), scheinbare Frequenz, exakte Orakel und Operationszählung.

Reine Funktionen, kein Zufall (der Generator liefert die Signale), kein Streamlit. Konvention: zweiseitige Leistungsdichte in Leistung je Hz,
Periodogramm I(f) = |sum x e^{-iwn}|^2 / (n fs) an den Bins f_k = k fs / n, k = 0 .. n/2.
"""

import math

import numpy as np

import signal_generator as G

TAPS = 129            # Länge des Tiefpasses; Gruppenlaufzeit (TAPS - 1) / 2 Abtastwerte
KAISER_BETA = 8.0
CUTOFF = 0.9          # -6 dB bei 0,9 mal der neuen Nyquist-Frequenz


def bin_freqs(n: int, fs: float) -> np.ndarray:
    """Frequenzen der Bins 0 .. n // 2 in Hz."""
    return np.arange(n // 2 + 1) * fs / n


def inner_bins(n: int) -> slice:
    """Bins mit zwei Freiheitsgraden: ohne Gleichanteil (k = 0) und, bei geradem n, ohne den Nyquist-Bin (k = n / 2)."""
    return slice(1, n // 2) if n % 2 == 0 else slice(1, n // 2 + 1)


def periodogram(x, fs: float) -> np.ndarray:
    """Rechteckfenster-Periodogramm |X_k|^2 / (n fs) für k = 0 .. n // 2 (letzte Achse; auch für runs x n)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return np.abs(np.fft.rfft(x, axis=-1)) ** 2 / (n * fs)


def apparent_frequency(f, fs: float):
    """Scheinbare Frequenz einer Linie bei f nach Abtastung mit fs: Abstand zum nächsten Vielfachen von fs."""
    f = np.asarray(f, dtype=float)
    return np.abs(f - fs * np.round(f / fs))


def antialias_taps(q: int, ntaps: int = TAPS, beta: float = KAISER_BETA, cutoff: float = CUTOFF) -> np.ndarray:
    """Tiefpass vor der Unterabtastung um q: Kaiser-Fenster-Sinc, -6 dB bei `cutoff` mal der Nyquist-Frequenz der neuen Rate, Gleichanteil 1."""
    m = np.arange(ntaps) - (ntaps - 1) / 2.0
    c = cutoff / q                                   # Grenzfrequenz relativ zur Nyquist-Frequenz der alten Rate
    h = (c / 2.0) * np.sinc(c * m) * np.kaiser(ntaps, beta)
    return h / h.sum()


def hi_length(n: int, q: int, h) -> int:
    """Länge der hoch abgetasteten Folge, aus der n Werte entstehen."""
    return n * q + (0 if h is None else len(h) - 1)


def acquire(x_hi, n: int, q: int, h=None) -> np.ndarray:
    """n Werte aus der hoch abgetasteten Folge: jeder q-te Wert, mit Tiefpass `h` vorher oder ohne (h = None)."""
    x_hi = np.asarray(x_hi, dtype=float)
    if len(x_hi) < hi_length(n, q, h):
        raise ValueError("hoch abgetastete Folge zu kurz")
    if h is None:
        return x_hi[: n * q : q].copy()
    y = np.convolve(x_hi, h, mode="valid")
    return y[: (n - 1) * q + 1 : q].copy()


def filter_response(f, fs_hi: float, h) -> np.ndarray:
    """Betragsquadrat der Frequenzantwort |H(f)|^2 des FIR-Filters `h` bei der Rate fs_hi (Eins ohne Filter)."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    if h is None:
        return np.ones(len(f))
    k = np.arange(len(h))
    return np.abs(np.exp(-2j * np.pi * np.outer(f, k) / fs_hi) @ h) ** 2


def sampled_psd(f, fs_hi: float, q: int, h, poles, ar_sigma: float, white_sigma: float) -> np.ndarray:
    """Leistungsdichte der Folge nach (Tiefpass und) Unterabtastung um q: gefaltete Summe |H|^2 S an f + k fs_lo (zweiseitig, Leistung je Hz)."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fs_lo = fs_hi / q
    total = np.zeros(len(f))
    for k in range(-q, q + 1):
        ff = f + k * fs_lo
        inside = (ff > -fs_hi / 2 + 1e-9) & (ff <= fs_hi / 2 + 1e-9)      # halboffen: der Rand zählt einmal
        if not inside.any():
            continue
        s = G.true_psd(np.where(inside, ff, 0.0), fs_hi, poles, ar_sigma, white_sigma) * filter_response(np.where(inside, ff, 0.0), fs_hi, h)
        total += np.where(inside, s, 0.0)
    return total


def sampled_autocov(poles, ar_sigma: float, white_sigma: float, n: int, q: int, h) -> np.ndarray:
    """Autokovarianz r_y(0 .. n-1) des Rauschens nach (Tiefpass und) Unterabtastung (exakt über die Autokovarianz bei hoher Rate)."""
    if h is None:
        r = G.noise_autocov(poles, ar_sigma, white_sigma, (n - 1) * q + 1)
        return r[::q][:n].copy()
    m = len(h)
    r = G.noise_autocov(poles, ar_sigma, white_sigma, (n - 1) * q + m)
    rho = np.correlate(h, h, mode="full")                 # Lags d = -(m-1) .. m-1
    d = np.arange(-(m - 1), m)
    tau = np.arange(n)
    idx = np.abs(tau[:, None] * q + d[None, :])
    return (r[idx] * rho[None, :]).sum(axis=1)


def expected_sampled_periodogram(f, fs_lo: float, n: int, q: int, h, poles, ar_sigma: float, white_sigma: float, clean_y=None) -> np.ndarray:
    """E[I(f)] des Periodogramms der gewonnenen n Werte, exakt: Rauschen über die Autokovarianz, Linien über die tatsächlich gewonnene Folge."""
    r = sampled_autocov(poles, ar_sigma, white_sigma, n, q, h)
    return G.expected_periodogram(f, fs_lo, r, clean_y)


def log_stats(ratio) -> tuple[float, float]:
    """Mittel und Standardabweichung von 10 log10 (I / E[I]) in dB."""
    db = 10.0 * np.log10(np.asarray(ratio, dtype=float))
    return float(db.mean()), float(db.std())


def ops_direct(n: int) -> int:
    """Komplexe Multiplikationen der direkten DFT: n^2."""
    return n * n


def ops_radix2(n: int):
    """Komplexe Multiplikationen einer Radix-2-FFT: (n / 2) log2 n; nur für Zweierpotenzen, sonst None."""
    if n < 2 or n & (n - 1):
        return None
    return n // 2 * int(round(math.log2(n)))


def ops_bluestein(n: int) -> int:
    """Näherung für beliebige Längen (Bluestein): drei Radix-2-FFTs der Länge M >= 2n - 1 plus 2n + M Multiplikationen."""
    m = 1 << (2 * n - 1 - 1).bit_length()
    return 3 * (m // 2) * int(round(math.log2(m))) + 2 * n + m
