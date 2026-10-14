"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import pgm_constants as C
from pgm_evaluation import Analysis

EPS = 1e-30


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, dtype=float), EPS))


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_signal(a: Analysis) -> go.Figure:
    """Gewonnene Folge über der Zeit in Millisekunden."""
    t_ms = np.arange(len(a.x)) / a.fs * 1000.0
    fig = go.Figure(go.Scatter(x=t_ms, y=a.x, mode="lines", line=dict(color=C.COLOR_I, width=1), hovertemplate="%{x:.1f} ms: %{y:.2f}<extra></extra>", name="Messwerte"))
    fig.update_xaxes(title_text="Zeit (ms)")
    fig.update_yaxes(title_text="Wert")
    return _fixed(fig, height=240, showlegend=False)


def build_periodogram(a: Analysis) -> go.Figure:
    """Periodogramm in dB gegen Frequenz mit exaktem Erwartungswert, wahrer Dichte (vor der Abtastung) und der Dichte der gewonnenen Folge."""
    f = a.f
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=_db(a.I), mode="lines", line=dict(color=C.COLOR_I, width=1.2), name="Periodogramm",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=_db(a.expected), mode="lines", line=dict(color=C.COLOR_EXPECTED, width=2.5), name="Erwartungswert (exakt)",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=f, y=_db(a.true_density), mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dot"), name="wahre Dichte des Hintergrunds",
                             hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    if a.settings.q > 1:
        fig.add_trace(go.Scatter(x=f, y=_db(a.sampled_density), mode="lines", line=dict(color="#7B5EA7", width=2, dash="dash"), name="Dichte nach der Abtastung",
                                 hovertemplate="%{x:.1f} Hz: %{y:.1f} dB<extra></extra>"))
    fig.add_vline(x=a.apparent, line=dict(color=C.COLOR_LINE, width=1.5, dash="dash"))
    fig.update_xaxes(title_text="Frequenz (Hz)", range=[0, a.fs / 2])
    fig.update_yaxes(title_text="Leistungsdichte (dB re 1 Leistung/Hz)")
    return _fixed(fig, height=420)


def build_variation(rows) -> go.Figure:
    xs = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["cv"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_I, width=3), name="Std / Mittel je Bin",
                             hovertemplate="n = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot"), annotation_text="Exponentialverteilung: 1,00")
    fig.update_xaxes(title_text="Länge n", type="log", tickvals=xs)
    fig.update_yaxes(title_text="Std / Mittel von I / S", range=[0, 1.4])
    return _fixed(fig, height=320)


def build_line(rows) -> go.Figure:
    xs = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["over_floor_db"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name="Linie über dem Untergrund (dB)",
                             hovertemplate="n = %{x}: %{y:.1f} dB<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["p_max"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_I, width=3, dash="dash"), name="Linie ist der höchste Wert (%)",
                             yaxis="y2", hovertemplate="n = %{x}: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Länge n", type="log", tickvals=xs)
    fig.update_yaxes(title_text="dB über dem Untergrund")
    fig.update_layout(yaxis2=dict(title="Anteil der Läufe (%)", overlaying="y", side="right", range=[0, 105], fixedrange=True))
    return _fixed(fig, height=340)


def build_bias(rows) -> go.Figure:
    fig = go.Figure()
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["frac"] for r in sel], y=[r["bias_db"] for r in sel], mode="lines+markers", name=f"n = {n}",
                                 hovertemplate="f = %{x:.2f} fs: %{y:.1f} dB<extra></extra>"))
    fig.update_xaxes(title_text="Frequenz (Bruchteil der Abtastrate)")
    fig.update_yaxes(title_text="Mittelwert des Periodogramms über wahrer Dichte (dB)")
    return _fixed(fig, height=340)


def build_sampling(rows) -> go.Figure:
    labels = [f"jeder {r['q']}." for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["true_db"] for r in rows], name="wahre Dichte", marker_color=C.COLOR_TRUE))
    fig.add_trace(go.Bar(x=labels, y=[r["raw_db"] for r in rows], name="gemessen ohne Tiefpass", marker_color=C.COLOR_LINE))
    fig.add_trace(go.Bar(x=labels, y=[r["filtered_db"] for r in rows], name="gemessen mit Tiefpass", marker_color=C.COLOR_I))
    fig.update_yaxes(title_text="Mittlere Dichte im oberen Band (dB)")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)
