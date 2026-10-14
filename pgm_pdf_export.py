"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from pgm_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Das Periodogramm: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"n = {s.n} Werte, Abtastung: {s.sampling} (fs = {de(a.fs, 0)} Hz), Linie bei {de(s.line, 1)} Hz mit {s.snr} dB je Abtastwert, "
                            f"Hintergrund: {s.background}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Scheinbare Frequenz der Linie: {de(a.apparent, 1)} Hz; höchster Wert des Periodogramms bei {de(a.peak_f, 1)} Hz "
        f"({'die Linie' if a.line_is_max else 'nicht die Linie'}).",
        f"Streuung der Werte um den exakten Erwartungswert (Std/Mittel): {de(a.cv, 2)}; für eine Exponentialverteilung gilt 1,00.",
        f"Bins ohne Linie, die über dem Linienwert liegen: {a.higher} von {len(a.f) - 1}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Periodogramm: I(f) = |X(f)|^2 / (n fs), Rechteckfenster. Quelle: periodogramm-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
