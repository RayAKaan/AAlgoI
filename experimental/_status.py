# aalgoi/_status.py

"""
Beautiful terminal output — box, table, progress bar.

Windows-safe: falls back to ASCII when Unicode encoding fails.
"""

from __future__ import annotations

import sys

# ── Detect Unicode support ───────────────────────────────────────────

def _supports_unicode() -> bool:
    """Check if stdout can handle Unicode box-drawing characters."""
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        # Test the characters we actually use
        test = "╔╗╚╝║═┌┐└┘│─┬┴┼├┤█░"
        test.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


# Character sets — Unicode and ASCII fallback
if _supports_unicode():
    _BOX_TL = "╔"
    _BOX_TR = "╗"
    _BOX_BL = "╚"
    _BOX_BR = "╝"
    _BOX_H  = "═"
    _BOX_V  = "║"
    _BOX_T  = "╤"
    _BOX_B  = "╧"
    _BOX_X  = "╪"
    _BOX_L  = "╟"
    _BOX_R  = "╢"
    _BOX_MI = "┼"
    _BOX_ML = "├"
    _BOX_MR = "┤"
    _BOX_MH = "─"
    _BAR_FILL = "█"
    _BAR_EMPTY = "░"
else:
    _BOX_TL = "+"
    _BOX_TR = "+"
    _BOX_BL = "+"
    _BOX_BR = "+"
    _BOX_H  = "-"
    _BOX_V  = "|"
    _BOX_T  = "+"
    _BOX_B  = "+"
    _BOX_X  = "+"
    _BOX_L  = "+"
    _BOX_R  = "+"
    _BOX_MI = "+"
    _BOX_ML = "+"
    _BOX_MR = "+"
    _BOX_MH = "-"
    _BAR_FILL = "#"
    _BAR_EMPTY = "-"


# ── Box ───────────────────────────────────────────────────────────────

_MAX_DISPLAY = 50

def box(lines: list[str], title: str | None = None) -> str:
    """
    Draw a box around lines of text.

    >>> box(["hello", "world"], title="Test")
    ╔══════════╗
    ║ Test      ║
    ╟──────────╢
    ║ hello     ║
    ║ world     ║
    ╚══════════╝
    """
    if not lines:
        return ""

    # Truncate if too many lines
    display = lines[:_MAX_DISPLAY]
    truncated = len(lines) > _MAX_DISPLAY

    all_lines = list(display)
    if title:
        all_lines = [title, _BOX_H * 3] + all_lines
    if truncated:
        all_lines.append(f"... ({len(lines) - _MAX_DISPLAY} more)")

    max_len = max(len(_strip_ansi(line)) for line in all_lines)
    width = max_len + 2  # 1 padding each side

    top    = _BOX_TL + _BOX_H * width + _BOX_TR
    bottom = _BOX_BL + _BOX_H * width + _BOX_BR
    mid    = [_BOX_V + " " + line.ljust(max_len) + " " + _BOX_V for line in all_lines]

    return "\n".join([top] + mid + [bottom])


# ── Table ────────────────────────────────────────────────────────────

def table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Draw an ASCII table.

    >>> table(["A", "B"], [["1", "2"], ["3", "4"]])
    ┌───┬───┐
    │ A │ B │
    ├───┼───┤
    │ 1 │ 2 │
    │ 3 │ 4 │
    └───┴───┘
    """
    if not headers:
        return ""

    n_cols = len(headers)
    col_widths = [
        max(
            len(_strip_ansi(headers[i])),
            max((len(_strip_ansi(row[i])) for row in rows if i < len(row)), default=0),
        )
        for i in range(n_cols)
    ]

    def sep(left: str, mid: str, right: str, fill: str) -> str:
        return left + mid.join(fill * (w + 2) for w in col_widths) + right

    def row(vals: list[str]) -> str:
        cells = []
        for i, val in enumerate(vals):
            if i < n_cols:
                cells.append(_BOX_V + " " + val.ljust(col_widths[i]) + " " + _BOX_V)
            else:
                cells.append(_BOX_V + " " + val + " " + _BOX_V)
        return "".join(cells[:n_cols]) + ("".join(cells[n_cols:]) if len(cells) > n_cols else "")

    top    = sep(_BOX_TL, _BOX_MI, _BOX_TR, _BOX_H)
    header = row(headers)
    divider = sep(_BOX_ML, _BOX_MI, _BOX_MR, _BOX_H)
    bottom = sep(_BOX_BL, _BOX_MI, _BOX_BR, _BOX_H)

    lines = [top, header, divider]
    for r in rows:
        lines.append(row(r))
    lines.append(bottom)

    return "\n".join(lines)


# ── Progress Bar ─────────────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 20, label: str = "") -> str:
    """
    Draw a progress bar.

    >>> progress_bar(75, 100, label="Training")
    Training |███████████████████░░░░░| 75%
    """
    if total == 0:
        pct = 100
    else:
        pct = round(current / total * 100)

    filled = int(width * pct / 100)
    empty = width - filled

    bar = _BAR_FILL * filled + _BAR_EMPTY * empty

    if label:
        return f"{label} |{bar}| {pct}%"
    return f"|{bar}| {pct}%"


# ── Helpers ──────────────────────────────────────────────────────────

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for width calculation."""
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', text)
