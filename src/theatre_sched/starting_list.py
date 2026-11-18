"""Describing today's waiting list.

Three ways, from least to most detail:

* **Totals** (``Scenario.start_waiting`` with ``start_over_target_pct`` and ``start_longest_weeks``):
  shared evenly between the surgeons, waits drawn by ``rolling.starting_waits``.
* **By surgeon** (``Scenario.start_by_surgeon``): the same three numbers for each surgeon.
* **Counts by weeks waited** (``Scenario.start_bands``), as in a waiting-list extract: how many
  patients have waited 0-6 weeks, 6-12 weeks and so on, optionally for each surgeon. Waits are
  spread evenly within each band. ``parse_counts`` reads them from pasted text or a CSV file.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

OPEN_BAND_WEEKS = 26.0  # how far an open-ended band such as "52+" is taken to run


@dataclass(frozen=True)
class Band:
    surgeon: str  # "" when the extract has no surgeon column
    lo_weeks: float  # waited at least this long...
    hi_weeks: float  # ...and less than this
    patients: int


class CountsError(ValueError):
    pass


_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(?:w|wk|wks|weeks?)?\s*$", re.I)
_OPEN = re.compile(r"^\s*(?:>|over\s+|more than\s+)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:w|wk|wks|weeks?)?\s*(\+)?\s*$", re.I)
_UNDER = re.compile(r"^\s*(?:<|under\s+|less than\s+)\s*(\d+(?:\.\d+)?)\s*(?:w|wk|wks|weeks?)?\s*$", re.I)


def parse_weeks(text: str) -> tuple[float, float]:
    """'12-18' -> (12, 18); '52+' or '>52' -> (52, 52 + OPEN_BAND_WEEKS); '<6' -> (0, 6); '7' -> (7, 8)."""
    t = str(text).strip()
    if m := _RANGE.match(t):
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi <= lo:
            raise CountsError(f"'{t}': the end of a band must be after its start")
        return lo, hi
    if m := _UNDER.match(t):
        return 0.0, float(m.group(1))
    if (m := _OPEN.match(t)):
        lo = float(m.group(1))
        open_ended = "+" in t or t.lstrip().startswith(">") or t.lower().lstrip().startswith(("over", "more"))
        return (lo, lo + OPEN_BAND_WEEKS) if open_ended else (lo, lo + 1)
    raise CountsError(f"'{t}' isn't a number of weeks or a band like 12-18 or 52+")


def parse_counts(text: str) -> list[Band]:
    """Read counts of patients by weeks waited.

    Each line is ``weeks, patients`` or ``surgeon, weeks, patients``, separated by commas, tabs or
    semicolons (so a block copied from a spreadsheet works). A header line is optional. Weeks can
    be a whole number of weeks ("7" means 7 to 8 weeks), a band ("12-18", meaning at least 12 and
    under 18) or open-ended ("52+", taken as 52 to 78 weeks).
    """
    text = (text or "").strip()
    if not text:
        raise CountsError("Nothing to read: paste one line per band, e.g. 0-6, 130")
    dialect = "excel-tab" if "\t" in text else "excel"
    lines = text.replace(";", ",").splitlines() if dialect == "excel" else text.splitlines()
    bands: list[Band] = []
    width = None
    for n, row in enumerate(csv.reader(io.StringIO("\n".join(lines)), dialect=dialect), start=1):
        cells = [c.strip() for c in row if c.strip() != ""]
        if not cells:
            continue
        if len(cells) not in (2, 3):
            raise CountsError(f"Line {n}: expected 'weeks, patients' or 'surgeon, weeks, patients', got {len(cells)} values")
        try:
            patients = int(float(cells[-1].replace(",", "")))
        except ValueError:
            if not bands and width is None:  # a header line
                width = len(cells)
                continue
            raise CountsError(f"Line {n}: '{cells[-1]}' isn't a number of patients") from None
        if width is not None and len(cells) != width:
            raise CountsError(f"Line {n}: expected {width} values like the header")
        width = len(cells)
        if patients < 0:
            raise CountsError(f"Line {n}: negative number of patients")
        try:
            lo, hi = parse_weeks(cells[-2])
        except CountsError as e:
            raise CountsError(f"Line {n}: {e}") from None
        bands.append(Band(cells[0] if len(cells) == 3 else "", lo, hi, patients))
    if not bands:
        raise CountsError("No counts found")
    if sum(b.patients for b in bands) == 0:
        raise CountsError("The counts add up to no patients")
    return bands


def surgeon_names(bands: list[Band]) -> list[str]:
    """Distinct surgeon names in the order they first appear ([] if the extract has none)."""
    return list(dict.fromkeys(b.surgeon for b in bands if b.surgeon))


def summarise(bands: list[Band], target_weeks: float) -> dict:
    """Total, share over target and longest wait, assuming waits are spread evenly within each band."""
    total = sum(b.patients for b in bands)
    over = 0.0
    for b in bands:
        if b.lo_weeks >= target_weeks:
            over += b.patients
        elif b.hi_weeks > target_weeks:
            over += b.patients * (b.hi_weeks - target_weeks) / (b.hi_weeks - b.lo_weeks)
    longest = max(b.hi_weeks for b in bands if b.patients > 0)
    return {"patients": total, "over_target_pct": 100 * over / total if total else 0.0, "longest_weeks": longest}


def to_scenario_bands(bands: list[Band], n_surgeons: int) -> tuple[tuple[int, float, float, int], ...]:
    """Bands as (surgeon index, lo weeks, hi weeks, patients), for ``Scenario.start_bands``. Surgeon names
    map, in order of first appearance, to the simulation's surgeons S1, S2, ...; -1 means "share between
    all surgeons"."""
    names = surgeon_names(bands)
    if len(names) > n_surgeons:
        raise CountsError(f"The extract has {len(names)} surgeons but the scenario has {n_surgeons}: "
                          f"set Surgeons to at least {len(names)}")
    index = {name: i for i, name in enumerate(names)}
    return tuple((index.get(b.surgeon, -1), b.lo_weeks, b.hi_weeks, int(b.patients)) for b in bands)
