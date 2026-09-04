"""Mechanically check the arithmetic claims the paper makes about its own numbers.

    python -m cdfscil.audit_paper

Every check below is computed from `cdfscil/assets/paper/table1_miniimagenet.json`,
which is a verbatim transcription of Table 1.  Nothing here depends on our
reproduction -- it is the paper checked against itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAB = Path(__file__).parent / "assets" / "paper" / "table1_miniimagenet.json"
TAB2 = Path(__file__).parent / "assets" / "paper" / "table2_cub200_ablation.json"

# Claims made in the running text of Sec. 4.1, as (description, quoted value).
TEXT_CLAIMS = [
    ("Sec 4.1: 'improvements of +0.72%, +0.64%, +0.53% in the first three "
     "sessions' vs Tri-WE", "sessions_vs_triwe", [0.72, 0.64, 0.53]),
    ("Sec 4.1: '+1.91% gain on the overall average accuracy' vs Tri-WE",
     "avg_gain_vs_triwe", 1.91),
    ("Sec 4.1: 'at Session 8, CD-FSCIL outperforms FACT by 9.64%'",
     "s8_vs_FACT", 9.64),
    ("Sec 4.1: '... MetaFSCIL by 10.94%'", "s8_vs_MetaFSCIL", 10.94),
    ("Sec 4.1: '... Replay by 12.92%'", "s8_vs_Replay", 12.92),
    ("Sec 4.1: 'while iCaRL drops from 61.31% to 17.20%, CD-FSCIL preserves "
     "60.13% at the last session'", "icarl_last", 17.20),
]


def main():
    d = json.load(open(TAB))
    by = {m["name"]: m for m in d["methods"]}
    ours = by["CD-FSCIL"]["acc"]
    tri = by["Tri-WE"]["acc"]

    fails = 0

    print("=" * 78)
    print("CHECK 1 - does each row's printed Avg equal the mean of that row?")
    print("=" * 78)
    bad = []
    for m in d["methods"]:
        rec = float(np.mean(m["acc"]))
        delta = m["avg_printed"] - rec
        if abs(delta) > 0.05:
            bad.append((m["name"], m["avg_printed"], rec, delta))
    print(f"  rows checked           : {len(d['methods'])}")
    print(f"  rows agreeing (<=0.05) : {len(d['methods']) - len(bad)}")
    for n, p, r, dl in bad:
        print(f"  MISMATCH {n:12s} printed {p:6.2f}   actual mean {r:6.2f}   "
              f"overstated by {dl:+.2f}")
    fails += len(bad)

    print()
    print("=" * 78)
    print("CHECK 2 - arithmetic claims in the running text of Sec. 4.1")
    print("=" * 78)
    computed = {
        "sessions_vs_triwe": [round(ours[i] - tri[i], 2) for i in range(3)],
        "avg_gain_vs_triwe": round(float(np.mean(ours)) - float(np.mean(tri)), 2),
        "s8_vs_FACT": round(ours[8] - by["FACT"]["acc"][8], 2),
        "s8_vs_MetaFSCIL": round(ours[8] - by["MetaFSCIL"]["acc"][8], 2),
        "s8_vs_Replay": round(ours[8] - by["Replay"]["acc"][8], 2),
        "icarl_last": by["iCaRL"]["acc"][8],
    }
    for desc, key, claimed in TEXT_CLAIMS:
        got = computed[key]
        ok = (all(abs(a - b) < 0.02 for a, b in zip(got, claimed))
              if isinstance(claimed, list) else abs(got - claimed) < 0.02)
        print(f"  [{'OK ' if ok else 'BAD'}] {desc}")
        print(f"         claimed {claimed}   computed from Table 1: {got}")
        fails += (not ok)

    print()
    print("=" * 78)
    print("CHECK 3 - is the last-session claim of superiority supported?")
    print("=" * 78)
    print(f"  CD-FSCIL session 8 : {ours[8]:.2f}")
    print(f"  Tri-WE   session 8 : {tri[8]:.2f}")
    print(f"  difference         : {ours[8] - tri[8]:+.2f}   "
          f"({'tie' if ours[8] == tri[8] else 'not a tie'})")

    print()
    print("=" * 78)
    print("CHECK 4 - Table 2 (CUB200 ablation): printed Avg vs the row's own mean")
    print("=" * 78)
    d2 = json.load(open(TAB2))
    for r in d2["rows"]:
        rec = float(np.mean(r["acc"]))
        dl = r["avg_printed"] - rec
        flag = "  <-- MISMATCH" if abs(dl) > 0.05 else ""
        if abs(dl) > 0.05:
            fails += 1
        print(f"  {r['name']:28s} printed {r['avg_printed']:6.2f}   "
              f"actual mean {rec:6.2f}   delta {dl:+.2f}{flag}")
    print(f"  columns in the table: {len(d2['_meta']['columns'])}; "
          f"CUB200 sessions under the paper's own protocol: 11")

    print()
    print("=" * 78)
    print("CHECK 5 - Table 2 prose vs Table 2 numbers")
    print("=" * 78)
    base = next(r for r in d2["rows"] if r["name"] == "CLOSER")["avg_printed"]
    for nm in ("+ Diffusion", "+ LLMs"):
        v = next(r for r in d2["rows"] if r["name"] == nm)["avg_printed"]
        verdict = "IMPROVES" if v > base else "IS LOWER THAN"
        bad = v < base
        fails += bad
        print(f"  [{'BAD' if bad else 'OK '}] Sec 4.4 calls '{nm}' an improvement: "
              f"{v:.2f} {verdict} the CLOSER baseline {base:.2f}")

    print()
    print("=" * 78)
    print(f"SUMMARY: {fails} inconsistency/ies found in the paper's own numbers.")
    print("=" * 78)
    return fails


if __name__ == "__main__":
    main()
