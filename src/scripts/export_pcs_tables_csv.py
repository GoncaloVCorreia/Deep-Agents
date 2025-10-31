# src/scripts/export_pcs_tables_csv.py
import csv
from pathlib import Path
from src.procedures.pcs_db import get_pcs_db

OUT = Path("icd10pcs_generated_2026.csv")

if __name__ == "__main__":
    db = get_pcs_db()
    rows = []
    for e in db.entries:
        rows.append({
            "code": e.code,
            "title": e.title,
            "section": e.section,
            "table": e.table,
            "source": e.source
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows):,} procedure codes to {OUT}")
