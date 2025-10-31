# src/procedures/pcs_db.py
from __future__ import annotations
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path

DEFAULT_TABLES = Path("data/icd10pcs_tables_2026.xml")
DEFAULT_DEFS   = Path("data/icd10pcs_definitions_2026.xml")

def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    return Path(v) if v else default

PCS_TABLES_PATH = _env_path("PCS_TABLES_PATH", DEFAULT_TABLES)
PCS_DEFS_PATH   = _env_path("PCS_DEFS_PATH",   DEFAULT_DEFS)

@dataclass
class PcsEntry:
    code: str
    title: str
    section: Optional[str] = None   # 1st char
    table: Optional[str] = None     # first 3 chars
    source: str = "tables"          # "tables" | "definitions"

class PcsDB:
    def __init__(self, tables_path: Path = PCS_TABLES_PATH, defs_path: Path = PCS_DEFS_PATH):
        self.entries: List[PcsEntry] = []
        if tables_path.exists():
            self.entries += self._load_from_tables_generate(tables_path)
        else:
            print(f"[pcs_db] WARNING: Tables XML not found at {tables_path}")
        # Definitions don’t carry per-code titles, but we keep parser for future use
        if defs_path.exists():
            self._defs_stats = self._scan_definitions(defs_path)
        else:
            self._defs_stats = {}
            print(f"[pcs_db] WARNING: Definitions XML not found at {defs_path}")

        # fast indices
        self.by_code: Dict[str, PcsEntry] = {e.code: e for e in self.entries}
        self.title_index = [(e.title.lower(), i) for i, e in enumerate(self.entries)]

    # ---------- TABLES → CODES ----------
    def _load_from_tables_generate(self, path: Path) -> List[PcsEntry]:
        """
        Walk every <pcsTable>, read axis labels, and for each <pcsRow> produce the
        cartesian product of axis 4..7 labels. Prepend 1..3 labels to form 7-char code.
        Build a readable long title from Operation/BodyPart/Approach/Device/Qualifier.
        """
        out: List[PcsEntry] = []
        tree = ET.parse(str(path))
        root = tree.getroot()

        # Tag helper
        def lname(el): return el.tag.split('}')[-1]

        for table in root.findall(".//pcsTable"):
            # Collect axis 1..3 (usually single label each)
            axes_123: Dict[int, List[Tuple[str, str]]] = {1: [], 2: [], 3: []}
            for ax in table.findall("./axis"):
                pos = int(ax.attrib.get("pos", "0"))
                if pos not in (1, 2, 3):
                    continue
                for lab in ax.findall("./label"):
                    code = lab.attrib.get("code", "").strip()
                    text = (lab.text or "").strip()
                    if code:
                        axes_123[pos].append((code, text))

            # Skip malformed tables
            if not (axes_123[1] and axes_123[2] and axes_123[3]):
                continue

            # Turn 1..3 into table stem + text map
            secs  = axes_123[1]   # [(code,text)]
            bsys  = axes_123[2]
            opers = axes_123[3]

            # We expect one each, but support lists just in case
            for s_code, s_text in secs:
                for bs_code, bs_text in bsys:
                    for op_code, op_text in opers:
                        table_stem = f"{s_code}{bs_code}{op_code}"
                        # Rows with axis 4..7 combinations
                        for row in table.findall("./pcsRow"):
                            labels: Dict[int, List[Tuple[str, str]]] = {4: [], 5: [], 6: [], 7: []}
                            for ax in row.findall("./axis"):
                                p = int(ax.attrib.get("pos", "0"))
                                if p not in (4, 5, 6, 7):
                                    continue
                                for lab in ax.findall("./label"):
                                    code = lab.attrib.get("code", "").strip()
                                    text = (lab.text or "").strip()
                                    if code:
                                        labels[p].append((code, text))

                            # Some rows don’t have all axes, skip incomplete rows
                            if not (labels[4] and labels[5] and labels[6] and labels[7]):
                                continue

                            # Cartesian product across 4..7
                            for bp_code, bp_text in labels[4]:
                                for ap_code, ap_text in labels[5]:
                                    for dv_code, dv_text in labels[6]:
                                        for q_code, q_text in labels[7]:
                                            code = f"{table_stem}{bp_code}{ap_code}{dv_code}{q_code}"
                                            # Compose a readable long title
                                            title = self._compose_title(op_text, bs_text, bp_text, ap_text, dv_text, q_text)
                                            out.append(PcsEntry(
                                                code=code,
                                                title=title,
                                                section=s_code,
                                                table=table_stem,
                                                source="tables"
                                            ))
        return out

    def _compose_title(self, operation: str, body_system: str, body_part: str,
                       approach: str, device: str, qualifier: str) -> str:
        """
        Build a deterministic long title similar to PCS book phrasing.
        Examples:
          'Measurement of Cardiac Electrical Activity, External Approach'
          'Insertion of Monitoring Device into Brain, Percutaneous Approach'
        """
        # Base “<Operation> of <Body Part>” (include body system in parens when helpful)
        base = f"{operation} of {body_part}"
        # Add approach
        title = f"{base}, {approach} Approach"
        # Device (omit if “No Device”)
        if device and device.lower() not in {"no device"}:
            # Some ops (e.g., Insertion) read better as “Insertion of <Device> into <Body Part>”
            if operation.lower() in {"insertion", "replacement", "supplement", "removal", "revision", "change"}:
                title = f"{operation} of {device} into {body_part}, {approach} Approach"
            else:
                title = f"{base} with {device}, {approach} Approach"
        # Qualifier (omit if “No Qualifier”)
        if qualifier and qualifier.lower() not in {"no qualifier"}:
            title = f"{title} ({qualifier})"
        return title

    # ---------- DEFINITIONS (for reference only) ----------
    def _scan_definitions(self, path: Path) -> Dict[str, Dict[str, str]]:
        # keep operation definitions handy if you want to enrich titles later
        out: Dict[str, Dict[str, str]] = {}
        try:
            tree = ET.parse(str(path))
            root = tree.getroot()
            for terms in root.findall(".//axis[@pos='3']/terms"):
                t = (terms.findtext("./title") or "").strip()
                d = (terms.findtext("./definition") or "").strip()
                if t:
                    out[t] = {"definition": d}
        except Exception as e:
            print(f"[pcs_db] definitions scan failed: {e}")
        return out

    # ---------- Search API ----------
    def search_title(self, query: str, must_contain: Optional[List[str]] = None) -> List[PcsEntry]:
        q = normalize(query)
        toks = [t for t in re.split(r"\W+", q) if t]
        res: List[PcsEntry] = []
        for t, idx in self.title_index:
            ok = all(tok in t for tok in toks)
            if ok and must_contain:
                ok = all(mc in t for mc in must_contain)
            if ok:
                res.append(self.entries[idx])
        # unique by code
        seen, final = set(), []
        for e in res:
            if e.code not in seen:
                final.append(e); seen.add(e.code)
        return final

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

# Singleton
PCS_DB: Optional[PcsDB] = None
def get_pcs_db() -> PcsDB:
    global PCS_DB
    if PCS_DB is None:
        PCS_DB = PcsDB()
    return PCS_DB
