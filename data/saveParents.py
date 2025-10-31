import xmlschema
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]

def extract_notes(field_node) -> List[str]:
    """
    Extract list of note strings from a node shaped per the XSD (noteType/contentType).
    Handles None | dict | list[dict] and returns unique, stripped strings.
    """
    if not field_node:
        return []
    notes_list: List[str] = []

    if isinstance(field_node, list):
        for item in field_node:
            if isinstance(item, dict):
                note_val = item.get("note")
                if note_val:
                    if isinstance(note_val, list):
                        for n in note_val:
                            if isinstance(n, str):
                                notes_list.append(n)
                    elif isinstance(note_val, str):
                        notes_list.append(note_val)
    elif isinstance(field_node, dict):
        note_val = field_node.get("note")
        if note_val:
            if isinstance(note_val, list):
                for n in note_val:
                    if isinstance(n, str):
                        notes_list.append(n)
            elif isinstance(note_val, str):
                notes_list.append(note_val)

    # de-dup, strip
    notes_list = [n.strip() for n in notes_list if isinstance(n, str) and n.strip()]
    return list(dict.fromkeys(notes_list))

def _get_text_from_content(node: Any) -> str:
    """
    xmlschema.to_dict() typically places element text under '$'
    and attributes under '@attr'. Be defensive.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        # prefer '$' for text content
        if "$" in node and isinstance(node["$"], str):
            return node["$"].strip()
        # sometimes content might be directly in 'value' or 'text'
        for key in ("value", "text"):
            if isinstance(node.get(key), str):
                return node[key].strip()
        # if it's a contentType-like dict with 'new'/'old'/'unc'
        # join any present parts as a fallback
        possible_text_keys = [k for k in ("new", "old", "unc") if k in node]
        if possible_text_keys:
            parts = []
            for k in possible_text_keys:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            return " ".join(parts).strip()
    return ""

def extract_seven_char_info(diag_node: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Returns:
      seven_chr_note_texts: list[str] from <sevenChrNote><note>...</note>
      seven_chr_extensions: dict { char: label_text } from <sevenChrDef><extension char="X">label</extension>
      seven_chr_def_notes:  list[str] for any <note> inside <sevenChrDef>
    """
    # 1) Seven character NOTE(s)
    seven_chr_note_texts = extract_notes(diag_node.get("sevenChrNote"))

    # 2) Seven character DEF (extensions + optional notes)
    seven_chr_extensions: Dict[str, str] = {}
    seven_chr_def_notes_all: List[str] = []

    seven_def_node = diag_node.get("sevenChrDef")

    # Handle dict OR list of dicts
    for seven_def in _as_list(seven_def_node):
        if not isinstance(seven_def, dict):
            # If xmlschema ever gives a raw string here (unlikely), skip
            continue

        # extensions can be a single dict or a list
        ext_nodes = seven_def.get("extension")
        for ext in _as_list(ext_nodes):
            if isinstance(ext, dict):
                ch = ext.get("@char") or ext.get("char")
                label = _get_text_from_content(ext)  # pulls "$" text or fallbacks
                if ch and isinstance(ch, str):
                    seven_chr_extensions[ch] = label

        # notes inside sevenChrDef
        seven_chr_def_notes_all.extend(extract_notes(seven_def))

    # de-dup notes
    seven_chr_def_notes = list(dict.fromkeys([n for n in seven_chr_def_notes_all if n]))

    return seven_chr_note_texts, seven_chr_extensions, seven_chr_def_notes


def parse_tabular_xml_top_level_only(xsd_path: str, xml_path: str) -> List[Dict[str, Any]]:
    """
    Parse and return ONLY top-level diagnosis entries (no parent_codes).
    Also captures sevenChrNote and sevenChrDef info.
    """
    logging.info(f"Loading schema from: {xsd_path}")
    schema = xmlschema.XMLSchema(xsd_path)

    logging.info(f"Parsing XML file: {xml_path}")
    data_dict = schema.to_dict(xml_path)

    records: List[Dict[str, Any]] = []

    tabular_root = data_dict.get("ICD10CM.tabular", data_dict)
    chapters = tabular_root.get("chapter", [])
    if not isinstance(chapters, list):
        chapters = [chapters]

    for chap in chapters:
        chap_name = chap.get("name")
        chap_desc = chap.get("desc")

        sections = chap.get("section", [])
        if not isinstance(sections, list):
            sections = [sections]

        for sect in sections:
            sect_id = sect.get("id")
            sect_desc = sect.get("desc")

            def process_diag(diag_node: Dict[str, Any], parent_codes: List[str]):
                name = diag_node.get("name")
                desc = diag_node.get("desc")

                seven_chr_note_texts, seven_chr_extensions, seven_chr_def_notes = extract_seven_char_info(diag_node)

                rec: Dict[str, Any] = {
                    "code": name,
                    "description": desc,
                    "chapter": chap_name,
                    "chapter_desc": chap_desc,
                    "section": sect_id,
                    "section_desc": sect_desc,
                    "inclusion_terms": extract_notes(diag_node.get("inclusionTerm")),
                    "includes": extract_notes(diag_node.get("includes")),
                    "excludes1": extract_notes(diag_node.get("excludes1")),
                    "excludes2": extract_notes(diag_node.get("excludes2")),
                    "use_additional_code": extract_notes(diag_node.get("useAdditionalCode")),
                    "code_first": extract_notes(diag_node.get("codeFirst")),
                    "notes": extract_notes(diag_node.get("notes")),
                    "seven_char_note": seven_chr_note_texts,            # NEW
                    "seven_char_def_extensions": seven_chr_extensions,   # NEW (dict like {"A": "initial encounter", ...})
                    "seven_char_def_notes": seven_chr_def_notes,         # NEW (notes within sevenChrDef, if any)
                    "parent_codes": parent_codes.copy()
                }

                # Only keep if it has NO parent codes
                if not parent_codes:
                    records.append(rec)

                # Recurse for children (but we do NOT add them to records)
                nested = diag_node.get("diag")
                if nested:
                    for nd in _as_list(nested):
                        process_diag(nd, parent_codes + [name])

            diags = sect.get("diag", [])
            if not isinstance(diags, list):
                diags = [diags]

            for d in diags:
                process_diag(d, [])

    logging.info(f"Top-level extraction complete – total records kept: {len(records)}")
    return records

def main():
    # Update paths as needed
    xsd_file = r"C:\Users\Utilizador\Desktop\ICD10\icd10cm-tabular-April-2024.xsd"
    xml_file = r"C:\Users\Utilizador\Desktop\ICD10\icd10cm-tabular-April-2024.xml"

    if not os.path.exists(xsd_file):
        logging.error(f"Schema file not found: {xsd_file}")
        return
    if not os.path.exists(xml_file):
        logging.error(f"XML file not found: {xml_file}")
        return

    records = parse_tabular_xml_top_level_only(xsd_file, xml_file)
    df = pd.DataFrame(records)
    logging.info(f"DataFrame created with {len(df)} top-level rows")

    print(f"📊 Total TOP-LEVEL diagnosis code entries extracted: {len(df)}\n")
    if not df.empty:
        sample = df.head(10).to_dict(orient="records")
        print("✅ Sample entries:")
        for i, rec in enumerate(sample, start=1):
            print(f"\n{i}. Code: {rec['code']}")
            print("   Description:           ", rec['description'])
            print("   Chapter:               ", rec['chapter'], "-", rec['chapter_desc'])
            print("   Section:               ", rec.get('section'), "-", rec.get('section_desc'))
            print("   Inclusion Terms:       ", rec['inclusion_terms'])
            print("   Includes:              ", rec['includes'])
            print("   Excludes (1):          ", rec['excludes1'])
            print("   Excludes (2):          ", rec['excludes2'])
            print("   Use Additional Code:   ", rec['use_additional_code'])
            print("   Code First:            ", rec['code_first'])
            print("   Notes:                 ", rec['notes'])
            print("   7th Char NOTE:         ", rec['seven_char_note'])
            print("   7th Char DEF (map):    ", rec['seven_char_def_extensions'])
            print("   7th Char DEF notes:    ", rec['seven_char_def_notes'])
            print("   Parent Codes:          ", rec['parent_codes'])

    out_csv = "icd10_tabular_top_level_only.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved extracted TOP-LEVEL records to: {out_csv}")

if __name__ == "__main__":
    main()
