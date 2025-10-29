import xmlschema
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def extract_notes(field_node) -> List[str]:
    """
    Helper to extract list of note strings from a field node.
    The field_node may be None, a dict, or a list of dicts.
    """
    if not field_node:
        return []
    notes_list: List[str] = []
    if isinstance(field_node, list):
        for fn in field_node:
            if isinstance(fn, dict):
                note_val = fn.get("note")
                if note_val:
                    if isinstance(note_val, list):
                        notes_list.extend(note_val)
                    else:
                        notes_list.append(note_val)
    elif isinstance(field_node, dict):
        note_val = field_node.get("note")
        if note_val:
            if isinstance(note_val, list):
                notes_list.extend(note_val)
            else:
                notes_list.append(note_val)
    # Remove duplicates, strip whitespace
    notes_list = [n.strip() for n in notes_list if isinstance(n, str) and n.strip()]
    return list(dict.fromkeys(notes_list))

def parse_tabular_xml(xsd_path: str, xml_path: str) -> List[Dict[str, Any]]:
    """
    Load the XSD schema, parse the XML, extract code entries into records.
    """
    logging.info(f"Loading schema from: {xsd_path}")
    schema = xmlschema.XMLSchema(xsd_path)
    logging.info(f"Parsing XML file: {xml_path}")
    # This returns nested dictionaries per the schema
    data_dict = schema.to_dict(xml_path)

    records: List[Dict[str, Any]] = []

    # Root key is likely "ICD10CM.tabular"
    tabular_root = data_dict.get("ICD10CM.tabular", data_dict)

    chapters = tabular_root.get("chapter", [])
    if not isinstance(chapters, list):
        chapters = [chapters]

    for chap in chapters:
        chap_name = chap.get("name")
        chap_desc = chap.get("desc")
        logging.debug(f"Processing Chapter: {chap_name} – {chap_desc}")

        sections = chap.get("section", [])
        if not isinstance(sections, list):
            sections = [sections]

        for sect in sections:
            sect_id = sect.get("id")
            sect_desc = sect.get("desc")
            logging.debug(f"  Section: {sect_id} – {sect_desc}")

            def process_diag(diag_node, parent_codes: List[str]):
                name = diag_node.get("name")
                desc = diag_node.get("desc")

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
                    # if you want parent_codes, you can store that too
                    "parent_codes": parent_codes.copy()
                }

                records.append(rec)

                nested = diag_node.get("diag")
                if nested:
                    if isinstance(nested, list):
                        for nd in nested:
                            process_diag(nd, parent_codes + [name])
                    else:
                        process_diag(nested, parent_codes + [name])

            diags = sect.get("diag", [])
            if not isinstance(diags, list):
                diags = [diags]

            for d in diags:
                process_diag(d, [])

    logging.info(f"Extraction complete – total records: {len(records)}")
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

    records = parse_tabular_xml(xsd_file, xml_file)

    df = pd.DataFrame(records)
    logging.info(f"DataFrame created with {len(df)} rows")

    # Print number of records and first few entries
    print(f"📊 Total diagnosis code entries extracted: {len(df)}\n")
    print("✅ Sample entries:")
    sample = df.head(10).to_dict(orient="records")
    for i, rec in enumerate(sample, start=1):
        print(f"\n{i}. Code: {rec['code']}")
        print("   Description: ", rec['description'])
        print("   Chapter:     ", rec['chapter'], "-", rec['chapter_desc'])
        print("   Section:     ", rec.get('section'), "-", rec.get('section_desc'))
        print("   Inclusion Terms:      ", rec['inclusion_terms'])
        print("   Includes:             ", rec['includes'])
        print("   Excludes (1):         ", rec['excludes1'])
        print("   Excludes (2):         ", rec['excludes2'])
        print("   Use Additional Code:  ", rec['use_additional_code'])
        print("   Code First:           ", rec['code_first'])
        print("   Notes:                ", rec['notes'])
        # optionally print parent_codes
        print("   Parent Codes:         ", rec['parent_codes'])

    # Save to CSV for inspection or further processing
    out_csv = "icd10_tabular_extracted.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved extracted records to: {out_csv}")

if __name__ == "__main__":
    main()
