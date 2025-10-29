import xmlschema
import pandas as pd
from typing import List, Dict, Any
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def extract_field(node, key: str) -> Any:
    """
    Helper: extract a child value or list of values from a node dict.
    If node is None or missing key, returns None or empty list depending on context.
    """
    if not node:
        return None
    val = node.get(key)
    return val

def parse_index_xml(xsd_path: str, xml_path: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading Index schema from: {xsd_path}")
    schema = xmlschema.XMLSchema(xsd_path)
    logging.info(f"Parsing Index XML file: {xml_path}")
    data_dict = schema.to_dict(xml_path)
    
    records: List[Dict[str, Any]] = []
    
    root = data_dict.get("ICD10CM.index", data_dict)
    letters = root.get("letter", [])
    if not isinstance(letters, list):
        letters = [letters]
    
    for letter_node in letters:
        letter_title = extract_field(letter_node, "title")
        logging.debug(f"Processing Letter: {letter_title}")
        
        main_terms = letter_node.get("mainTerm", [])
        if not isinstance(main_terms, list):
            main_terms = [main_terms]
        
        for main in main_terms:
            main_title = extract_field(main, "title")
            main_code = extract_field(main, "code")
            main_see = extract_field(main, "see")
            main_seeAlso = extract_field(main, "seeAlso")
            
            # Records for the main term itself
            rec_main = {
                "letter": letter_title,
                "term_level": 0,
                "title": main_title,
                "code": main_code,
                "see": main_see,
                "seeAlso": main_seeAlso,
                "parent_term": None
            }
            records.append(rec_main)
            
            # Process nested “term” nodes under the mainTerm
            nested_terms = main.get("term", [])
            if not nested_terms:
                continue
            if not isinstance(nested_terms, list):
                nested_terms = [nested_terms]
            
            for term_node in nested_terms:
                # We can recursively process deeper levels if needed
                def process_term(node, parent_title, parent_code, level):
                    title = extract_field(node, "title")
                    code = extract_field(node, "code")
                    see = extract_field(node, "see")
                    seeAlso = extract_field(node, "seeAlso")
                    
                    rec = {
                        "letter": letter_title,
                        "term_level": level,
                        "title": title,
                        "code": code,
                        "see": see,
                        "seeAlso": seeAlso,
                        "parent_term": parent_title,
                        "parent_code": parent_code
                    }
                    records.append(rec)
                    
                    # if further nested term elements
                    subs = node.get("term", [])
                    if subs:
                        if not isinstance(subs, list):
                            subs = [subs]
                        for s in subs:
                            process_term(s, title, code, level+1)
                
                process_term(term_node, main_title, main_code, 1)
    
    logging.info(f"Index extraction complete – total records: {len(records)}")
    return records

def main():
    xsd_file = r"C:\Users\Utilizador\Desktop\ICD10\icd10cm-index-April-2024.xsd"
    xml_file = r"C:\Users\Utilizador\Desktop\ICD10\icd10cm-index-April-2024.xml"
    
    if not os.path.exists(xsd_file):
        logging.error(f"Index XSD file not found: {xsd_file}")
        return
    if not os.path.exists(xml_file):
        logging.error(f"Index XML file not found: {xml_file}")
        return
    
    records = parse_index_xml(xsd_file, xml_file)
    df = pd.DataFrame(records)
    logging.info(f"DataFrame created with {len(df)} rows")
    
    print(f"📊 Total index entries extracted: {len(df)}\n")
    print("✅ Sample entries:")
    sample = df.head(10).to_dict(orient="records")
    for i, rec in enumerate(sample, start=1):
        print(f"\n{i}. Title: {rec['title']}")
        print(f"   Code: {rec.get('code')}")
        print(f"   See: {rec.get('see')}")
        print(f"   SeeAlso: {rec.get('seeAlso')}")
        print(f"   Level: {rec['term_level']} | Letter: {rec['letter']}")
        print(f"   Parent Term: {rec.get('parent_term')} | Parent Code: {rec.get('parent_code')}")
    
    out_csv = "icd10_index_extracted.csv"
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved extracted index records to: {out_csv}")

if __name__ == "__main__":
    main()
