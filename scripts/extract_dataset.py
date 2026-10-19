"""
Extract dataset images and annotations from Pothole_detection_dataset.xlsx
Maps modern Excel in-cell RichValue images to rows and metadata.
"""

import os
import zipfile
import csv
import xml.etree.ElementTree as ET

def extract_dataset(xlsx_path="Pothole_detection_dataset.xlsx", output_dir="dataset"):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    with zipfile.ZipFile(xlsx_path, "r") as z:
        # 1. Parse shared strings
        sst_xml = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(t.text or "" for t in elem.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
            for elem in sst_xml.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si")
        ]
        
        # 2. Parse richValue relationships: rId -> image path
        rv_rels_xml = ET.fromstring(z.read("xl/richData/_rels/richValueRel.xml.rels"))
        rid_to_target = {}
        for rel in rv_rels_xml.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
            rid = rel.attrib["Id"]
            target = rel.attrib["Target"]  # e.g. ../media/image13.jpeg
            clean_target = "xl/" + target.replace("../", "")
            rid_to_target[rid] = clean_target
            
        # 3. Parse richValueRel: index -> rId
        rv_rel_xml = ET.fromstring(z.read("xl/richData/richValueRel.xml"))
        rel_indices = []
        for rel in rv_rel_xml.findall("{http://schemas.microsoft.com/office/spreadsheetml/2022/richvaluerel}rel"):
            rel_indices.append(rel.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
            
        # 4. Parse rdrichvalue.xml: rich value index -> index in richValueRel
        # Each <rv s="0"><v>{rel_idx}</v><v>5</v></rv>
        rdrich_xml = ET.fromstring(z.read("xl/richData/rdrichvalue.xml"))
        rv_to_rel_idx = []
        for rv in rdrich_xml.findall("{http://schemas.microsoft.com/office/spreadsheetml/2017/richdata}rv"):
            v_elems = rv.findall("{http://schemas.microsoft.com/office/spreadsheetml/2017/richdata}v")
            if v_elems:
                rv_to_rel_idx.append(int(v_elems[0].text))
            else:
                rv_to_rel_idx.append(None)
                
        # 5. Parse sheet1.xml rows
        sheet_xml = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows_elem = sheet_xml.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row")
        
        records = []
        for row in rows_elem:
            row_id = int(row.attrib.get("r", 0))
            if row_id == 1:
                continue  # Header
            
            cells = {c.attrib.get("r")[0]: c for c in row.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c")}
            
            def get_cell_val(col_letter):
                if col_letter not in cells:
                    return ""
                c = cells[col_letter]
                t = c.attrib.get("t")
                v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val = v.text if v is not None else ""
                if t == "s" and val.isdigit():
                    return shared_strings[int(val)]
                return val
            
            category = get_cell_val("A").strip()
            item_type = get_cell_val("C").strip()
            condition = get_cell_val("D").strip()
            severity = get_cell_val("E").strip().lower()
            
            # Get image for cell B
            image_filename = ""
            if "B" in cells:
                cell_b = cells["B"]
                vm = cell_b.attrib.get("vm")
                if vm and vm.isdigit():
                    # vm index is 1-based in metadata
                    vm_idx = int(vm) - 1
                    if 0 <= vm_idx < len(rv_to_rel_idx):
                        rel_idx = rv_to_rel_idx[vm_idx]
                        if rel_idx is not None and 0 <= rel_idx < len(rel_indices):
                            r_id = rel_indices[rel_idx]
                            img_zip_path = rid_to_target.get(r_id)
                            if img_zip_path and img_zip_path in z.namelist():
                                orig_ext = os.path.splitext(img_zip_path)[1]
                                safe_cat = category.lower().replace(" ", "_")
                                image_filename = f"row_{row_id:02d}_{safe_cat}_{severity}{orig_ext}"
                                dst_path = os.path.join(images_dir, image_filename)
                                with open(dst_path, "wb") as f_out:
                                    f_out.write(z.read(img_zip_path))
            
            records.append({
                "row_id": row_id,
                "category": category,
                "image_filename": image_filename,
                "type": item_type,
                "condition": condition,
                "severity": severity
            })

        # Save metadata CSV
        meta_csv = os.path.join(output_dir, "metadata.csv")
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["row_id", "category", "image_filename", "type", "condition", "severity"])
            writer.writeheader()
            writer.writerows(records)

        print(f"Extracted {len(records)} metadata records.")
        extracted_imgs = [r['image_filename'] for r in records if r['image_filename']]
        print(f"Successfully extracted {len(extracted_imgs)} images to {images_dir}")

if __name__ == "__main__":
    extract_dataset()
