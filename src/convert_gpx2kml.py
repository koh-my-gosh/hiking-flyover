#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ezgpx import GPX
import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

def ensure_style_defined(kml_file, style_id="style1"):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    ET.register_namespace("", ns["kml"])

    tree = ET.parse(kml_file)
    root = tree.getroot()

    doc = root.find("kml:Document", ns)
    if doc is None:
        print("No <Document> found in KML.")
        return

    # Check if style with given id exists
    style_exists = any(
        el.get("id") == style_id for el in doc.findall("kml:Style", ns)
    )

    if not style_exists:
        print(f"Adding missing style {style_id}")
        style = ET.Element(f"{{{ns['kml']}}}Style", id=style_id)
        linestyle = ET.SubElement(style, f"{{{ns['kml']}}}LineStyle")
        ET.SubElement(linestyle, f"{{{ns['kml']}}}color").text = "ff0000ff"  # red
        ET.SubElement(linestyle, f"{{{ns['kml']}}}width").text = "3"
        doc.insert(0, style)

        tree.write(kml_file, encoding="utf-8", xml_declaration=True)

def patch_missing_style(kml_file, style_id="style1"):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    ET.register_namespace("", ns["kml"])

    tree = ET.parse(kml_file)
    root = tree.getroot()

    doc = root.find("kml:Document", ns)
    if doc is None:
        print("No <Document> found.")
        return

    # Look for style definition
    exists = any(el.get("id") == style_id for el in doc.findall("kml:Style", ns))
    print(exists)
    if not exists:
        print(f"Adding missing style #{style_id}")
        style = ET.Element(f"{{{ns['kml']}}}Style", id=style_id)
        ls = ET.SubElement(style, f"{{{ns['kml']}}}LineStyle")
        ET.SubElement(ls, f"{{{ns['kml']}}}color").text = "ff0000ff"  # red (KML = aabbggrr)
        ET.SubElement(ls, f"{{{ns['kml']}}}width").text = "3"
        doc.insert(0, style)

        tree.write(kml_file, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument(
        "--altitude-mode", "-m", type=str, default="absolute",
        choices=["absolute", "clampToGround", "relativeToGround"],
        help="How to interpret elevations in KML (default: absolute)"
    )
    args = parser.parse_args()
    
    gpx = GPX(args.input)
    
    # Save as KML file
    gpx.to_kml(args.output)
    
    # ensure_style_defined(args.output, style_id="style1")
    patch_missing_style(args.output, "style1")