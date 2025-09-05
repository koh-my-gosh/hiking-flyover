#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpx_to_google_earth_tour.py

Create a Google Earth KML animation (gx:Tour) and time-enabled gx:Track
from one or more GPX files containing hiking tracks.

Usage:
  python gpx_to_google_earth_tour.py input.gpx [more.gpx ...] \
      -o output.kml --range 220 --tilt 70 --speed 6.0 --min-step 10 \
      --altitude-mode clampToGround --playback 1.0 --name "My Hike"

Key options:
  --speed          Constant speed (m/s) if timestamps are missing (default 5.0 m/s).
  --use-timestamps Use GPX timestamps for real timing. You can still scale with --playback.
  --playback       Multiply all durations by this factor (e.g., 0.5 = twice as fast).
  --min-step       Minimum spacing in meters between points (downsampling) to smooth the tour.
  --range          Camera range in meters (distance from LookAt point).
  --tilt           Camera tilt in degrees (0 = nadir, 90 = horizon).
  --altitude-mode  KML altitude mode: clampToGround|relativeToGround|absolute

Outputs:
  A single .kml containing:
    - Styled path (LineString)
    - Time-enabled gx:Track (for time slider playback)
    - gx:Tour (auto-play flyover)
"""

import argparse
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Tuple, Optional

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)

def parse_time(t: Optional[str]) -> Optional[datetime]:
    if not t:
        return None
    # Handle common GPX time forms like 2023-05-10T12:34:56Z
    try:
        if t.endswith("Z"):
            return datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(t).astimezone(timezone.utc)
    except Exception:
        return None

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    # Initial bearing (forward azimuth)
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1)*math.tan(phi2) - math.sin(phi1)*math.cos(dlambda)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def read_gpx_points(paths: List[str]) -> List[dict]:
    """
    Returns a flat list of dicts with keys: lat, lon, ele (may be None), time (datetime|None)
    Concatenates all tracks (in order).
    """
    pts = []
    for p in paths:
        tree = ET.parse(p)
        root = tree.getroot()
        ns = {"gpx": root.tag.split("}")[0].strip("{")}
        for trk in root.findall(".//gpx:trk", ns):
            for seg in trk.findall(".//gpx:trkseg", ns):
                for trkpt in seg.findall("gpx:trkpt", ns):
                    lat = float(trkpt.get("lat"))
                    lon = float(trkpt.get("lon"))
                    ele_el = trkpt.find("gpx:ele", ns)
                    ele = float(ele_el.text) if ele_el is not None else None
                    time_el = trkpt.find("gpx:time", ns)
                    t = parse_time(time_el.text.strip()) if time_el is not None and time_el.text else None
                    pts.append({"lat": lat, "lon": lon, "ele": ele, "time": t})
    return pts

def downsample_by_distance(pts: List[dict], min_step_m: float) -> List[dict]:
    if not pts:
        return pts
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        d = haversine_m(pts[i-1]["lat"], pts[i-1]["lon"], pts[i]["lat"], pts[i]["lon"])
        acc += d
        if acc >= min_step_m:
            out.append(pts[i])
            acc = 0.0
    if out[-1] is not pts[-1]:
        out.append(pts[-1])
    return out

def total_distance_m(pts: List[dict]) -> float:
    s = 0.0
    for i in range(1, len(pts)):
        s += haversine_m(pts[i-1]["lat"], pts[i-1]["lon"], pts[i]["lat"], pts[i]["lon"])
    return s

def build_styles() -> List[ET.Element]:
    styles = []

    # Path style
    style = ET.Element(f"{{{KML_NS}}}Style", id="pathStyle")
    ls = ET.SubElement(style, f"{{{KML_NS}}}LineStyle")
    ET.SubElement(ls, f"{{{KML_NS}}}color").text = "ff00a5ff"  # aabbggrr (opaque orange-ish: RR=ff, GG=a5, BB=00) -> Actually KML is aabbggrr; this resolves to orange.
    ET.SubElement(ls, f"{{{KML_NS}}}width").text = "4"
    ps = ET.SubElement(style, f"{{{KML_NS}}}PolyStyle")
    ET.SubElement(ps, f"{{{KML_NS}}}color").text = "33ffffff"
    styles.append(style)

    # Point (camera target) style (small dot)
    pstyle = ET.Element(f"{{{KML_NS}}}Style", id="pointStyle")
    istyle = ET.SubElement(pstyle, f"{{{KML_NS}}}IconStyle")
    ET.SubElement(istyle, f"{{{KML_NS}}}scale").text = "0.6"
    icon = ET.SubElement(istyle, f"{{{KML_NS}}}Icon")
    ET.SubElement(icon, f"{{{KML_NS}}}href").text = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
    styles.append(pstyle)

    return styles

def make_linestring_placemark(name: str, pts: List[dict], altitude_mode: str) -> ET.Element:
    placemark = ET.Element(f"{{{KML_NS}}}Placemark")
    ET.SubElement(placemark, f"{{{KML_NS}}}name").text = name
    ET.SubElement(placemark, f"{{{KML_NS}}}styleUrl").text = "#pathStyle"
    geom = ET.SubElement(placemark, f"{{{KML_NS}}}LineString")
    ET.SubElement(geom, f"{{{KML_NS}}}tessellate").text = "1"
    ET.SubElement(geom, f"{{{KML_NS}}}altitudeMode").text = altitude_mode
    coords = ET.SubElement(geom, f"{{{KML_NS}}}coordinates")
    def coord_str(p):
        if p["ele"] is not None and altitude_mode == "absolute":
            return f'{p["lon"]:.7f},{p["lat"]:.7f},{p["ele"]:.2f}'
        else:
            return f'{p["lon"]:.7f},{p["lat"]:.7f}'

    coords.text = " ".join(coord_str(p) for p in pts)
    return placemark

def make_gx_track_placemark(name: str, pts: List[dict], altitude_mode: str) -> ET.Element:
    placemark = ET.Element(f"{{{KML_NS}}}Placemark")
    ET.SubElement(placemark, f"{{{KML_NS}}}name").text = f"{name} (Time Track)"
    track = ET.SubElement(placemark, f"{{{GX_NS}}}Track")
    ET.SubElement(track, f"{{{KML_NS}}}altitudeMode").text = altitude_mode
    # <when> then <gx:coord> in pairs
    for p in pts:
        t = p["time"]
        if t is None:
            # If any timestamp is missing, skip times to avoid broken time slider
            # (The tour will still animate.)
            break
    else:
        for p in pts:
            ET.SubElement(track, f"{{{KML_NS}}}when").text = p["time"].isoformat().replace("+00:00", "Z")
        for p in pts:
            if p["ele"] is None:
                coord = f'{p["lon"]:.7f} {p["lat"]:.7f} 0'
            else:
                coord = f'{p["lon"]:.7f} {p["lat"]:.7f} {p["ele"]:.2f}'
            ET.SubElement(track, f"{{{GX_NS}}}coord").text = coord
    return placemark

def compute_segment_durations(
    pts: List[dict],
    use_timestamps: bool,
    default_speed_mps: float,
    playback_scale: float,
    min_duration: float = 0.2,
    max_duration: float = 5.0,
) -> List[float]:
    durs = []
    for i in range(1, len(pts)):
        if use_timestamps and pts[i-1]["time"] and pts[i]["time"]:
            dt = (pts[i]["time"] - pts[i-1]["time"]).total_seconds()
            dur = max(min_duration, min(max_duration, dt))
        else:
            d = haversine_m(pts[i-1]["lat"], pts[i-1]["lon"], pts[i]["lat"], pts[i]["lon"])
            dur = max(min_duration, min(max_duration, d / max(0.1, default_speed_mps)))
        durs.append(max(0.05, dur * playback_scale))
    if not durs:
        durs = [1.0]
    return durs

def make_tour(name: str, pts: List[dict], durations: List[float], range_m: float, tilt_deg: float, altitude_mode: str) -> ET.Element:
    tour = ET.Element(f"{{{GX_NS}}}Tour")
    ET.SubElement(tour, f"{{{KML_NS}}}name").text = f"{name} (Tour)"
    playlist = ET.SubElement(tour, f"{{{GX_NS}}}Playlist")

    # Start with an overview FlyTo to the first point
    if pts:
        first = pts[0]
        fly0 = ET.SubElement(playlist, f"{{{GX_NS}}}FlyTo")
        ET.SubElement(fly0, f"{{{GX_NS}}}duration").text = "1.2"
        ET.SubElement(fly0, f"{{{GX_NS}}}flyToMode").text = "smooth"
        look = ET.SubElement(fly0, f"{{{KML_NS}}}LookAt")
        ET.SubElement(look, f"{{{KML_NS}}}longitude").text = f"{first['lon']:.7f}"
        ET.SubElement(look, f"{{{KML_NS}}}latitude").text = f"{first['lat']:.7f}"
        ET.SubElement(look, f"{{{KML_NS}}}altitude").text = "0"
        ET.SubElement(look, f"{{{KML_NS}}}range").text = f"{range_m:.2f}"
        ET.SubElement(look, f"{{{KML_NS}}}tilt").text = f"{tilt_deg:.2f}"
        ET.SubElement(look, f"{{{KML_NS}}}heading").text = "0"
        ET.SubElement(look, f"{{{KML_NS}}}altitudeMode").text = altitude_mode

    # Fly along the track
    for i in range(1, len(pts)):
        prev = pts[i-1]; curr = pts[i]
        hdg = bearing_deg(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        fly = ET.SubElement(playlist, f"{{{GX_NS}}}FlyTo")
        ET.SubElement(fly, f"{{{GX_NS}}}duration").text = f"{durations[i-1]:.3f}"
        ET.SubElement(fly, f"{{{GX_NS}}}flyToMode").text = "smooth"
        look = ET.SubElement(fly, f"{{{KML_NS}}}LookAt")
        ET.SubElement(look, f"{{{KML_NS}}}longitude").text = f"{curr['lon']:.7f}"
        ET.SubElement(look, f"{{{KML_NS}}}latitude").text = f"{curr['lat']:.7f}"
        ET.SubElement(look, f"{{{KML_NS}}}altitude").text = "0"
        ET.SubElement(look, f"{{{KML_NS}}}range").text = f"{range_m:.2f}"
        ET.SubElement(look, f"{{{KML_NS}}}tilt").text = f"{tilt_deg:.2f}"
        ET.SubElement(look, f"{{{KML_NS}}}heading").text = f"{hdg:.2f}"
        ET.SubElement(look, f"{{{KML_NS}}}altitudeMode").text = altitude_mode
        # Optional pacing pause (0s keeps it smooth, but you can add <gx:Wait>)
        # wait = ET.SubElement(playlist, f"{{{GX_NS}}}Wait")
        # ET.SubElement(wait, f"{{{GX_NS}}}duration").text = "0.0"

    return tour

def build_kml(
    name: str,
    pts: List[dict],
    altitude_mode: str,
    use_timestamps: bool,
    speed_mps: float,
    playback_scale: float,
    range_m: float,
    tilt_deg: float,
) -> ET.ElementTree:
    kml = ET.Element(f"{{{KML_NS}}}kml")
    doc = ET.SubElement(kml, f"{{{KML_NS}}}Document")
    ET.SubElement(doc, f"{{{KML_NS}}}name").text = name

    # Styles
    for s in build_styles():
        doc.append(s)

    if not pts:
        ET.SubElement(doc, f"{{{KML_NS}}}Placemark").append(ET.Element(f"{{{KML_NS}}}name"))
        return ET.ElementTree(kml)

    # Add the path linestring
    doc.append(make_linestring_placemark(f"{name} Path", pts, altitude_mode))
    # Add gx:Track if times are present
    doc.append(make_gx_track_placemark(name, pts, altitude_mode))

    # Build the tour
    durs = compute_segment_durations(pts, use_timestamps, speed_mps, playback_scale)
    tour = make_tour(name, pts, durs, range_m, tilt_deg, altitude_mode)
    doc.append(tour)

    return ET.ElementTree(kml)

def main():
    ap = argparse.ArgumentParser(description="Create a Google Earth KML tour from GPX tracks.")
    ap.add_argument("gpx", nargs="+", help="Input GPX file(s)")
    ap.add_argument("-o", "--output", required=True, help="Output KML path")
    ap.add_argument("--name", default="Hiking Tour", help="Name for the KML Document")
    ap.add_argument("--range", type=float, default=220.0, help="Camera range (meters) for LookAt")
    ap.add_argument("--tilt", type=float, default=70.0, help="Camera tilt (degrees)")
    ap.add_argument("--speed", type=float, default=5.0, help="Default speed (m/s) if timestamps missing")
    ap.add_argument("--use-timestamps", action="store_true", help="Use GPX timestamps for segment timing")
    ap.add_argument("--playback", type=float, default=1.0, help="Multiply all durations by this factor")
    ap.add_argument("--min-step", type=float, default=8.0, help="Downsample step (meters) to smooth the tour")
    ap.add_argument("--altitude-mode", default="relativeToGround",
                    choices=["clampToGround", "relativeToGround", "absolute"],
                    help="KML altitude mode for path/camera")
    args = ap.parse_args()

    pts = read_gpx_points(args.gpx)
    if len(pts) < 2:
        raise SystemExit("No usable track points found in GPX.")

    # Downsample for a smoother, lighter tour
    if args.min_step > 0:
        pts = downsample_by_distance(pts, args.min_step)

    # Normalize times (optional): ensure non-decreasing
    # (Google Earth handles proper ISO 8601; we keep as-is if present.)

    dist_km = total_distance_m(pts) / 1000.0
    print(f"Track points after downsampling: {len(pts)} | Total distance: {dist_km:.2f} km")

    tree = build_kml(
        name=args.name,
        pts=pts,
        altitude_mode=args.altitude_mode,
        use_timestamps=args.use_timestamps,
        speed_mps=args.speed,
        playback_scale=args.playback,
        range_m=args.range,
        tilt_deg=args.tilt,
    )
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote KML tour to: {args.output}")
    print("Open the file in Google Earth. You’ll see the path and a Tour you can play.")

if __name__ == "__main__":
    main()
