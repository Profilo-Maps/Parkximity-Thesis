import os
import json
from typing import Any, cast
import pandas as pd
import folium
from shapely import wkt
from shapely.geometry import shape, box

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "San_Francisco_County_California_USA_network.parquet")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "test_maps")
OUTPUT_NAME  = "test_map"

CENTER_LAT = 37 + 46/60 + 16.6/3600    # 37°46'16.6"N
CENTER_LON = -(122 + 25/60 + 27.1/3600) # 122°25'27.1"W
ZOOM       = 19
BBOX_M     = 750  # bounding box radius in metres

# ── BOUNDING BOX (750 m around centre) ────────────────────────────────────────
# 1 degree lat  ≈ 111,320 m
# 1 degree lon  ≈ 111,320 * cos(lat) m
# Geometries are stored in UTM Zone 10N (EPSG:32610) — project centre to same CRS
from pyproj import Transformer
_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
_to_wgs = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)
CX, CY = _to_utm.transform(CENTER_LON, CENTER_LAT)

BBOX = box(CX - BBOX_M, CY - BBOX_M, CX + BBOX_M, CY + BBOX_M)

# Precompute degree-based bbox corners for the folium Rectangle display
import math
LAT_DEG = BBOX_M / 111320
LON_DEG = BBOX_M / (111320 * math.cos(math.radians(CENTER_LAT)))

# ── LOAD ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows")

if df.empty:
    raise ValueError(f"Parquet is empty: {PARQUET_PATH}")

geom_cols = ['street_geometry', 'sidewalk_left_geometry', 'sidewalk_right_geometry',
             'bikeway_left_1_geometry', 'bikeway_right_1_geometry']
for col in geom_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].dropna().__len__()} non-null values")
    else:
        print(f"  {col}: COLUMN MISSING")

# ── GEOMETRY HELPERS ───────────────────────────────────────────────────────────
def parse_geom(val):
    """Parse a geometry from WKB bytes, WKB hex string, WKT string, GeoJSON, or shapely object."""
    from shapely import wkb
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, 'geom_type'):
            return val
        if isinstance(val, bytes):
            return wkb.loads(val)
        if isinstance(val, dict):
            return shape(val)
        s = str(val).strip()
        if not s or s.lower() in ('none', 'nan', 'null'):
            return None
        if s.startswith('{'):
            return shape(json.loads(s))
        if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
            return wkb.loads(s, hex=True)
        return wkt.loads(s)
    except Exception:
        return None

def _utm_to_latlon(x, y):
    lon, lat = _to_wgs.transform(x, y)
    return [lat, lon]

def geom_to_latlons(geom):
    """Return list of [lat, lon] pairs for LineString / MultiLineString (UTM -> WGS84)."""
    if geom is None:
        return []
    if geom.geom_type == 'LineString':
        return [_utm_to_latlon(c[0], c[1]) for c in geom.coords]
    if geom.geom_type == 'MultiLineString':
        coords = []
        for line in geom.geoms:
            coords.extend([_utm_to_latlon(c[0], c[1]) for c in line.coords])
        return coords
    return []

def in_bbox(geom):
    """Return True if the geometry intersects the 750 m bounding box."""
    if geom is None:
        return False
    return BBOX.intersects(geom)

# ── BUILD MAP ──────────────────────────────────────────────────────────────────
m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=ZOOM,
               tiles='CartoDB positron')

# ── FEATURE GROUPS (one per toggle-able layer) ─────────────────────────────────
fg_streets = folium.FeatureGroup(name='Streets',              show=True)
fg_sw_sep  = folium.FeatureGroup(name='Sidewalks – separate', show=True)
fg_sw_buf  = folium.FeatureGroup(name='Sidewalks – buffered', show=True)
fg_bk_sep  = folium.FeatureGroup(name='Bikeways – separate',  show=True)
fg_bk_buf  = folium.FeatureGroup(name='Bikeways – buffered',  show=True)
fg_bbox    = folium.FeatureGroup(name='750 m bbox',           show=True)

# Draw the bounding box as a rectangle
folium.Rectangle(
    bounds=[
        [CENTER_LAT - LAT_DEG, CENTER_LON - LON_DEG],
        [CENTER_LAT + LAT_DEG, CENTER_LON + LON_DEG],
    ],
    color='gray', weight=1.5, fill=False, dash_array='6 4',
    tooltip=f"750 m bounding box"
).add_to(fg_bbox)

def add_endpoints(coords, color, target):
    """Draw a small circle at the start and end of a segment."""
    for pt in (coords[0], coords[-1]):
        folium.CircleMarker(
            location=pt, radius=3,
            color=color, fill=True, fill_color=color, fill_opacity=1.0,
            weight=1
        ).add_to(target)

def make_popup(title, attrs):
    """Build an HTML popup table from a title and a list of (label, value) pairs."""
    rows = ''.join(
        f'<tr><td style="padding:2px 8px 2px 0;color:#555;white-space:nowrap">'
        f'<b>{k}</b></td>'
        f'<td style="padding:2px 0">{v if (v is not None and str(v).strip() not in ("", "nan", "None")) else "<i>—</i>"}</td></tr>'
        for k, v in attrs
    )
    html = (
        f'<div style="font-family:sans-serif;font-size:12px;min-width:180px">'
        f'<b style="font-size:13px">{title}</b>'
        f'<table style="border-collapse:collapse;margin-top:4px">{rows}</table>'
        f'</div>'
    )
    return folium.Popup(html, max_width=320)


def _sw_attrs_from_row(row, side):
    return [
        ('Side',        side),
        ('Presence',    row.get(f'sidewalk_{side}_presence')),
        ('Width (m)',   row.get(f'sidewalk_{side}_width')),
        ('Surface',     row.get(f'sidewalk_{side}_surface')),
        ('Condition',   row.get(f'sidewalk_{side}_condition')),
        ('Kerb',        row.get(f'sidewalk_{side}_kerb')),
        ('Obstacle',    row.get(f'sidewalk_{side}_obstacle')),
        ('Street name', row.get('name')),
    ]

def _is_buffered(val):
    """Return True if a buffered column value is truthy (True, 'yes', non-empty string)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s not in ('', 'none', 'nan', 'null', 'no', '0', 'false')


def _draw_sidewalk(coords, row, side, label, target_sep, target_buf):
    presence = row.get(f'sidewalk_{side}_presence', '') if side else ''
    width    = row.get(f'sidewalk_{side}_width', '')    if side else ''
    buffered = _is_buffered(row.get(f'sidewalk_{side}_buffered')) if side else False
    color    = '#add8e6' if buffered else '#00008b'   # light blue vs dark blue
    kind     = 'buffered' if buffered else 'separate'
    target   = target_buf if buffered else target_sep
    attrs    = _sw_attrs_from_row(row, side) if side else [
        ('Name',       row.get('name')),
        ('Highway',    row.get('highway')),
        ('OSM ID',     row.get('osmid')),
        ('Length (m)', row.get('length')),
        ('Surface',    row.get('surface')),
        ('Access',     row.get('access')),
    ]
    folium.PolyLine(
        coords, color=color, weight=2, opacity=0.85,
        tooltip=f"Sidewalk {label} [{kind}]: presence={presence}, width={width}",
        popup=make_popup(f'Sidewalk ({label}) [{kind}]', attrs)
    ).add_to(target)
    add_endpoints(coords, color, target)


counts = {
    'street': 0,
    'sidewalk_separate': 0,
    'sidewalk_buffered': 0,
    'bikeway_separate': 0,
    'bikeway_buffered': 0
}

for _, row in df.iterrows():

    hw        = str(row.get('highway') or '').strip().lower()
    is_footway = hw == 'footway'

    # ── Streets (red) — skip footway edges ───────────────────────────────────
    street_geom = parse_geom(row.get('street_geometry'))
    if in_bbox(street_geom):
        coords = geom_to_latlons(street_geom)
        if coords:
            name = row.get('name', '') or ''
            if is_footway:
                # Treat the OSM footway edge itself as a sidewalk
                is_buffered = _is_buffered(row.get('sidewalk_buffered', False)) if not is_footway else False
                _draw_sidewalk(coords, row, None, f'footway – {name or "(unnamed)"}',
                               fg_sw_sep, fg_sw_buf)
                counts['sidewalk_buffered' if is_buffered else 'sidewalk_separate'] += 1
            else:
                street_attrs = [
                    ('Name',       row.get('name')),
                    ('Highway',    row.get('highway')),
                    ('OSM ID',     row.get('osmid')),
                    ('Lanes',      row.get('lanes')),
                    ('Max speed',  row.get('maxspeed')),
                    ('Oneway',     row.get('oneway')),
                    ('Length (m)', row.get('length')),
                    ('Surface',    row.get('surface')),
                    ('Access',     row.get('access')),
                ]
                folium.PolyLine(
                    coords, color='red', weight=3, opacity=0.85,
                    tooltip=f"Street: {name} ({hw})",
                    popup=make_popup(f'Street: {name or "(unnamed)"}', street_attrs)
                ).add_to(fg_streets)
                add_endpoints(coords, 'red', fg_streets)
                counts['street'] += 1

    # ── Sidewalks (blue) ─────────────────────────────────────────────────────
    for side in ('left', 'right'):
        sw_geom = parse_geom(row.get(f'sidewalk_{side}_geometry'))
        if in_bbox(sw_geom):
            coords = geom_to_latlons(sw_geom)
            if coords:
                is_sw_buffered = _is_buffered(row.get(f'sidewalk_{side}_buffered'))
                _draw_sidewalk(coords, row, side, side, fg_sw_sep, fg_sw_buf)
                counts['sidewalk_buffered' if is_sw_buffered else 'sidewalk_separate'] += 1

    # ── Bikeways (green) ─────────────────────────────────────────────────────
    for side in ('left', 'right'):
        bk_buffered = _is_buffered(row.get(f'bikeway_{side}_buffered'))
        bk_color    = '#90ee90' if bk_buffered else '#006400'   # light green vs dark green
        bk_kind     = 'buffered' if bk_buffered else 'separate'
        bk_target   = fg_bk_buf if bk_buffered else fg_bk_sep
        for num in (1, 2):
            bk_geom = parse_geom(row.get(f'bikeway_{side}_{num}_geometry'))
            if in_bbox(bk_geom):
                coords = geom_to_latlons(bk_geom)
                if coords:
                    bk_type = row.get(f'bikeway_{side}_{num}_type', '')
                    bk_attrs = [
                        ('Side',        side),
                        ('Lane #',      num),
                        ('Kind',        bk_kind),
                        ('Type',        row.get(f'bikeway_{side}_{num}_type')),
                        ('Width (m)',   row.get(f'bikeway_{side}_{num}_width')),
                        ('Surface',     row.get(f'bikeway_{side}_{num}_surface')),
                        ('Condition',   row.get(f'bikeway_{side}_{num}_condition')),
                        ('Separation',  row.get(f'bikeway_{side}_{num}_separation')),
                        ('Street name', row.get('name')),
                    ]
                    folium.PolyLine(
                        coords, color=bk_color, weight=2, opacity=0.85,
                        tooltip=f"Bikeway {side}-{num} [{bk_kind}]: {bk_type}",
                        popup=make_popup(f'Bikeway ({side}-{num}) [{bk_kind}]', bk_attrs)
                    ).add_to(bk_target)
                    add_endpoints(coords, bk_color, bk_target)
                    counts['bikeway_buffered' if bk_buffered else 'bikeway_separate'] += 1

# ── ADD FEATURE GROUPS TO MAP ──────────────────────────────────────────────────
for fg in (fg_streets, fg_sw_sep, fg_sw_buf, fg_bk_sep, fg_bk_buf, fg_bbox):
    fg.add_to(m)

# ── CENTER MARKER ──────────────────────────────────────────────────────────────
folium.Marker(
    [CENTER_LAT, CENTER_LON],
    popup='Market Street Block',
    icon=folium.Icon(color='orange', icon='map-marker')
).add_to(m)

# ── INTERACTIVE LEGEND WITH LAYER TOGGLES ──────────────────────────────────────
# Each checkbox calls toggleFG() with the folium-generated JS variable name for
# the corresponding FeatureGroup and the map variable name.
map_var   = m.get_name()
js_street = fg_streets.get_name()
js_sw_sep = fg_sw_sep.get_name()
js_sw_buf = fg_sw_buf.get_name()
js_bk_sep = fg_bk_sep.get_name()
js_bk_buf = fg_bk_buf.get_name()
js_bbox   = fg_bbox.get_name()

legend_html = f"""
<div id="px-legend" style="
    position:fixed;bottom:30px;left:30px;z-index:9999;
    background:white;padding:10px 14px;border:2px solid #aaa;
    border-radius:6px;font-size:13px;font-family:sans-serif;line-height:1.9;">
  <b style="font-size:14px">Legend</b><br>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_streets" checked
           onchange="toggleFG('{js_street}', this.checked)">
    <span style="color:red;font-size:18px;line-height:1">&#9644;</span>
    Streets ({counts['street']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_sep" checked
           onchange="toggleFG('{js_sw_sep}', this.checked)">
    <span style="color:#00008b;font-size:18px;line-height:1">&#9644;</span>
    Sidewalks – separate ({counts['sidewalk_separate']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_sw_buf" checked
           onchange="toggleFG('{js_sw_buf}', this.checked)">
    <span style="color:#add8e6;font-size:18px;line-height:1">&#9644;</span>
    Sidewalks – buffered ({counts['sidewalk_buffered']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_sep" checked
           onchange="toggleFG('{js_bk_sep}', this.checked)">
    <span style="color:#006400;font-size:18px;line-height:1">&#9644;</span>
    Bikeways – separate ({counts['bikeway_separate']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bk_buf" checked
           onchange="toggleFG('{js_bk_buf}', this.checked)">
    <span style="color:#90ee90;font-size:18px;line-height:1">&#9644;</span>
    Bikeways – buffered ({counts['bikeway_buffered']})
  </label>

  <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
    <input type="checkbox" id="cb_bbox" checked
           onchange="toggleFG('{js_bbox}', this.checked)">
    <span style="color:gray;font-size:18px;line-height:1">&#9644;</span>
    750 m bbox
  </label>
</div>

<script>
function toggleFG(varName, show) {{
  var layer = window[varName];
  var map   = window['{map_var}'];
  if (!layer || !map) return;
  if (show) {{ map.addLayer(layer); }} else {{ map.removeLayer(layer); }}
}}
</script>
"""
cast(Any, m.get_root()).html.add_child(folium.Element(legend_html))

print(f"Streets: {counts['street']}")
print(f"Sidewalks: {counts['sidewalk_separate']} separate + {counts['sidewalk_buffered']} buffered = {counts['sidewalk_separate'] + counts['sidewalk_buffered']} total")
print(f"Bikeways: {counts['bikeway_separate']} separate + {counts['bikeway_buffered']} buffered = {counts['bikeway_separate'] + counts['bikeway_buffered']} total")

# ── EXPORT MAP ────────────────────────────────────────────────────────────────
html_str = m.get_root().render()
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.html")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_str)
print(f"Map saved to: {out_path}")
