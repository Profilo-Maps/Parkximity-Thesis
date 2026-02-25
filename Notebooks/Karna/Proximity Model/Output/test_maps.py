import os
import pandas as pd
import folium
from shapely import wkt
from shapely.geometry import shape
import json

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PARQUET_PATH = 'Notebooks\Karna\Proximity Model\Output\San_Francisco_County_California_USA_network.parquet'
OUTPUT_DIR   = "Notebooks\Karna\Proximity Model\Output\test_maps"
OUTPUT_NAME  = "test_map"

CENTER_LAT = 37 + 46/60 + 16.6/3600   # 37°46'16.6"N
CENTER_LON = -(122 + 25/60 + 27.1/3600)  # 122°25'27.1"W
ZOOM = 19

# ── LOAD ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows")

if df.empty:
    raise ValueError(f"Parquet is empty: {PARQUET_PATH}")

geom_cols = ['street_geometry', 'sidewalk_left_geometry', 'sidewalk_right_geometry',
             'bikeway_left_1_geometry', 'bikeway_right_1_geometry']
for col in geom_cols:
    if col in df.columns:
        n = df[col].dropna().__len__()
        print(f"  {col}: {n} non-null values")
    else:
        print(f"  {col}: COLUMN MISSING")

# ── GEOMETRY HELPER ────────────────────────────────────────────────────────────
def parse_geom(val):
    """Parse a geometry from WKB bytes, WKB hex string, WKT string, GeoJSON, or shapely object."""
    from shapely import wkb
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        if hasattr(val, 'geom_type'):        # already shapely
            return val
        if isinstance(val, bytes):           # WKB bytes
            return wkb.loads(val)
        if isinstance(val, dict):            # GeoJSON dict
            return shape(val)
        s = str(val).strip()
        if not s or s.lower() in ('none', 'nan', 'null'):
            return None
        if s.startswith('{'):               # GeoJSON string
            return shape(json.loads(s))
        # WKB hex string: all hex chars and even length
        if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 2 == 0:
            return wkb.loads(s, hex=True)
        return wkt.loads(s)                 # WKT fallback
    except Exception:
        return None

def geom_to_latlons(geom):
    """Return list of [lat, lon] pairs for LineString / MultiLineString."""
    if geom is None:
        return []
    if geom.geom_type == 'LineString':
        return [[c[1], c[0]] for c in geom.coords]
    if geom.geom_type == 'MultiLineString':
        coords = []
        for line in geom.geoms:
            coords.extend([[c[1], c[0]] for c in line.coords])
        return coords
    return []

# ── BUILD MAP ──────────────────────────────────────────────────────────────────
m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=ZOOM,
                tiles='CartoDB positron')

counts = {'street': 0, 'sidewalk': 0, 'bikeway': 0}

for _, row in df.iterrows():

    # ── Streets (red) ────────────────────────────────────────────────────────
    street_geom = parse_geom(row.get('street_geometry'))
    coords = geom_to_latlons(street_geom)
    if coords:
        name = row.get('name', '') or ''
        hw   = row.get('highway', '') or ''
        folium.PolyLine(
            coords, color='red', weight=3, opacity=0.85,
            tooltip=f"Street: {name} ({hw})"
        ).add_to(m)
        counts['street'] += 1

    # ── Sidewalks (blue) ─────────────────────────────────────────────────────
    for side in ('left', 'right'):
        sw_geom = parse_geom(row.get(f'sidewalk_{side}_geometry'))
        coords = geom_to_latlons(sw_geom)
        if coords:
            presence = row.get(f'sidewalk_{side}_presence', '')
            width    = row.get(f'sidewalk_{side}_width', '')
            folium.PolyLine(
                coords, color='blue', weight=2, opacity=0.75,
                tooltip=f"Sidewalk {side}: presence={presence}, width={width}"
            ).add_to(m)
            counts['sidewalk'] += 1

    # ── Bikeways (green) ─────────────────────────────────────────────────────
    for side in ('left', 'right'):
        for num in (1, 2):
            bk_geom = parse_geom(row.get(f'bikeway_{side}_{num}_geometry'))
            coords = geom_to_latlons(bk_geom)
            if coords:
                bk_type = row.get(f'bikeway_{side}_{num}_type', '')
                folium.PolyLine(
                    coords, color='green', weight=2, opacity=0.75,
                    tooltip=f"Bikeway {side}-{num}: {bk_type}"
                ).add_to(m)
                counts['bikeway'] += 1

# ── CENTER MARKER ──────────────────────────────────────────────────────────────
folium.Marker(
    [CENTER_LAT, CENTER_LON],
    popup='Market Street Block',
    icon=folium.Icon(color='orange', icon='map-marker')
).add_to(m)

# ── LEGEND ─────────────────────────────────────────────────────────────────────
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:9999;
            background:white;padding:10px 14px;border:2px solid #aaa;
            border-radius:6px;font-size:13px;line-height:1.8;">
<b>Legend</b><br>
<span style="color:red">&#9644;</span> Streets ({counts['street']})<br>
<span style="color:blue">&#9644;</span> Sidewalks ({counts['sidewalk']})<br>
<span style="color:green">&#9644;</span> Bikeways ({counts['bikeway']})
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

print(f"Streets: {counts['street']}  |  Sidewalks: {counts['sidewalk']}  |  Bikeways: {counts['bikeway']}")

# ── EXPORT MAP ────────────────────────────────────────────────────────────────
html_str  = m.get_root().render()
out_dir   = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
os.makedirs(out_dir, exist_ok=True)
out_path  = os.path.join(out_dir, f"{OUTPUT_NAME}.html")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_str)
print(f"Map saved to: {out_path}")

