import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- Global Config ---
OUTPUT_DIR = Path("Notebooks/Karna/Proximity Model/Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Cities Config ---
CITIES_CONFIG = {"San Francisco County, California, USA"}


#---Data Pipeline---
def populate_schema(place: str) -> gpd.GeoDataFrame:
    """Load OSM street network for a place, map edge/node data into the proximity
    schema columns, export the result as a parquet to OUTPUT_DIR, and return it."""
    # --- Load OSM graph ---
    bike_tags = [
        "lanes", "lane_width", "maxspeed", "surface",
        # Primary bikeway tags
        "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
        "cycleway:left:surface", "cycleway:right:surface", "cycleway:surface",
        "cycleway:left:width", "cycleway:right:width", "cycleway:width",
        "cycleway:left:buffer", "cycleway:right:buffer", "cycleway:buffer",
        "cycleway:left:lane", "cycleway:right:lane",
        # Secondary bikeway tags (parallel/second bikeway on same edge)
        "cycleway:left:2", "cycleway:right:2", "cycleway:both:2",
        "cycleway:left:2:surface", "cycleway:right:2:surface",
        "cycleway:left:2:width", "cycleway:right:2:width",
        "cycleway:left:2:buffer", "cycleway:right:2:buffer",
        "cycleway:left:2:lane", "cycleway:right:2:lane",
        "bicycle", "incline",
    ]
    sidewalk_tags = [
        # Presence / side
        "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both",
        # Surface
        "sidewalk:left:surface", "sidewalk:right:surface",
        # Width
        "sidewalk:left:width", "sidewalk:right:width",
        # Incline
        "sidewalk:left:incline", "sidewalk:right:incline",
        # Smoothness / quality
        "sidewalk:left:smoothness", "sidewalk:right:smoothness",
        # Buffer (distance from road)
        "sidewalk:left:buffer", "sidewalk:right:buffer",
        # Foot access permission
        "foot",
    ]
    all_extra_tags = bike_tags + sidewalk_tags
    ox.settings.useful_tags_way = ox.settings.useful_tags_way + [
        tag for tag in all_extra_tags if tag not in ox.settings.useful_tags_way
    ]
    with tqdm(total=4, desc=f"Loading street network ({place})", unit="step") as pbar:
        pbar.set_postfix_str("downloading OSM graph")
        G = ox.graph_from_place(place, network_type="all")
        pbar.update(1)

        pbar.set_postfix_str("converting to GeoDataFrames")
        nodes, edges = ox.graph_to_gdfs(G)
        pbar.update(1)

        pbar.set_postfix_str("reprojecting to EPSG:32610")
        edges = edges.to_crs("EPSG:32610")
        nodes = nodes.to_crs("EPSG:32610")
        pbar.update(1)

        pbar.set_postfix_str("resetting edge index")
        edges_reset = edges.reset_index()
        pbar.update(1)

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")

    # --- Map OSM edge/node data into schema columns ---
    schema = _create_schema_dataframe()

    def _get(col):
        return edges_reset[col] if col in edges_reset.columns else None

    def _coalesce(*cols):
        """Return the first non-null series across the given OSM tag columns."""
        result = pd.Series(pd.NA, index=edges_reset.index, dtype=object)
        for col in cols:
            if col in edges_reset.columns:
                result = result.where(result.notna(), edges_reset[col])
        return result

    init_data = {col: pd.NA for col in schema.columns}
    init_data["street_geometry"] = edges_reset["geometry"].values
    populated = gpd.GeoDataFrame(
        init_data,
        index=edges_reset.index,
        geometry="street_geometry",
        crs="EPSG:32610",
    )

    # Street identifiers & topology
    populated["street_id"]     = _get("osmid")
    populated["start_node_id"] = _get("u")
    populated["end_node_id"]   = _get("v")

    # OSM tags that map directly
    for tag in ("name", "highway", "maxspeed", "oneway", "lanes", "lane_width", "surface"):
        populated[tag] = _get(tag)

    # Main street geometry (the edge LineString)
    populated["street_geometry"] = edges_reset["geometry"]

    # Start/end node point geometries looked up from the nodes GDF
    node_geom = nodes["geometry"]
    populated["start_node_geometry"] = edges_reset["u"].map(node_geom)
    populated["end_node_geometry"]   = edges_reset["v"].map(node_geom)

    # Mark intersection nodes (degree > 2 in the undirected sense)
    undirected_degree = pd.Series(dict(G.degree()), name="degree")
    populated["start_node_is_intersection_node"] = (
        edges_reset["u"].map(undirected_degree) > 2
    )
    populated["end_node_is_intersection_node"] = (
        edges_reset["v"].map(undirected_degree) > 2
    )
    # --- Bikeway data ---
    with tqdm(total=5, desc="Loading bikeway data", unit="step") as pbar:
        # Type: prefer side-specific tag, fall back to cycleway:both, then bare cycleway
        pbar.set_postfix_str("type")
        populated["bikeway_left_1_type"]  = _coalesce("cycleway:left",  "cycleway:both", "cycleway")
        populated["bikeway_right_1_type"] = _coalesce("cycleway:right", "cycleway:both", "cycleway")
        populated["bikeway_left_2_type"]  = _coalesce("cycleway:left:2",  "cycleway:both:2")
        populated["bikeway_right_2_type"] = _coalesce("cycleway:right:2", "cycleway:both:2")
        pbar.update(1)

        # Sub-type, surface, width, buffer
        pbar.set_postfix_str("quality / surface / width / buffer")
        populated["bikeway_left_1_quality"]  = _coalesce("cycleway:left:lane",  "cycleway:left:lane")
        populated["bikeway_right_1_quality"] = _coalesce("cycleway:right:lane", "cycleway:right:lane")
        populated["bikeway_left_1_surface"]  = _coalesce("cycleway:left:surface",  "cycleway:surface")
        populated["bikeway_right_1_surface"] = _coalesce("cycleway:right:surface", "cycleway:surface")
        populated["bikeway_left_1_width"]    = _coalesce("cycleway:left:width",  "cycleway:width")
        populated["bikeway_right_1_width"]   = _coalesce("cycleway:right:width", "cycleway:width")
        populated["bikeway_left_buffered"]   = _coalesce("cycleway:left:buffer",  "cycleway:buffer")
        populated["bikeway_right_buffered"]  = _coalesce("cycleway:right:buffer", "cycleway:buffer")
        pbar.update(1)

        # Permitted / incline
        pbar.set_postfix_str("permitted / incline")
        populated["bikeway_left_1_permitted"]  = _get("bicycle")
        populated["bikeway_right_1_permitted"] = _get("bicycle")
        populated["bikeway_left_1_incline"]    = _get("incline")
        populated["bikeway_right_1_incline"]   = _get("incline")
        pbar.update(1)

        # Secondary bikeway slots (_2)
        pbar.set_postfix_str("secondary bikeway slots")
        populated["bikeway_left_2_quality"]   = _get("cycleway:left:2:lane")
        populated["bikeway_right_2_quality"]  = _get("cycleway:right:2:lane")
        populated["bikeway_left_2_surface"]   = _get("cycleway:left:2:surface")
        populated["bikeway_right_2_surface"]  = _get("cycleway:right:2:surface")
        populated["bikeway_left_2_width"]     = _get("cycleway:left:2:width")
        populated["bikeway_right_2_width"]    = _get("cycleway:right:2:width")
        populated["bikeway_left_2_permitted"] = _get("bicycle")
        populated["bikeway_right_2_permitted"]= _get("bicycle")
        populated["bikeway_left_2_incline"]   = _get("incline")
        populated["bikeway_right_2_incline"]  = _get("incline")
        pbar.update(1)

        # Bikeway geometries — same LineString as street edge
        pbar.set_postfix_str("geometries")
        has_left   = populated["bikeway_left_1_type"].notna()
        has_right  = populated["bikeway_right_1_type"].notna()
        has_left2  = populated["bikeway_left_2_type"].notna()
        has_right2 = populated["bikeway_right_2_type"].notna()
        populated.loc[has_left,   "bikeway_left_1_geometry"]  = edges_reset.loc[has_left,   "geometry"]
        populated.loc[has_right,  "bikeway_right_1_geometry"] = edges_reset.loc[has_right,  "geometry"]
        populated.loc[has_left2,  "bikeway_left_2_geometry"]  = edges_reset.loc[has_left2,  "geometry"]
        populated.loc[has_right2, "bikeway_right_2_geometry"] = edges_reset.loc[has_right2, "geometry"]
        pbar.update(1)
    # --- Sidewalk data (centerline tags) ---
    with tqdm(total=4, desc="Loading sidewalk data", unit="step") as pbar:
        # Presence: prefer side-specific, fall back to sidewalk:both, then bare sidewalk
        pbar.set_postfix_str("presence")
        populated["sidewalk_left_presence"]  = _coalesce("sidewalk:left",  "sidewalk:both", "sidewalk")
        populated["sidewalk_right_presence"] = _coalesce("sidewalk:right", "sidewalk:both", "sidewalk")
        pbar.update(1)

        # Surface, width, incline, quality, buffer
        pbar.set_postfix_str("surface / width / incline / quality / buffer")
        populated["sidewalk_left_surface"]   = _get("sidewalk:left:surface")
        populated["sidewalk_right_surface"]  = _get("sidewalk:right:surface")
        populated["sidewalk_left_width"]     = _get("sidewalk:left:width")
        populated["sidewalk_right_width"]    = _get("sidewalk:right:width")
        populated["sidewalk_left_incline"]   = _get("sidewalk:left:incline")
        populated["sidewalk_right_incline"]  = _get("sidewalk:right:incline")
        populated["sidewalk_left_quality"]   = _get("sidewalk:left:smoothness")
        populated["sidewalk_right_quality"]  = _get("sidewalk:right:smoothness")
        populated["sidewalk_left_buffered"]  = _get("sidewalk:left:buffer")
        populated["sidewalk_right_buffered"] = _get("sidewalk:right:buffer")
        pbar.update(1)

        # Sidewalk geometries for centerline-tagged sidewalks
        pbar.set_postfix_str("geometries")
        has_sw_left  = populated["sidewalk_left_presence"].notna()
        has_sw_right = populated["sidewalk_right_presence"].notna()
        populated.loc[has_sw_left,  "sidewalk_left_geometry"]  = edges_reset.loc[has_sw_left,  "geometry"]
        populated.loc[has_sw_right, "sidewalk_right_geometry"] = edges_reset.loc[has_sw_right, "geometry"]
        populated = populated.set_geometry("street_geometry")
        pbar.update(1)

        # Separate cycleway and footway geometries
        pbar.set_postfix_str("separate cycleways / footways")
        populated = _populate_separate_bikelanes(populated, edges_reset)
        populated = _populate_footway_data(populated, edges_reset)
        pbar.update(1)
    # --- Export ---
    # Geometry columns other than the active one must be serialized to WKB so
    # they round-trip correctly through parquet (GeoParquet only encodes the
    # active geometry column; raw Shapely objects in object columns do not survive).
    # Use:
#     from shapely import wkb
#     gdf["bikeway_left_1_geometry"] = gdf["bikeway_left_1_geometry"].apply(
#     lambda h: wkb.loads(h, hex=True) if h else None)

    SECONDARY_GEOM_COLS = [
        "start_node_geometry", "end_node_geometry",
        "sidewalk_left_geometry", "sidewalk_right_geometry", "curb_return_geometry",
        "bikeway_left_1_geometry", "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
        "street_feature_geometry", "street_feature_geometry_projected",
        "sidewalk_left_feature_geometry", "sidewalk_left_feature_geometry_projected",
        "sidewalk_right_feature_geometry", "sidewalk_right_feature_geometry_projected",
        "bikeway_left_1_feature_geometry", "bikeway_left_1_feature_geometry_projected",
        "bikeway_left_2_feature_geometry", "bikeway_left_2_feature_geometry_projected",
        "bikeway_right_1_feature_geometry", "bikeway_right_1_feature_geometry_projected",
        "bikeway_right_2_feature_geometry", "bikeway_right_2_feature_geometry_projected",
        "crosswalk_start_geometry", "crosswalk_start_island_geometry",
        "crosswalk_end_geometry", "crosswalk_end_island_geometry",
        "sidewalk_left_curbramp_start_1_geometry", "sidewalk_left_curbramp_start_2_geometry",
        "sidewalk_left_curbramp_start_3_geometry", "sidewalk_left_curbramp_end_1_geometry",
        "sidewalk_left_curbramp_end_2_geometry", "sidewalk_left_curbramp_end_3_geometry",
        "sidewalk_right_curbramp_start_1_geometry", "sidewalk_right_curbramp_start_2_geometry",
        "sidewalk_right_curbramp_start_3_geometry", "sidewalk_right_curbramp_end_1_geometry",
        "sidewalk_right_curbramp_end_2_geometry", "sidewalk_right_curbramp_end_3_geometry",
    ]
    present_geom_cols = [col for col in SECONDARY_GEOM_COLS if col in populated.columns]
    with tqdm(total=len(present_geom_cols) + 1, desc="Exporting parquet", unit="col") as pbar:
        export_df = populated.copy()
        for col in present_geom_cols:
            pbar.set_postfix_str(col)
            export_df[col] = export_df[col].apply(
                lambda g: g.wkb_hex if hasattr(g, 'wkb_hex') else None
            )
            pbar.update(1)

        pbar.set_postfix_str("writing parquet")
        # Stringify any object columns that contain list values (pyarrow can't mix list/scalar)
        def _safe_str(x):
            if x is None:
                return None
            if isinstance(x, (list, np.ndarray)):
                return str(x)
            try:
                if pd.isna(x):
                    return None
            except (TypeError, ValueError):
                pass
            return str(x)
        for col in export_df.columns:
            if col in present_geom_cols or col == export_df.geometry.name:
                continue
            if export_df[col].dtype == object:
                has_list = any(isinstance(x, (list, np.ndarray)) for x in export_df[col])
                if has_list:
                    export_df[col] = export_df[col].apply(_safe_str)
        place_slug = place.replace(", ", "_").replace(" ", "_")
        output_path = OUTPUT_DIR / f"{place_slug}_network.parquet"
        export_df.to_parquet(output_path)
        pbar.update(1)

    print(f"Populated schema parquet exported to {output_path}")

    return populated

def _create_schema_dataframe():
    """Create an empty parquet file with columns from ProximitySchema.md"""
    columns = [
        # Street Centerlines
        "block_ids", "street_id", "block_sides", "public_data_id_street",
        "start_node_id", "start_node_is_block_node", "start_node_is_intersection_node",
        "end_node_id", "end_node_is_block_node", "end_node_is_intersection_node",
        "public_data_id_start_end_nodes", "normalized_bearing", "name", "highway",
        "maxspeed", "oneway", "lanes", "lane_width", "surface",
        # Street Centerline Features
        "street_feature_types", "public_data_id_street_feature",
        "street_feature_geometry", "street_feature_geometry_projected",
        # Sidewalk Left
        "sidewalk_left_ID", "sidewalk_left_block_ID", "sidewalk_left_presence",
        "public_data_id_sidewalk_left", "sidewalk_left_surface", "sidewalk_left_quality",
        "sidewalk_left_width", "sidewalk_left_incline", "sidewalk_left_buffered",
        # Curb Ramps Left
        "sidewalk_left_curbramp_start_1_ID", "public_data_id_sidewalk_left_curbramp_start_1",
        "sidewalk_left_curbramp_start_1_returnloc", "sidewalk_left_curbramp_start_1_returnposition",
        "sidewalk_left_curbramp_start_1_condition_score", "sidewalk_left_curbramp_start_1_geometry",
        "sidewalk_left_curbramp_start_2_ID", "public_data_id_sidewalk_left_curbramp_start_2",
        "sidewalk_left_curbramp_start_2_returnloc", "sidewalk_left_curbramp_start_2_returnposition",
        "sidewalk_left_curbramp_start_2_condition_score", "sidewalk_left_curbramp_start_2_geometry",
        "sidewalk_left_curbramp_start_3_ID", "public_data_id_sidewalk_left_curbramp_start_3",
        "sidewalk_left_curbramp_start_3_returnloc", "sidewalk_left_curbramp_start_3_returnposition",
        "sidewalk_left_curbramp_start_3_condition_score", "sidewalk_left_curbramp_start_3_geometry",
        "sidewalk_left_curbramp_end_1_ID", "public_data_id_sidewalk_left_curbramp_end_1",
        "sidewalk_left_curbramp_end_1_returnloc", "sidewalk_left_curbramp_end_1_returnposition",
        "sidewalk_left_curbramp_end_1_condition_score", "sidewalk_left_curbramp_end_1_geometry",
        "sidewalk_left_curbramp_end_2_ID", "public_data_id_sidewalk_left_curbramp_end_2",
        "sidewalk_left_curbramp_end_2_returnloc", "sidewalk_left_curbramp_end_2_returnposition",
        "sidewalk_left_curbramp_end_2_condition_score", "sidewalk_left_curbramp_end_2_geometry",
        "sidewalk_left_curbramp_end_3_ID", "public_data_id_sidewalk_left_curbramp_end_3",
        "sidewalk_left_curbramp_end_3_returnloc", "sidewalk_left_curbramp_end_3_returnposition",
        "sidewalk_left_curbramp_end_3_condition_score", "sidewalk_left_curbramp_end_3_geometry",
        # Sidewalk Left Features
        "sidewalk_left_feature_ids", "sidewalk_left_feature_types",
        "public_data_id_sidewalk_left_feature", "sidewalk_left_feature_geometry",
        "sidewalk_left_feature_geometry_projected",
        # Sidewalk Right
        "sidewalk_right_ID", "sidewalk_right_block_ID", "sidewalk_right_presence",
        "public_data_id_sidewalk_right", "sidewalk_right_surface", "sidewalk_right_quality",
        "sidewalk_right_width", "sidewalk_right_incline", "sidewalk_right_buffered",
        # Curb Ramps Right
        "sidewalk_right_curbramp_start_1_ID", "public_data_id_sidewalk_right_curbramp_start_1",
        "sidewalk_right_curbramp_start_1_returnloc", "sidewalk_right_curbramp_start_1_returnposition",
        "sidewalk_right_curbramp_start_1_condition_score", "sidewalk_right_curbramp_start_1_geometry",
        "sidewalk_right_curbramp_start_2_ID", "public_data_id_sidewalk_right_curbramp_start_2",
        "sidewalk_right_curbramp_start_2_returnloc", "sidewalk_right_curbramp_start_2_returnposition",
        "sidewalk_right_curbramp_start_2_condition_score", "sidewalk_right_curbramp_start_2_geometry",
        "sidewalk_right_curbramp_start_3_ID", "public_data_id_sidewalk_right_curbramp_start_3",
        "sidewalk_right_curbramp_start_3_returnloc", "sidewalk_right_curbramp_start_3_returnposition",
        "sidewalk_right_curbramp_start_3_condition_score", "sidewalk_right_curbramp_start_3_geometry",
        "sidewalk_right_curbramp_end_1_ID", "public_data_id_sidewalk_right_curbramp_end_1",
        "sidewalk_right_curbramp_end_1_returnloc", "sidewalk_right_curbramp_end_1_returnposition",
        "sidewalk_right_curbramp_end_1_condition_score", "sidewalk_right_curbramp_end_1_geometry",
        "sidewalk_right_curbramp_end_2_ID", "public_data_id_sidewalk_right_curbramp_end_2",
        "sidewalk_right_curbramp_end_2_returnloc", "sidewalk_right_curbramp_end_2_returnposition",
        "sidewalk_right_curbramp_end_2_condition_score", "sidewalk_right_curbramp_end_2_geometry",
        "sidewalk_right_curbramp_end_3_ID", "public_data_id_sidewalk_right_curbramp_end_3",
        "sidewalk_right_curbramp_end_3_returnloc", "sidewalk_right_curbramp_end_3_returnposition",
        "sidewalk_right_curbramp_end_3_condition_score", "sidewalk_right_curbramp_end_3_geometry",
        # Sidewalk Right Features
        "sidewalk_right_feature_ids", "sidewalk_right_feature_types",
        "public_data_id_sidewalk_right_feature", "sidewalk_right_feature_geometry",
        "sidewalk_right_feature_geometry_projected",
        # Crosswalks
        "crosswalk_start_id", "crosswalk_start_block_ids", "crosswalk_start_type",
        "public_data_id_crosswalk_start", "crosswalk_start_controlled", "crosswalk_start_marked",
        "crosswalk_start_markings", "crosswalk_start_signals", "crosswalk_start_island",
        "crosswalk_start_kerb", "crosswalk_start_tactile_paving", "crosswalk_start_traffic_calming",
        "crosswalk_start_continuous", "crosswalk_start_condition", "crosswalk_start_geometry",
        "crosswalk_start_island_geometry",
        "crosswalk_end_id", "crosswalk_end_block_ids", "crosswalk_end_type",
        "public_data_id_crosswalk_end", "crosswalk_end_controlled", "crosswalk_end_marked",
        "crosswalk_end_markings", "crosswalk_end_signals", "crosswalk_end_island",
        "crosswalk_end_kerb", "crosswalk_end_tactile_paving", "crosswalk_end_traffic_calming",
        "crosswalk_end_continuous", "crosswalk_end_condition", "crosswalk_end_geometry",
        "crosswalk_end_island_geometry",
        # Bikeways Left
        "bikeway_left_1_id", "bikeway_left_1_block_id", "public_data_id_bikeway_left_1",
        "bikeway_left_1_type", "bikeway_left_1_surface", "bikeway_left_1_quality",
        "bikeway_left_1_permitted", "bikeway_left_1_width", "bikeway_left_1_incline",
        "bikeway_left_buffered", "bikeway_left_2_id", "public_data_id_bikeway_left_2",
        "bikeway_left_2_type", "bikeway_left_2_surface", "bikeway_left_2_quality",
        "bikeway_left_2_permitted", "bikeway_left_2_width", "bikeway_left_2_incline",
        # Bikeway Features Left
        "bikeway_left_1_feature_ids", "bikeway_left_1_feature_types",
        "public_data_id_bikeway_left_1_features", "bikeway_left_1_feature_geometry",
        "bikeway_left_1_feature_geometry_projected", "bikeway_left_2_feature_types",
        "public_data_id_bikeway_left_2_features", "bikeway_left_2_feature_geometry",
        "bikeway_left_2_feature_geometry_projected",
        # Bikeways Right
        "bikeway_right_1_id", "bikeway_right_1_block_id", "public_data_id_bikeway_right_1",
        "bikeway_right_1_type", "bikeway_right_1_surface", "bikeway_right_1_quality",
        "bikeway_right_1_permitted", "bikeway_right_1_width", "bikeway_right_1_incline",
        "bikeway_right_buffered", "bikeway_right_2_id", "public_data_id_bikeway_right_2",
        "bikeway_right_2_type", "bikeway_right_2_surface", "bikeway_right_2_quality",
        "bikeway_right_2_permitted", "bikeway_right_2_width", "bikeway_right_2_incline",
        # Bikeway Features Right
        "bikeway_right_1_feature_ids", "bikeway_right_1_feature_types",
        "public_data_id_bikeway_right_1_features", "bikeway_right_1_feature_geometry",
        "bikeway_right_1_feature_geometry_projected", "bikeway_right_2_feature_types",
        "public_data_id_bikeway_right_2_features", "bikeway_right_2_feature_geometry",
        "bikeway_right_2_feature_geometry_projected",
        # Main Geometries
        "street_geometry", "start_node_geometry", "end_node_geometry",
        "sidewalk_left_geometry", "sidewalk_right_geometry", "curb_return_geometry",
        "bikeway_left_1_geometry", "bikeway_left_2_geometry",
        "bikeway_right_1_geometry", "bikeway_right_2_geometry",
    ]


    # Create empty GeoDataFrame with specified columns
    gdf = gpd.GeoDataFrame(columns=columns)

    
    return gdf

def _populate_separate_bikelanes(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Find independently mapped cycleway ways in edges_reset, spatially match each
    to its nearest road edge in populated, determine which side it lies on, and
    write the cycleway geometry + tags into the appropriate bikeway_left/right slot."""

    # Identify standalone cycleway edges
    CYCLEWAY_HIGHWAY_VALUES = {"cycleway", "path", "footway", "bridleway"}
    hw = edges_reset.get("highway", pd.Series(dtype=object))
    bicycle = edges_reset.get("bicycle", pd.Series(dtype=object))

    is_standalone = (
        hw.isin(CYCLEWAY_HIGHWAY_VALUES) |
        (hw.isin({"path", "footway"}) & bicycle.isin({"designated", "yes"}))
    )
    cycleways = edges_reset[is_standalone].copy()

    if cycleways.empty:
        return populated

    # Road edges only (exclude the cycleways themselves)
    roads = populated[~is_standalone.values].copy()
    road_sindex = roads.geometry.sindex

    def _side(road_geom, point):
        """Return 'left' or 'right' based on cross product of road direction × road→point."""
        coords = list(road_geom.coords)
        ax, ay = coords[0]
        bx, by = coords[-1]
        px, py = point.x, point.y
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        return "left" if cross > 0 else "right"

    def _free_slot(row, side):
        """Return '1' if the _1 slot for this side is empty, else '2', else None."""
        if pd.isna(row.get(f"bikeway_{side}_1_type")):
            return "1"
        if pd.isna(row.get(f"bikeway_{side}_2_type")):
            return "2"
        return None

    for _, cy_row in cycleways.iterrows():
        cy_geom   = cy_row["geometry"]
        cy_mid    = cy_geom.interpolate(0.5, normalized=True)
        cy_type   = cy_row.get("highway", pd.NA)
        cy_surf   = cy_row.get("surface",  pd.NA)
        cy_width  = cy_row.get("width",    pd.NA)
        cy_bicycle = cy_row.get("bicycle", pd.NA)
        cy_incline = cy_row.get("incline", pd.NA)

        # Find the nearest road edge
        best_pos = int(np.asarray(road_sindex.nearest(cy_geom)).flat[0])
        road_idx = roads.index[best_pos]
        road_geom = populated.at[road_idx, "street_geometry"]

        side = _side(road_geom, cy_mid)
        slot = _free_slot(populated.loc[road_idx], side)
        if slot is None:
            continue  # both slots already occupied

        prefix = f"bikeway_{side}_{slot}"
        populated.at[road_idx, f"{prefix}_type"]      = cy_type
        populated.at[road_idx, f"{prefix}_surface"]   = cy_surf
        populated.at[road_idx, f"{prefix}_width"]     = cy_width
        populated.at[road_idx, f"{prefix}_permitted"] = cy_bicycle
        populated.at[road_idx, f"{prefix}_incline"]   = cy_incline
        populated.at[road_idx, f"{prefix}_geometry"]  = cy_geom

    print(f"Matched {len(cycleways)} separate cycleway edges to road segments.")
    return populated

def _populate_footway_data(
    populated: gpd.GeoDataFrame,
    edges_reset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Find independently mapped footway/pedestrian ways in edges_reset, spatially
    match each to its nearest road edge in populated, determine which side it lies
    on, and write the footway geometry + tags into the sidewalk_left/right slot."""

    FOOTWAY_HIGHWAY_VALUES = {"footway", "pedestrian", "path", "steps", "corridor"}
    hw      = edges_reset.get("highway", pd.Series(dtype=object))
    foot    = edges_reset.get("foot",    pd.Series(dtype=object))
    bicycle = edges_reset.get("bicycle", pd.Series(dtype=object))

    is_footway = (
        hw.isin(FOOTWAY_HIGHWAY_VALUES) |
        (hw.isin({"path"}) & foot.isin({"designated", "yes"}))
    ) & ~bicycle.isin({"designated"})   # exclude dedicated cycleways already handled
    footways = edges_reset[is_footway].copy()

    if footways.empty:
        return populated

    # Road edges only
    roads = populated[~is_footway.values].copy()
    road_sindex = roads.geometry.sindex

    def _side(road_geom, point):
        coords = list(road_geom.coords)
        ax, ay = coords[0]
        bx, by = coords[-1]
        px, py = point.x, point.y
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        return "left" if cross > 0 else "right"

    def _slot_free(row, side):
        return pd.isna(row.get(f"sidewalk_{side}_presence"))

    for _, fw_row in footways.iterrows():
        fw_geom    = fw_row["geometry"]
        fw_mid     = fw_geom.interpolate(0.5, normalized=True)
        fw_surface = fw_row.get("surface",    pd.NA)
        fw_width   = fw_row.get("width",      pd.NA)
        fw_incline = fw_row.get("incline",    pd.NA)
        fw_smooth  = fw_row.get("smoothness", pd.NA)
        fw_hw      = fw_row.get("highway",    pd.NA)

        best_pos = int(np.asarray(road_sindex.nearest(fw_geom)).flat[0])
        road_idx = roads.index[best_pos]
        road_geom = populated.at[road_idx, "street_geometry"]

        side = _side(road_geom, fw_mid)
        if not _slot_free(populated.loc[road_idx], side):
            continue  # slot already occupied by centerline-tagged sidewalk

        prefix = f"sidewalk_{side}"
        populated.at[road_idx, f"{prefix}_presence"] = fw_hw
        populated.at[road_idx, f"{prefix}_surface"]  = fw_surface
        populated.at[road_idx, f"{prefix}_width"]    = fw_width
        populated.at[road_idx, f"{prefix}_incline"]  = fw_incline
        populated.at[road_idx, f"{prefix}_quality"]  = fw_smooth
        populated.at[road_idx, f"{prefix}_geometry"] = fw_geom

    print(f"Matched {len(footways)} separate footway edges to road segments.")
    return populated


def run_multi_city():
    results = {}
    for city in CITIES_CONFIG:
        print(f"\n=== Processing: {city} ===")
        results[city] = populate_schema(city)
    return results




# def create_boundary_grid():
#     return

# def block_assignment():
#     return

# def intersection_analysis():
#     return

if __name__ == "__main__":
    run_multi_city()

