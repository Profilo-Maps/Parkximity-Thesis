"""
ProximityModel.py

Complete implementation of the 21-step data pipeline described in OSMNetworkDescription.md.

Generates two parquet files per city:
  - [city]_sanity.parquet: Sanity buffer map with osmid and max_offset_width
  - [city]_network.parquet: Full denormalized street network with 206 columns

Features:
  - Integrates OSM street, footway, cycleway, and crossing data
  - Supports government data integration (centerlines, sidewalks, bikeways, curb ramps)
  - Generates offset geometries for buffered sidewalks and bikeways
  - Detects blocks using cycle detection with shoelace formula
  - Assigns default and government curb ramps
  - Generates crosswalk geometries (explicit and implicit)
  - GPU acceleration support for spatial operations

All geometries stored as WKB bytes in EPSG:4326.
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONSTANTS
# ============================================================================

import os
import sys
import warnings
import gc  # For explicit memory management
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    Point, LineString, Polygon, MultiPoint, MultiLineString,
    box as shapely_box
)
from shapely import wkb
from shapely.ops import nearest_points
from shapely.strtree import STRtree
import osmnx as ox
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

# OSMnx configuration
ox.settings.log_console = False
ox.settings.use_cache = True

# ============================================================================
# GPU DETECTION AND CONTEXT
# ============================================================================

# Try to import GPU libraries
GPU_AVAILABLE = False
_gpu_backend = None
GPU_MEMORY_LIMIT_PERCENT = 0.7  # Use max 70% of GPU memory to leave headroom for display

try:
    import cupy as cp
    # Verify GPU is actually accessible
    if cp.cuda.runtime.getDeviceCount() > 0:
        GPU_AVAILABLE = True
        _gpu_backend = 'cupy'
        # Set memory pool limit to prevent display issues
        try:
            mempool = cp.get_default_memory_pool()
            # Get total GPU memory
            device_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            total_memory = props['totalGlobalMem']
            # Set limit to 70% to leave headroom for display
            mempool.set_limit(size=int(total_memory * GPU_MEMORY_LIMIT_PERCENT))
        except Exception:
            pass  # If setting limit fails, continue without it
except (ImportError, Exception):
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            _gpu_backend = 'torch'
            # Set memory fraction for PyTorch
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_LIMIT_PERCENT)
            except Exception:
                pass
    except (ImportError, Exception):
        pass


class GPUContext:
    """Thin wrapper for GPU operations with graceful CPU fallback."""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = GPU_AVAILABLE and use_gpu  # Respect both hardware availability and user preference
        self.backend = _gpu_backend if self.use_gpu else None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def nearest_neighbors(self, reference_points, query_points, k=1):
        """
        Find k nearest neighbors using GPU if available.
        
        Args:
            reference_points: numpy array of shape (n, 2)
            query_points: numpy array of shape (m, 2)
            k: number of neighbors
        
        Returns:
            distances, indices (both numpy arrays)
        """
        if not self.use_gpu:
            # CPU fallback
            tree = cKDTree(reference_points)
            distances, indices = tree.query(query_points, k=k)
            return distances, indices
        
        try:
            if self.backend == 'cupy':
                # CuPy implementation with memory cleanup
                ref_gpu = cp.asarray(reference_points)
                query_gpu = cp.asarray(query_points)
                
                # Compute pairwise distances
                diff = query_gpu[:, cp.newaxis, :] - ref_gpu[cp.newaxis, :, :]
                distances_all = cp.sqrt(cp.sum(diff ** 2, axis=2))
                
                # Find k nearest
                indices_gpu = cp.argpartition(distances_all, k-1, axis=1)[:, :k]
                distances_gpu = cp.take_along_axis(distances_all, indices_gpu, axis=1)
                
                # Convert to numpy
                distances_result = cp.asnumpy(distances_gpu)
                indices_result = cp.asnumpy(indices_gpu)
                
                # Explicit GPU memory cleanup
                del ref_gpu, query_gpu, diff, distances_all, indices_gpu, distances_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
                return distances_result, indices_result
            
            elif self.backend == 'torch':
                # PyTorch implementation with memory cleanup
                ref_gpu = torch.from_numpy(reference_points).cuda()
                query_gpu = torch.from_numpy(query_points).cuda()
                
                # Compute pairwise distances
                diff = query_gpu.unsqueeze(1) - ref_gpu.unsqueeze(0)
                distances_all = torch.sqrt(torch.sum(diff ** 2, dim=2))
                
                # Find k nearest
                distances_gpu, indices_gpu = torch.topk(distances_all, k, largest=False, dim=1)
                
                # Convert to numpy
                distances_result = distances_gpu.cpu().numpy()
                indices_result = indices_gpu.cpu().numpy()
                
                # Explicit GPU memory cleanup
                del ref_gpu, query_gpu, diff, distances_all, distances_gpu, indices_gpu
                torch.cuda.empty_cache()
                
                return distances_result, indices_result
        
        except Exception as e:
            # If GPU fails, fall back to CPU
            print(f"  GPU computation failed, falling back to CPU: {e}")
            tree = cKDTree(reference_points)
            distances, indices = tree.query(query_points, k=k)
            return distances, indices
        
        # Fallback to CPU
        tree = cKDTree(reference_points)
        distances, indices = tree.query(query_points, k=k)
        return distances, indices


def get_gpu_info():
    """Get GPU information for display."""
    info = {
        'available': GPU_AVAILABLE,
        'backend': _gpu_backend,
        'gpu_names': [],
        'total_memory_gb': []
    }
    
    if not GPU_AVAILABLE:
        return info
    
    try:
        if _gpu_backend == 'cupy':
            import cupy as cp
            # Get device properties using runtime API
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                info['gpu_names'].append(props['name'].decode('utf-8'))
                info['total_memory_gb'].append(props['totalGlobalMem'] / (1024**3))
        elif _gpu_backend == 'torch':
            import torch
            info['gpu_names'].append(torch.cuda.get_device_name(0))
            info['total_memory_gb'].append(torch.cuda.get_device_properties(0).total_memory / 1e9)
    except Exception:
        pass
    
    return info

# ============================================================================
# CONSTANTS
# ============================================================================

# Highway sanity buffer parameters
HIGHWAY_SANITY = {
    'motorway': {'lane_width': 3.7, 'buffer': 3.0},
    'trunk': {'lane_width': 3.7, 'buffer': 3.0},
    'primary': {'lane_width': 3.4, 'buffer': 2.5},
    'secondary': {'lane_width': 3.3, 'buffer': 2.0},
    'tertiary': {'lane_width': 3.0, 'buffer': 1.5},
    'residential': {'lane_width': 3.0, 'buffer': 1.2},
    'service': {'lane_width': 2.7, 'buffer': 0.5},
    'unclassified': {'lane_width': 3.0, 'buffer': 1.2},
    'living_street': {'lane_width': 2.7, 'buffer': 0.5},
}

# Default lane widths by highway type
DEFAULT_LANE_WIDTHS = {
    'motorway': 3.7, 'trunk': 3.7, 'primary': 3.4,
    'secondary': 3.3, 'tertiary': 3.0, 'residential': 3.0,
    'service': 2.7, 'unclassified': 3.0, 'living_street': 2.7,
}

# Minimum sanity floor (meters)
MIN_SANITY_FLOOR = 1.5

# Curb ramp search radius (meters)
RAMP_SEARCH_RADIUS_M = 30.0

# NEW: Vertex deflection threshold (degrees)
DEFLECTION_THRESHOLD_DEG = 45.0

# NEW: Intersection analysis bounding box (meters)
INTERSECTION_BBOX_M = 30.0

# NEW: Default trustworthiness buffers (meters)
DEFAULT_OUTER_TRUST_BUFFER_M = 15.0
DEFAULT_INNER_TRUST_BUFFER_M = 5.0


# ============================================================================
# SECTION 2: CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class ColumnMappingConfig:
    """Maps government dataset column names to canonical fields."""
    # Street centerline columns
    street_id: Optional[str] = None
    street_name: Optional[str] = None
    street_highway: Optional[str] = None
    street_maxspeed: Optional[str] = None
    street_lanes: Optional[str] = None
    street_surface: Optional[str] = None
    
    # Sidewalk columns
    sidewalk_id: Optional[str] = None
    sidewalk_surface: Optional[str] = None
    sidewalk_width: Optional[str] = None
    sidewalk_incline: Optional[str] = None
    
    # Bikelane columns
    bikelane_id: Optional[str] = None
    bikelane_type: Optional[str] = None
    bikelane_surface: Optional[str] = None
    bikelane_width: Optional[str] = None
    
    # Curb ramp columns
    curbramp_id: Optional[str] = None
    curbramp_return_loc: Optional[str] = None
    curbramp_position: Optional[str] = None
    curbramp_condition: Optional[str] = None
    
    # Feature columns (street/sidewalk/bikeway features)
    feature_id: Optional[str] = None
    feature_type: Optional[str] = None


@dataclass
class GovernmentDataPaths:
    """Optional file paths for government data layers."""
    street_centerlines: Optional[str] = None
    intersection_nodes: Optional[str] = None
    sidewalks: Optional[str] = None
    bikelanes: Optional[str] = None
    curb_ramps: Optional[str] = None
    crosswalks: Optional[str] = None
    parcels: Optional[str] = None
    # Feature 8, 9, 10: Add feature data paths
    street_features: Optional[str] = None
    sidewalk_features: Optional[str] = None
    bikeway_features: Optional[str] = None


@dataclass
class CityConfig:
    """Configuration for a single city."""
    name: str
    government_data_paths: GovernmentDataPaths = field(default_factory=GovernmentDataPaths)
    column_mappings: ColumnMappingConfig = field(default_factory=ColumnMappingConfig)
    curbramp_trustworthy: bool = False


@dataclass
class GlobalConfig:
    """Global configuration for all cities."""
    output_dir: str
    cities: List[CityConfig]
    default_max_speed: int = 25
    curb_ramp_trustworthiness_outer_buffer: float = DEFAULT_OUTER_TRUST_BUFFER_M
    curb_ramp_trustworthiness_inner_buffer: float = DEFAULT_INNER_TRUST_BUFFER_M
    use_gpu: bool = True  # Set to False to disable GPU and prevent screen flickering
    python_env: str = "python"  # Path to Python executable (e.g., "python", "C:/path/to/python.exe", or conda env name)


def load_config(config_dict: Dict[str, Any]) -> GlobalConfig:
    """Load configuration from dictionary."""
    cities = []
    for city_dict in config_dict.get('cities', []):
        gov_paths_dict = city_dict.get('government_data_paths', {})
        gov_paths = GovernmentDataPaths(**gov_paths_dict)
        
        col_map_dict = city_dict.get('column_mappings', {})
        col_map = ColumnMappingConfig(**col_map_dict)
        
        city = CityConfig(
            name=city_dict['name'],
            government_data_paths=gov_paths,
            column_mappings=col_map,
            curbramp_trustworthy=city_dict.get('curbramp_trustworthy', False)
        )
        cities.append(city)
    
    return GlobalConfig(
        output_dir=config_dict.get('output_dir', 'Output'),
        cities=cities,
        default_max_speed=config_dict.get('default_max_speed', 25),
        curb_ramp_trustworthiness_outer_buffer=config_dict.get(
            'curb_ramp_trustworthiness_outer_buffer', DEFAULT_OUTER_TRUST_BUFFER_M
        ),
        curb_ramp_trustworthiness_inner_buffer=config_dict.get(
            'curb_ramp_trustworthiness_inner_buffer', DEFAULT_INNER_TRUST_BUFFER_M
        )
    )


# ============================================================================
# SECTION 3: SEQUENTIAL ID COUNTER
# ============================================================================

class SequentialIDCounter:
    """Sequential ID counter for assigning unique IDs."""
    
    def __init__(self, start: int = 1):
        self._next = start
        self._next_neg = -1
    
    def next(self) -> int:
        """Returns next positive sequential ID: 1, 2, 3, ..."""
        current = self._next
        self._next += 1
        return current
    
    def next_negative(self) -> int:
        """Returns next negative sequential ID: -1, -2, -3, ..."""
        current = self._next_neg
        self._next_neg -= 1
        return current


# Module-level counters (reset per city via _reset_pipeline_counters)
sidewalk_counter = SequentialIDCounter()
sidewalk_feature_counter = SequentialIDCounter()  # Feature 9: Sidewalk feature counter
bikeway_counter = SequentialIDCounter()
curbramp_counter = SequentialIDCounter()
crosswalk_counter = SequentialIDCounter()
split_node_counter = SequentialIDCounter()


# ============================================================================
# SECTION 4: HELPER FUNCTIONS
# ============================================================================

def normalize_tag(v: Any) -> Optional[str]:
    """Normalize OSM tag values to strings."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) > 0 else None
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ['', 'none', 'nan', 'unknown'] else None


def resolve_lane_width(edge_row, highway: str) -> float:
    """Resolve lane width from edge data or defaults."""
    # Try explicit width tag
    if 'width' in edge_row.index and edge_row['width'] is not None:
        try:
            w = float(edge_row['width'])
            if w > 0:
                return w
        except (ValueError, TypeError):
            pass
    
    # Try lanes * lane_width
    lanes = edge_row.get('lanes', 1)
    try:
        lanes = int(lanes) if lanes else 1
    except (ValueError, TypeError):
        lanes = 1
    
    if lanes < 1:
        lanes = 1
    
    # Use default lane width for highway type
    default_width = DEFAULT_LANE_WIDTHS.get(highway, 3.0)
    return lanes * default_width


def compute_bearing(geometry: LineString) -> float:
    """
    Compute bearing of a LineString from start to end point.
    Returns bearing in degrees [0, 360).
    """
    if geometry is None or geometry.is_empty:
        return 0.0
    
    coords = list(geometry.coords)
    if len(coords) < 2:
        return 0.0
    
    start = coords[0]
    end = coords[-1]
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    bearing = np.degrees(np.arctan2(dx, dy))
    return (bearing + 360) % 360


def compute_bearings_vectorized(geometries: gpd.GeoSeries) -> np.ndarray:
    """Vectorized bearing computation for GeoSeries of LineStrings."""
    bearings = np.zeros(len(geometries))
    
    for idx, geom in enumerate(geometries):
        if geom is not None and not geom.is_empty:
            coords = list(geom.coords)
            if len(coords) >= 2:
                start = coords[0]
                end = coords[-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                bearing = np.degrees(np.arctan2(dx, dy))
                bearings[idx] = (bearing + 360) % 360
    
    return bearings


def geom_to_wkb(geom) -> Optional[bytes]:
    """Convert Shapely geometry to WKB bytes."""
    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
        return None
    try:
        return geom.wkb
    except Exception:
        return None


def project_features_to_segment(
    feature_geometry: Point,
    segment_geometry: LineString
) -> Optional[Point]:
    """
    Project a feature point onto a segment LineString.
    Returns the closest point on the segment.
    """
    if feature_geometry is None or segment_geometry is None:
        return None
    if segment_geometry.is_empty:
        return None
    
    try:
        projected = segment_geometry.interpolate(
            segment_geometry.project(feature_geometry)
        )
        return projected
    except Exception:
        return None


# ============================================================================
# SECTION 5: STAGE 1 — SANITY BUFFER (Steps 1-3)
# ============================================================================

def _compute_parcel_distance_worker(args):
    """Worker function for parallel parcel distance computation with GPU support."""
    chunk_data, parcel_tree, parcel_geoms, use_gpu = args
    
    results = []
    
    if use_gpu and GPU_AVAILABLE:
        # GPU-accelerated path
        try:
            with GPUContext(use_gpu=use_gpu) as ctx:
                # Extract centroids from chunk
                centroids = np.array([row['geometry'].centroid.coords[0] for _, row in chunk_data.iterrows()])
                parcel_coords = np.array([g.centroid.coords[0] for g in parcel_geoms])
                
                # Find nearest neighbors using GPU
                distances, indices = ctx.nearest_neighbors(parcel_coords, centroids, k=1)
                
                # Compute actual distances to parcel geometries
                for i, (idx, row) in enumerate(chunk_data.iterrows()):
                    street_geom = row['geometry']
                    if street_geom is None or street_geom.is_empty:
                        results.append((row['osmid'], MIN_SANITY_FLOOR))
                        continue
                    
                    nearest_idx = indices[i, 0]
                    nearest_parcel = parcel_geoms[nearest_idx]
                    distance = street_geom.distance(nearest_parcel)
                    max_offset = max(distance, MIN_SANITY_FLOOR)
                    results.append((row['osmid'], max_offset))
        except Exception:
            # Fallback to CPU if GPU fails
            use_gpu = False
    
    if not use_gpu:
        # CPU fallback path
        for idx, row in chunk_data.iterrows():
            street_geom = row['geometry']
            if street_geom is None or street_geom.is_empty:
                results.append((row['osmid'], MIN_SANITY_FLOOR))
                continue
            
            # Find nearest parcel
            nearest_idx = parcel_tree.query(street_geom.centroid.coords[0])[1]
            nearest_parcel = parcel_geoms[nearest_idx]
            
            # Compute distance
            distance = street_geom.distance(nearest_parcel)
            max_offset = max(distance, MIN_SANITY_FLOOR)
            results.append((row['osmid'], max_offset))
    
    return results


def compute_sanity_buffer_from_parcels(
    network: gpd.GeoDataFrame,
    parcel_path: str,
    n_jobs: int = None
) -> pd.DataFrame:
    """
    Compute sanity buffer from parcel data with GPU acceleration.
    Aggregates parcels into building footprints before distance calculation.
    Returns DataFrame with columns: osmid, max_offset_width
    """
    print(f"  Loading parcels from {parcel_path}...")
    parcels = gpd.read_file(parcel_path)
    
    # Ensure same CRS
    if parcels.crs != network.crs:
        parcels = parcels.to_crs(network.crs)
    
    # Dissolve/aggregate parcels into continuous building masses
    print("  Aggregating parcels into building footprints...")
    try:
        # Try dissolve first (preserves GeoDataFrame structure)
        parcels = parcels.dissolve()
        # If result is a single polygon, explode it back to individual geometries
        if len(parcels) == 1:
            parcels = parcels.explode(index_parts=False)
    except Exception:
        # Fallback: use unary_union and create new GeoDataFrame
        aggregated_geom = parcels.geometry.unary_union
        if aggregated_geom.geom_type == 'MultiPolygon':
            parcels = gpd.GeoDataFrame(
                geometry=list(aggregated_geom.geoms),
                crs=parcels.crs
            )
        else:
            parcels = gpd.GeoDataFrame(
                geometry=[aggregated_geom],
                crs=parcels.crs
            )
    
    print(f"  Aggregated to {len(parcels)} building footprints")
    
    # Build spatial index
    print("  Building parcel spatial index...")
    parcel_geoms = list(parcels.geometry)
    parcel_coords = np.array([g.centroid.coords[0] for g in parcel_geoms])
    parcel_tree = cKDTree(parcel_coords)
    
    # Check GPU availability
    use_gpu = GPU_AVAILABLE
    if use_gpu:
        try:
            gpu_info = get_gpu_info()
            if gpu_info['available']:
                print(f"  Using GPU acceleration: {gpu_info['gpu_names'][0]}")
            else:
                use_gpu = False
        except:
            use_gpu = False
    
    if not use_gpu:
        print("  Using CPU (GPU not available)")
    
    # Parallel processing
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    print(f"  Computing distances (using {n_jobs} workers)...")
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, parcel_tree, parcel_geoms, use_gpu) for chunk in chunks]
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_compute_parcel_distance_worker, args),
            total=len(args),
            desc="  Parcel distances"
        ))
    
    # Flatten results
    all_results = []
    for chunk_results in results:
        all_results.extend(chunk_results)
    
    sanity_df = pd.DataFrame(all_results, columns=['osmid', 'max_offset_width'])
    return sanity_df


def compute_sanity_buffer_from_highway(
    network: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Compute sanity buffer from highway classification.
    Vectorized for performance.
    Returns DataFrame with columns: osmid, max_offset_width
    
    Formula from spec:
    - motorway/trunk: (Lanes * 1.85) + 3.0
    - primary: (Lanes * 1.7) + 2.5
    - secondary: (Lanes * 1.65) + 2.0
    - tertiary: (Lanes * 1.5) + 1.5
    - residential: (Lanes * 1.5) + 1.2
    - service: (Lanes * 1.35) + 0.5
    """
    print("  Computing sanity buffer from highway classification...")
    
    # Formula coefficients from spec
    FORMULAS = {
        'motorway': (1.85, 3.0),
        'trunk': (1.85, 3.0),
        'primary': (1.7, 2.5),
        'secondary': (1.65, 2.0),
        'tertiary': (1.5, 1.5),
        'residential': (1.5, 1.2),
        'service': (1.35, 0.5),
        'unclassified': (1.5, 1.2),
        'living_street': (1.35, 0.5),
    }
    
    # Normalize highway types (avoid full copy - work with view where possible)
    highway_normalized = network['highway'].apply(
        lambda x: normalize_tag(x) if pd.notna(x) else 'residential'
    ).apply(
        lambda x: x if x in FORMULAS else 'residential'
    )
    
    # Normalize lanes
    def parse_lanes(val):
        try:
            return max(1, int(val)) if pd.notna(val) else 1
        except (ValueError, TypeError):
            return 1
    
    lanes_normalized = network.get('lanes', 1).apply(parse_lanes)
    
    # Vectorized formula application
    def compute_offset(highway, lanes):
        lane_coef, buffer = FORMULAS[highway]
        max_offset = (lanes * lane_coef) + buffer
        return max(max_offset, MIN_SANITY_FLOOR)
    
    max_offset_width = [compute_offset(h, l) for h, l in zip(highway_normalized, lanes_normalized)]
    
    # Create minimal dataframe with only required columns
    sanity_df = pd.DataFrame({
        'osmid': network['osmid'].values,
        'max_offset_width': max_offset_width
    })
    return sanity_df


def export_sanity_buffer(
    sanity_df: pd.DataFrame,
    city_name: str,
    output_path: str
) -> None:
    """Export sanity buffer to parquet file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert to PyArrow table
    schema = pa.schema([
        ('osmid', pa.int64()),
        ('max_offset_width', pa.float64())
    ])
    
    table = pa.Table.from_pandas(sanity_df, schema=schema)
    pq.write_table(table, output_path)
    print(f"  Exported sanity buffer: {output_path}")


# ============================================================================
# SECTION 6: STAGE 2 — OSM DATA FETCH (Step 4, adapted)
# ============================================================================

def fetch_osm_network(city_config: CityConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fetch OSM network data for a city using geometry-first approach.
    Returns: (edges_gdf, crossings_cache)

    This function fetches raw OSM geometries without graph simplification,
    preserving all geometric detail needed for spatial analysis (offsets,
    buffers, intersection detection).

    Fetches four facility types per OSMNetworkDescription Phase 1, Step 1:

    Streets + inline sidewalk/cycleway tags:
        highway=motorway/trunk/primary/secondary/tertiary/residential/service/...
        sidewalk:both/left/right=yes/no/separate/none
        sidewalk:*:surface, sidewalk:*:width, sidewalk:*:incline
        cycleway:left/right=lane/track/opposite_lane/shared_lane/share_busway/separate/no
        cycleway:*:width, cycleway:*:surface
        oneway:bicycle=*, bicycle=*

    Separate footways (footway=sidewalk):
        highway=footway/path/pedestrian/steps + footway=sidewalk
        surface, smoothness, incline, width

    Separate cycleways:
        highway=cycleway
        surface, smoothness, incline, width, bicycle, oneway:bicycle

    Crossings (cached separately — never included in edges):
        highway=crossing or footway=crossing
        crossing:signals=yes/no, crossing:markings=*
        traffic_signals=*, button_operated=yes/no
        traffic_signals:sound=yes/no, traffic_signals:vibration=yes/no
        flashing_lights=yes/button/sensor, crossing:island=yes/no
        kerb=*, tactile_paving=yes/no
        traffic_calming=table, crossing:continuous=yes/no
    """
    print(f"  Fetching OSM data for {city_config.name}...")

    # Streets, separate footways (sidewalk), and separate cycleways in one query.
    # All inline sidewalk:*/cycleway:/bicycle/oneway:bicycle tags are returned as
    # columns automatically by OSMnx for any matched way.
    highway_tags = {
        'highway': [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'service', 'unclassified', 'living_street',
            'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
            'footway', 'path', 'pedestrian', 'steps',
            'cycleway',
        ]
    }

    # Fetch street geometries with all attributes
    print("  Fetching street geometries...")
    edges = ox.features_from_place(
        city_config.name,
        tags=highway_tags
    )

    # Filter out areas and unwanted highway types
    if not edges.empty:
        # Remove area features
        if 'area' in edges.columns:
            edges = edges[edges['area'] != 'yes']

        # Keep LineString geometries only.  Crossing nodes (Points) are handled
        # through crossings_cache; crossing Ways are also excluded here so they
        # don't appear as street segments.
        edges = edges[edges.geometry.type == 'LineString'].copy()

        # Exclude crossing ways — these belong only in crossings_cache
        if 'footway' in edges.columns:
            edges = edges[edges['footway'] != 'crossing']
        if 'highway' in edges.columns:
            edges = edges[edges['highway'] != 'crossing']

        # Ensure we have osmid
        if 'osmid' not in edges.columns and edges.index.name == 'osmid':
            edges = edges.reset_index()
        elif 'osmid' not in edges.columns:
            edges['osmid'] = range(len(edges))

    print(f"  Found {len(edges)} street segments")

    # Assign start/end node IDs directly from edge endpoint coordinates
    coord_to_id = {}
    node_id_counter = 1
    start_node_ids = []
    end_node_ids = []

    for row in edges.itertuples():
        geom = row.geometry
        if geom and hasattr(geom, 'coords'):
            start_key = (round(geom.coords[0][0], 7), round(geom.coords[0][1], 7))
            if start_key not in coord_to_id:
                coord_to_id[start_key] = node_id_counter
                node_id_counter += 1

            end_key = (round(geom.coords[-1][0], 7), round(geom.coords[-1][1], 7))
            if end_key not in coord_to_id:
                coord_to_id[end_key] = node_id_counter
                node_id_counter += 1

            start_node_ids.append(coord_to_id[start_key])
            end_node_ids.append(coord_to_id[end_key])
        else:
            start_node_ids.append(None)
            end_node_ids.append(None)

    edges['start_node_osmid'] = start_node_ids
    edges['end_node_osmid'] = end_node_ids
    print(f"  Assigned node IDs to {len(edges)} edges ({len(coord_to_id)} unique nodes)")

    # Initialize node geometry columns from edge endpoints.
    # merge_government_intersection_nodes (Stage 3) may update these with
    # government-authoritative positions; they must exist beforehand.
    edges['start_node_geometry'] = edges['geometry'].apply(
        lambda g: Point(list(g.coords)[0]) if g is not None and hasattr(g, 'coords') else None
    )
    edges['end_node_geometry'] = edges['geometry'].apply(
        lambda g: Point(list(g.coords)[-1]) if g is not None and hasattr(g, 'coords') else None
    )
    edges['public_data_id_start_end_nodes'] = None

    # Initialize street feature columns
    edges['street_feature_types'] = None
    edges['public_data_id_street_feature'] = None
    edges['street_feature_geometry'] = None
    edges['street_feature_geometry_projected'] = None

    # Fetch crossing data separately and cache it.
    # Captures both Point nodes (highway=crossing) and Way LineStrings
    # (footway=crossing), preserving all crossing attribute tags so that
    # extract_crosswalk_tags can read crossing:signals, crossing:markings,
    # traffic_signals:sound/vibration, button_operated, flashing_lights,
    # crossing:island, kerb, tactile_paving, traffic_calming, crossing:continuous.
    print("  Fetching crossing data...")
    crossings_cache = gpd.GeoDataFrame()
    try:
        # OR query: features tagged highway=crossing OR footway=crossing
        crossing_tags = {
            'highway': ['crossing'],
            'footway': ['crossing'],
        }
        crossings = ox.features_from_place(city_config.name, tags=crossing_tags)
        if not crossings.empty:
            # Keep Point nodes AND LineString ways; exclude Polygon/area crossings
            crossings_cache = crossings[
                crossings.geometry.type.isin(['Point', 'LineString'])
            ].copy()
            print(f"  Found {len(crossings_cache)} crossing features "
                  f"({(crossings_cache.geometry.type == 'Point').sum()} nodes, "
                  f"{(crossings_cache.geometry.type == 'LineString').sum()} ways)")
    except Exception as e:
        print(f"  Warning: Could not fetch crossings: {e}")

    print(f"  Completed fetch: {len(edges)} edges")
    return edges, crossings_cache


# ============================================================================
# SECTION 7: STAGE 3 — GOVERNMENT DATA INTEGRATION (Step 4, new)
# ============================================================================

def load_geospatial_file(
    filepath: str,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    geom_col: Optional[str] = None,
    wkt_col: Optional[str] = None
) -> Optional[gpd.GeoDataFrame]:
    """
    Robust file loader that handles multiple formats:
    - GeoJSON/Shapefile: Direct geometry
    - CSV with lat/lon columns
    - CSV with WKT geometry column
    - CSV with WKB geometry column
    
    Args:
        filepath: Path to the file
        lat_col: Name of latitude column (for CSV)
        lon_col: Name of longitude column (for CSV)
        geom_col: Name of WKB geometry column (for CSV)
        wkt_col: Name of WKT geometry column (for CSV)
    
    Returns:
        GeoDataFrame or None if file doesn't exist
    """
    if not filepath or not os.path.exists(filepath):
        return None
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Handle GeoJSON and Shapefiles directly
    if file_ext in ['.geojson', '.json', '.shp']:
        try:
            return gpd.read_file(filepath)
        except Exception as e:
            print(f"  ⚠ Error reading {filepath}: {e}")
            return None
    
    # Handle CSV files
    if file_ext == '.csv':
        try:
            df = pd.read_csv(filepath, engine="python", on_bad_lines="warn")
            
            # Try WKT geometry column first
            if wkt_col and wkt_col in df.columns:
                from shapely import wkt as shapely_wkt
                geometries = df[wkt_col].apply(lambda x: shapely_wkt.loads(x) if pd.notna(x) else None)
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf[gdf.geometry.notna()]
            
            # Try WKB geometry column
            if geom_col and geom_col in df.columns:
                geometries = df[geom_col].apply(lambda x: wkb.loads(x) if pd.notna(x) else None)
                gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                return gdf[gdf.geometry.notna()]
            
            # Try lat/lon columns
            if lat_col and lon_col:
                if lat_col in df.columns and lon_col in df.columns:
                    valid = df.dropna(subset=[lat_col, lon_col])
                    geometries = gpd.points_from_xy(valid[lon_col], valid[lat_col])
                    return gpd.GeoDataFrame(valid, geometry=geometries, crs="EPSG:4326")
            
            # Auto-detect lat/lon columns
            lat_candidates = ['latitude', 'lat', 'y', 'yloc']
            lon_candidates = ['longitude', 'lon', 'lng', 'long', 'x', 'xloc']
            
            df_cols_lower = {col.lower(): col for col in df.columns}
            
            lat_found = None
            lon_found = None
            
            for candidate in lat_candidates:
                if candidate in df_cols_lower:
                    lat_found = df_cols_lower[candidate]
                    break
            
            for candidate in lon_candidates:
                if candidate in df_cols_lower:
                    lon_found = df_cols_lower[candidate]
                    break
            
            if lat_found and lon_found:
                print(f"  Auto-detected coordinate columns: {lat_found}, {lon_found}")
                valid = df.dropna(subset=[lat_found, lon_found])
                geometries = gpd.points_from_xy(valid[lon_found], valid[lat_found])
                return gpd.GeoDataFrame(valid, geometry=geometries, crs="EPSG:4326")
            
            print(f"  ⚠ Could not find geometry in CSV. Tried:")
            print(f"     - WKT column: {wkt_col}")
            print(f"     - WKB column: {geom_col}")
            print(f"     - Lat/Lon columns: {lat_col}/{lon_col}")
            print(f"     - Auto-detect failed. Available columns: {list(df.columns)[:10]}")
            return None
            
        except Exception as e:
            print(f"  ⚠ Error reading CSV {filepath}: {e}")
            return None
    
    print(f"  ⚠ Unsupported file format: {file_ext}")
    return None


def load_government_centerlines(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """Load government street centerline data if provided."""
    path = city_config.government_data_paths.street_centerlines
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading government centerlines from {path}...")
    gdf = load_geospatial_file(path)
    return gdf


def merge_government_centerlines(
    osm_edges: gpd.GeoDataFrame,
    gov_lines: gpd.GeoDataFrame,
    city_config: CityConfig
) -> gpd.GeoDataFrame:
    """
    Merge government centerline data with OSM edges.
    Uses Hausdorff distance matching (<20m threshold in UTM projection).
    Optimized with batch updates.
    """
    if gov_lines is None or gov_lines.empty:
        return osm_edges
    
    print("  Merging government centerlines with OSM data...")
    
    # Ensure same CRS (working CRS is UTM)
    if gov_lines.crs != osm_edges.crs:
        gov_lines = gov_lines.to_crs(osm_edges.crs)
    
    # Build spatial index (already in UTM — distances are in meters)
    osm_tree = STRtree(osm_edges.geometry)
    
    col_map = city_config.column_mappings
    
    # Batch updates
    updates = {}  # osm_idx -> {col: value}
    
    # Match government lines to OSM edges
    for idx in tqdm(gov_lines.index, total=len(gov_lines), desc="  Matching centerlines"):
        gov_row = gov_lines.loc[idx]
        gov_geom = gov_row.geometry
        
        if gov_geom is None or gov_geom.is_empty:
            continue
        
        # Find nearby OSM edges (buffer ~20m — already in meters since CRS is UTM)
        nearby_indices = osm_tree.query(gov_geom, predicate='dwithin', distance=20.0)
        
        if len(nearby_indices) == 0:
            continue
        
        # Find best match by Hausdorff distance (in meters)
        best_idx = None
        best_dist = float('inf')
        
        for osm_idx in nearby_indices:
            osm_geom = osm_edges.iloc[osm_idx].geometry
            try:
                dist = gov_geom.hausdorff_distance(osm_geom)
                if dist < best_dist and dist < 20.0:  # 20m threshold
                    best_dist = dist
                    best_idx = osm_idx
            except Exception:
                continue
        
        if best_idx is not None:
            osm_edge_idx = osm_edges.index[best_idx]
            
            if osm_edge_idx not in updates:
                updates[osm_edge_idx] = {}
            
            # Replace geometry (use government geometry in working CRS)
            updates[osm_edge_idx]['geometry'] = gov_geom
            
            # Set public_data_id_street
            if col_map.street_id and col_map.street_id in gov_row.index:
                updates[osm_edge_idx]['public_data_id_street'] = gov_row[col_map.street_id]
            
            # Apply column mappings
            if col_map.street_name and col_map.street_name in gov_row.index:
                updates[osm_edge_idx]['name'] = gov_row[col_map.street_name]
            if col_map.street_highway and col_map.street_highway in gov_row.index:
                updates[osm_edge_idx]['highway'] = gov_row[col_map.street_highway]
            if col_map.street_maxspeed and col_map.street_maxspeed in gov_row.index:
                updates[osm_edge_idx]['maxspeed'] = gov_row[col_map.street_maxspeed]
            if col_map.street_lanes and col_map.street_lanes in gov_row.index:
                updates[osm_edge_idx]['lanes'] = gov_row[col_map.street_lanes]
            if col_map.street_surface and col_map.street_surface in gov_row.index:
                updates[osm_edge_idx]['surface'] = gov_row[col_map.street_surface]
    
    # Batch apply all updates
    for osm_idx, cols in updates.items():
        for col, value in cols.items():
            osm_edges.at[osm_idx, col] = value
    
    print(f"  Merged government centerlines")
    return osm_edges


def load_government_sidewalks(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """Load government sidewalk data if provided."""
    path = city_config.government_data_paths.sidewalks
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading government sidewalks from {path}...")
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_government_bikelanes(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """Load government bikelane data if provided."""
    path = city_config.government_data_paths.bikelanes
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading government bikelanes from {path}...")
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_government_intersection_nodes(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load government intersection nodes if provided.
    Feature 3: Load government intersection nodes (Step 2 & 7)
    """
    path = city_config.government_data_paths.intersection_nodes
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading government intersection nodes from {path}...")
    
    # Geometry is auto-detected by load_geospatial_file
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def merge_government_intersection_nodes(
    edges: gpd.GeoDataFrame,
    gov_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Join government intersection nodes to OSM edge start/end nodes (Step 2).

    For each edge endpoint that lies within ~10 m of a government node, the
    start_node_geometry / end_node_geometry column is updated with the
    government point and the government ID is recorded in
    public_data_id_start_end_nodes as a (start_id, end_id) tuple.
    The OSM node ID columns (start_node_osmid / end_node_osmid) are preserved
    for graph connectivity; only the geometry and public ID are updated.
    """
    if gov_nodes is None or gov_nodes.empty:
        return edges

    print("  Joining government intersection nodes to edge endpoints...")

    if gov_nodes.crs != edges.crs:
        gov_nodes = gov_nodes.to_crs(edges.crs)

    gov_tree = STRtree(gov_nodes.geometry)
    # 10 m tolerance (working CRS is UTM, so distances are in meters)
    match_threshold = 10.0

    # Auto-detect the ID column in the government nodes dataset
    def _get_gov_id(row):
        for candidate in ['id', 'ID', 'OBJECTID', 'node_id', 'NodeID', 'FID']:
            val = row.get(candidate)
            if val is not None:
                return str(val)
        return None

    matched = 0
    for idx, row in edges.iterrows():
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue

        coords = list(geom.coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])

        start_gov_id = None
        end_gov_id = None

        # Match start node
        nearby = gov_tree.query(start_pt, predicate='dwithin', distance=match_threshold)
        if len(nearby) > 0:
            gov_row = gov_nodes.iloc[nearby[0]]
            edges.at[idx, 'start_node_geometry'] = gov_row.geometry
            start_gov_id = _get_gov_id(gov_row)

        # Match end node
        nearby = gov_tree.query(end_pt, predicate='dwithin', distance=match_threshold)
        if len(nearby) > 0:
            gov_row = gov_nodes.iloc[nearby[0]]
            edges.at[idx, 'end_node_geometry'] = gov_row.geometry
            end_gov_id = _get_gov_id(gov_row)

        if start_gov_id is not None or end_gov_id is not None:
            edges.at[idx, 'public_data_id_start_end_nodes'] = (start_gov_id, end_gov_id)
            matched += 1

    print(f"  Matched government intersection nodes to {matched} edges")
    return edges


def load_government_crosswalks(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load government crosswalk data if provided.
    Feature 4: Load government crosswalk data (Step 20)
    """
    path = city_config.government_data_paths.crosswalks
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading government crosswalks from {path}...")
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_street_features(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load street feature data (fire hydrants, street lights, benches, etc.).
    Feature 8: Load and populate street feature geometry
    """
    path = city_config.government_data_paths.street_features
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading street features from {path}...")
    
    # Geometry is auto-detected by load_geospatial_file
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_sidewalk_features(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load sidewalk feature data (benches, trash cans, trees, etc.).
    Feature 9: Load and populate sidewalk feature geometry
    """
    path = city_config.government_data_paths.sidewalk_features
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading sidewalk features from {path}...")
    
    # Geometry is auto-detected by load_geospatial_file
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def load_bikeway_features(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load bikeway feature data (bike parking, repair stations, etc.).
    Feature 10: Load and populate bikeway feature geometry
    """
    path = city_config.government_data_paths.bikeway_features
    if path is None or not os.path.exists(path):
        return None
    
    print(f"  Loading bikeway features from {path}...")
    
    # Geometry is auto-detected by load_geospatial_file
    gdf = load_geospatial_file(path)
    
    if gdf is not None:
        # Ensure CRS matches (assume WGS84 if not specified)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def merge_government_sidewalks(
    edges: gpd.GeoDataFrame,
    gov_sidewalks: gpd.GeoDataFrame,
    city_config: CityConfig
) -> gpd.GeoDataFrame:
    """
    Merge government sidewalk data with OSM edges.
    Feature 3: Load government sidewalk data (Step 2 - partial)
    """
    print("  Merging government sidewalk data...")

    col_map = city_config.column_mappings

    # Ensure same CRS
    if gov_sidewalks.crs != edges.crs:
        gov_sidewalks = gov_sidewalks.to_crs(edges.crs)

    # Spatial join to match government sidewalks to OSM edges
    # Use buffer for matching tolerance (10 meters — working CRS is UTM)
    # Optimize: Create temporary GeoDataFrame with only geometry and index (avoid full copy)
    edges_buffered = gpd.GeoDataFrame(
        geometry=edges.geometry.buffer(10.0),  # 10m buffer in meters (UTM)
        index=edges.index,
        crs=edges.crs
    )

    joined = gpd.sjoin(gov_sidewalks, edges_buffered, how='inner', predicate='intersects')
    del edges_buffered  # Free memory immediately

    # Group by edge index to handle multiple sidewalk matches
    for edge_idx in joined['index_right'].unique():
        matches = joined[joined['index_right'] == edge_idx]

        # Determine which side (left or right) based on geometry position
        edge_geom = edges.loc[edge_idx, 'geometry']

        for _, match in matches.iterrows():
            sidewalk_geom = match['geometry']

            # Simple heuristic: check which side of the street the sidewalk is on
            # Use the midpoint of the sidewalk and check its position relative to the street
            if isinstance(sidewalk_geom, LineString):
                midpoint = sidewalk_geom.interpolate(0.5, normalized=True)

                # Determine side based on bearing (simplified)
                # Note: Proper left/right determination would require geometric analysis
                # For now, assign to left side as default
                side = 'left'

                # Set geometry from government data
                edges.at[edge_idx, f'sidewalk_{side}_geometry'] = sidewalk_geom

                # Set public data ID using column mapping
                id_col = col_map.sidewalk_id if col_map.sidewalk_id else 'id'
                gov_id = match.get(id_col, match.get('ID', match.get('OBJECTID', None)))
                edges.at[edge_idx, f'public_data_id_sidewalk_{side}'] = str(gov_id) if gov_id is not None else None

                # Apply column mappings for attributes
                if col_map.sidewalk_surface and col_map.sidewalk_surface in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_surface'] = match[col_map.sidewalk_surface]
                if col_map.sidewalk_width and col_map.sidewalk_width in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_width'] = match[col_map.sidewalk_width]
                if col_map.sidewalk_incline and col_map.sidewalk_incline in match.index:
                    edges.at[edge_idx, f'sidewalk_{side}_incline'] = match[col_map.sidewalk_incline]

    return edges



def merge_government_bikelanes(
    edges: gpd.GeoDataFrame,
    gov_bikelanes: gpd.GeoDataFrame,
    city_config: CityConfig
) -> gpd.GeoDataFrame:
    """
    Merge government bikelane data with OSM edges.
    Feature 2: Load government bikelane data (Step 2 - partial)
    """
    print("  Merging government bikelane data...")
    
    col_map = city_config.column_mappings
    
    # Ensure same CRS
    if gov_bikelanes.crs != edges.crs:
        gov_bikelanes = gov_bikelanes.to_crs(edges.crs)
    
    # Spatial join to match government bikelanes to OSM edges
    # Optimize: Create temporary GeoDataFrame with only geometry and index (avoid full copy)
    edges_buffered = gpd.GeoDataFrame(
        geometry=edges.geometry.buffer(10.0),  # 10m buffer in meters (UTM)
        index=edges.index,
        crs=edges.crs
    )
    
    joined = gpd.sjoin(gov_bikelanes, edges_buffered, how='inner', predicate='intersects')
    del edges_buffered  # Free memory immediately
    
    # Group by edge index to handle multiple bikelane matches
    for edge_idx in joined['index_right'].unique():
        matches = joined[joined['index_right'] == edge_idx]
        
        # Determine which side (left or right) based on geometry position
        edge_geom = edges.loc[edge_idx, 'geometry']
        
        for _, match in matches.iterrows():
            bikelane_geom = match['geometry']
            
            # Simple heuristic: check which side of the street the bikelane is on
            if isinstance(bikelane_geom, LineString):
                # Note: Proper side determination would require geometric analysis
                # For now, assign to left side and lane 1 as default
                side = 'left'
                lane_n = 1
                
                # Set geometry from government data
                edges.at[edge_idx, f'bikeway_{side}_{lane_n}_geometry'] = bikelane_geom
                
                # Set public data ID using column mapping
                id_col = col_map.bikelane_id if col_map.bikelane_id else 'id'
                gov_id = match.get(id_col, match.get('ID', match.get('OBJECTID', None)))
                edges.at[edge_idx, f'public_data_id_bikeway_{side}_{lane_n}'] = str(gov_id) if gov_id is not None else None
                
                # Apply column mappings for attributes
                if col_map.bikelane_type and col_map.bikelane_type in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_type'] = match[col_map.bikelane_type]
                if col_map.bikelane_surface and col_map.bikelane_surface in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_surface'] = match[col_map.bikelane_surface]
                if col_map.bikelane_width and col_map.bikelane_width in match.index:
                    edges.at[edge_idx, f'bikeway_{side}_{lane_n}_width'] = match[col_map.bikelane_width]
    
    return edges


def populate_street_features(
    edges: gpd.GeoDataFrame,
    street_features: gpd.GeoDataFrame,
    city_config: CityConfig
) -> gpd.GeoDataFrame:
    """
    Populate street feature geometry columns.
    Feature 8: Load and populate street feature geometry
    """
    print("  Populating street features...")
    
    # Ensure same CRS
    if street_features.crs != edges.crs:
        street_features = street_features.to_crs(edges.crs)
    
    # Initialize feature columns as empty lists
    edges['street_feature_types'] = [[] for _ in range(len(edges))]
    edges['public_data_id_street_feature'] = [[] for _ in range(len(edges))]
    edges['street_feature_geometry'] = [[] for _ in range(len(edges))]
    edges['street_feature_geometry_projected'] = [[] for _ in range(len(edges))]
    
    # Check if edges is empty
    if len(edges) == 0:
        print("  Warning: No edges to merge street features with")
        return edges
    
    col_map = city_config.column_mappings
    
    # Spatial join to find nearest street segment for each feature
    for _, feature in street_features.iterrows():
        feature_point = feature['geometry']
        
        # Find nearest street segment
        distances = edges.geometry.distance(feature_point)
        if len(distances) == 0:
            continue
        nearest_idx = distances.idxmin()
        
        if distances[nearest_idx] > 20.0:  # 20m threshold (working CRS is UTM)
            continue
        
        # Get feature type using column mapping
        feature_type_col = col_map.feature_type if col_map.feature_type else 'feature_type'
        feature_type = feature.get(feature_type_col, feature.get('type', feature.get('amenity', 'unknown')))
        
        # Get government ID using column mapping
        feature_id_col = col_map.feature_id if col_map.feature_id else 'id'
        gov_id = feature.get(feature_id_col, feature.get('ID', feature.get('OBJECTID', None)))
        
        # Project feature onto street segment
        street_geom = edges.loc[nearest_idx, 'geometry']
        projected_point = street_geom.interpolate(street_geom.project(feature_point))
        
        # Append to lists
        edges.at[nearest_idx, 'street_feature_types'].append(str(feature_type))
        edges.at[nearest_idx, 'public_data_id_street_feature'].append(str(gov_id) if gov_id is not None else None)
        edges.at[nearest_idx, 'street_feature_geometry'].append(feature_point)
        edges.at[nearest_idx, 'street_feature_geometry_projected'].append(projected_point)
    
    # Convert lists to MultiPoint geometries
    for idx in edges.index:
        feature_geoms = edges.at[idx, 'street_feature_geometry']
        projected_geoms = edges.at[idx, 'street_feature_geometry_projected']
        
        if len(feature_geoms) > 0:
            edges.at[idx, 'street_feature_geometry'] = MultiPoint(feature_geoms)
            edges.at[idx, 'street_feature_geometry_projected'] = MultiPoint(projected_geoms)
        else:
            edges.at[idx, 'street_feature_geometry'] = MultiPoint([])
            edges.at[idx, 'street_feature_geometry_projected'] = MultiPoint([])
    
    return edges


def populate_sidewalk_features(
    edges: gpd.GeoDataFrame,
    sidewalk_features: gpd.GeoDataFrame,
    city_config: CityConfig,
    sidewalk_feature_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Populate sidewalk feature geometry columns.
    Feature 9: Load and populate sidewalk feature geometry
    """
    print("  Populating sidewalk features...")
    
    # Ensure same CRS
    if sidewalk_features.crs != edges.crs:
        sidewalk_features = sidewalk_features.to_crs(edges.crs)
    
    # Initialize feature columns as empty lists for both sides
    for side in ['left', 'right']:
        edges[f'sidewalk_{side}_feature_ids'] = [[] for _ in range(len(edges))]
        edges[f'sidewalk_{side}_feature_types'] = [[] for _ in range(len(edges))]
        edges[f'public_data_id_sidewalk_{side}_feature'] = [[] for _ in range(len(edges))]
        edges[f'sidewalk_{side}_feature_geometry'] = [[] for _ in range(len(edges))]
        edges[f'sidewalk_{side}_feature_geometry_projected'] = [[] for _ in range(len(edges))]
    
    # Check if edges is empty
    if len(edges) == 0:
        print("  Warning: No edges to merge sidewalk features with")
        return edges
    
    # Spatial join to find nearest sidewalk segment for each feature
    for _, feature in sidewalk_features.iterrows():
        feature_point = feature['geometry']
        
        # Find nearest edge with sidewalk
        min_distance = float('inf')
        nearest_idx = None
        nearest_side = None
        
        for idx in edges.index:
            for side in ['left', 'right']:
                sidewalk_geom = edges.loc[idx, f'sidewalk_{side}_geometry']
                if sidewalk_geom is not None and isinstance(sidewalk_geom, LineString):
                    distance = sidewalk_geom.distance(feature_point)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_idx = idx
                        nearest_side = side
        
        if nearest_idx is None or min_distance > 10.0:  # 10m threshold (UTM meters)
            continue
        
        # Get feature type
        feature_type = feature.get('type', feature.get('amenity', feature.get('feature_type', 'unknown')))
        
        # Get government ID
        gov_id = feature.get('id', feature.get('ID', feature.get('OBJECTID', None)))
        
        # Assign sequential feature ID
        feature_id = sidewalk_feature_counter.next()
        
        # Project feature onto sidewalk segment
        sidewalk_geom = edges.loc[nearest_idx, f'sidewalk_{nearest_side}_geometry']
        projected_point = sidewalk_geom.interpolate(sidewalk_geom.project(feature_point))
        
        # Append to lists
        edges.at[nearest_idx, f'sidewalk_{nearest_side}_feature_ids'].append(str(feature_id))
        edges.at[nearest_idx, f'sidewalk_{nearest_side}_feature_types'].append(str(feature_type))
        edges.at[nearest_idx, f'public_data_id_sidewalk_{nearest_side}_feature'].append(str(gov_id) if gov_id is not None else None)
        edges.at[nearest_idx, f'sidewalk_{nearest_side}_feature_geometry'].append(feature_point)
        edges.at[nearest_idx, f'sidewalk_{nearest_side}_feature_geometry_projected'].append(projected_point)
    
    # Convert lists to MultiPoint geometries
    for idx in edges.index:
        for side in ['left', 'right']:
            feature_geoms = edges.at[idx, f'sidewalk_{side}_feature_geometry']
            projected_geoms = edges.at[idx, f'sidewalk_{side}_feature_geometry_projected']
            
            if len(feature_geoms) > 0:
                edges.at[idx, f'sidewalk_{side}_feature_geometry'] = MultiPoint(feature_geoms)
                edges.at[idx, f'sidewalk_{side}_feature_geometry_projected'] = MultiPoint(projected_geoms)
            else:
                edges.at[idx, f'sidewalk_{side}_feature_geometry'] = MultiPoint([])
                edges.at[idx, f'sidewalk_{side}_feature_geometry_projected'] = MultiPoint([])
    
    return edges


def populate_bikeway_features(
    edges: gpd.GeoDataFrame,
    bikeway_features: gpd.GeoDataFrame,
    city_config: CityConfig
) -> gpd.GeoDataFrame:
    """
    Populate bikeway feature geometry columns.
    Feature 10: Load and populate bikeway feature geometry
    """
    print("  Populating bikeway features...")
    
    # Ensure same CRS
    if bikeway_features.crs != edges.crs:
        bikeway_features = bikeway_features.to_crs(edges.crs)
    
    # Initialize feature columns as empty lists for all combinations
    for side in ['left', 'right']:
        for n in [1, 2]:
            edges[f'bikeway_{side}_{n}_feature_ids'] = [[] for _ in range(len(edges))]
            edges[f'bikeway_{side}_{n}_feature_types'] = [[] for _ in range(len(edges))]
            edges[f'public_data_id_bikeway_{side}_{n}_features'] = [[] for _ in range(len(edges))]
            edges[f'bikeway_{side}_{n}_feature_geometry'] = [[] for _ in range(len(edges))]
            edges[f'bikeway_{side}_{n}_feature_geometry_projected'] = [[] for _ in range(len(edges))]
    
    # Check if edges is empty
    if len(edges) == 0:
        print("  Warning: No edges to merge bikeway features with")
        return edges
    
    # Spatial join to find nearest bikeway segment for each feature
    for _, feature in bikeway_features.iterrows():
        feature_point = feature['geometry']
        
        # Find nearest edge with bikeway
        min_distance = float('inf')
        nearest_idx = None
        nearest_side = None
        nearest_lane = None
        
        for idx in edges.index:
            for side in ['left', 'right']:
                for n in [1, 2]:
                    bikeway_geom = edges.loc[idx, f'bikeway_{side}_{n}_geometry']
                    if bikeway_geom is not None and isinstance(bikeway_geom, LineString):
                        distance = bikeway_geom.distance(feature_point)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_idx = idx
                            nearest_side = side
                            nearest_lane = n
        
        if nearest_idx is None or min_distance > 10.0:  # 10m threshold (UTM meters)
            continue
        
        # Get feature type
        feature_type = feature.get('type', feature.get('amenity', feature.get('feature_type', 'unknown')))
        
        # Get government ID
        gov_id = feature.get('id', feature.get('ID', feature.get('OBJECTID', None)))
        
        # Note: No sequential ID for bikeway features per spec
        
        # Project feature onto bikeway segment
        bikeway_geom = edges.loc[nearest_idx, f'bikeway_{nearest_side}_{nearest_lane}_geometry']
        projected_point = bikeway_geom.interpolate(bikeway_geom.project(feature_point))
        
        # Append to lists
        edges.at[nearest_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_ids'].append('')  # Empty per spec
        edges.at[nearest_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_types'].append(str(feature_type))
        edges.at[nearest_idx, f'public_data_id_bikeway_{nearest_side}_{nearest_lane}_features'].append(str(gov_id) if gov_id is not None else None)
        edges.at[nearest_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_geometry'].append(feature_point)
        edges.at[nearest_idx, f'bikeway_{nearest_side}_{nearest_lane}_feature_geometry_projected'].append(projected_point)
    
    # Convert lists to MultiPoint geometries
    for idx in edges.index:
        for side in ['left', 'right']:
            for n in [1, 2]:
                feature_geoms = edges.at[idx, f'bikeway_{side}_{n}_feature_geometry']
                projected_geoms = edges.at[idx, f'bikeway_{side}_{n}_feature_geometry_projected']
                
                if len(feature_geoms) > 0:
                    edges.at[idx, f'bikeway_{side}_{n}_feature_geometry'] = MultiPoint(feature_geoms)
                    edges.at[idx, f'bikeway_{side}_{n}_feature_geometry_projected'] = MultiPoint(projected_geoms)
                else:
                    edges.at[idx, f'bikeway_{side}_{n}_feature_geometry'] = MultiPoint([])
                    edges.at[idx, f'bikeway_{side}_{n}_feature_geometry_projected'] = MultiPoint([])
    
    return edges


# ============================================================================
# SECTION 8: STAGE 4 — OFFSET GEOMETRY GENERATION (Steps 3, 5, 6, adapted)
# ============================================================================

def load_sanity_buffer(sanity_path: str) -> Dict[int, float]:
    """Load sanity buffer map from parquet file."""
    df = pd.read_parquet(sanity_path)
    return dict(zip(df['osmid'], df['max_offset_width']))


def compute_offset_distance(
    edge_row,
    side: str,
    facility_type: str,
    sanity_buffer: Dict[int, float]
) -> float:
    """
    Compute offset distance for sidewalk or bikeway.
    
    Args:
        edge_row: DataFrame row with edge data
        side: 'left' or 'right'
        facility_type: 'sidewalk' or 'bikeway'
        sanity_buffer: Dict mapping osmid to max_offset_width
    
    Returns:
        Offset distance in meters
    """
    osmid = edge_row['osmid']
    highway = normalize_tag(edge_row.get('highway', 'residential'))
    
    # Get lane width
    lane_width = resolve_lane_width(edge_row, highway)
    
    # Base offset: half the street width
    base_offset = lane_width / 2.0
    
    # Add bikeway width(s) on the same side so the sidewalk clears both lanes.
    # Use the already-extracted bikeway_{side}_{n}_width columns (populated by
    # extract_cycleway_tags and potentially overridden by government data) rather
    # than re-reading raw OSM cycleway: tags, which would ignore government values.
    bikeway_width = 0.0
    if facility_type == 'sidewalk':
        for lane_n in (1, 2):
            bw_type = edge_row.get(f'bikeway_{side}_{lane_n}_type')
            if bw_type is None:
                continue
            w = 1.5  # default bikeway width (m)
            w_val = edge_row.get(f'bikeway_{side}_{lane_n}_width')
            if w_val is not None:
                try:
                    w = float(w_val)
                except (ValueError, TypeError):
                    pass
            bikeway_width += w
    
    # Total offset
    offset = base_offset + bikeway_width
    
    # Apply sanity buffer constraint
    if osmid in sanity_buffer:
        max_offset = sanity_buffer[osmid]
        offset = min(offset, max_offset)
    
    return max(offset, 0.5)  # Minimum 0.5m offset


def generate_offset_geometry(
    line_geom: LineString,
    offset_distance: float,
    side: str
) -> Optional[LineString]:
    """
    Generate parallel offset geometry for a LineString.
    
    Args:
        line_geom: Original LineString geometry
        offset_distance: Distance to offset (meters, always positive)
        side: 'left' or 'right'
    
    Returns:
        Offset LineString or None if failed
    """
    if line_geom is None or line_geom.is_empty:
        return None
    
    try:
        # Use offset_curve: positive for left, negative for right
        if side == 'left':
            offset_geom = line_geom.offset_curve(
                offset_distance,
                quad_segs=16,
                join_style=2,  # mitre
                mitre_limit=5.0
            )
        else:  # right
            offset_geom = line_geom.offset_curve(
                -offset_distance,
                quad_segs=16,
                join_style=2,  # mitre
                mitre_limit=5.0
            )
        
        if offset_geom is None or offset_geom.is_empty:
            return None
        
        # Handle MultiLineString result
        if isinstance(offset_geom, MultiLineString):
            # Take longest segment
            offset_geom = max(offset_geom.geoms, key=lambda g: g.length)
        
        # Ensure correct direction (same as original)
        if isinstance(offset_geom, LineString):
            return offset_geom
        
        return None
    except Exception:
        return None


def _enrich_bikeway_chunk(chunk_data):
    """Worker function for parallel bikeway geometry enrichment."""
    chunk, sanity_buffer = chunk_data

    for idx in chunk.index:
        row = chunk.loc[idx]

        for side in ('left', 'right'):
            # Lane 1 — only generate offset if geometry not already assigned
            # (separate highway=cycleway ways already have geometry set)
            if row.get(f'bikeway_{side}_1_type') is not None and row.get(f'bikeway_{side}_1_geometry') is None:
                offset_dist = compute_offset_distance(row, side, 'bikeway', sanity_buffer)
                geom = generate_offset_geometry(row['geometry'], offset_dist, side)
                chunk.at[idx, f'bikeway_{side}_1_geometry'] = geom
                chunk.at[idx, f'bikeway_{side}_buffered'] = True

            # Lane 2 (cycleway:{side}:2) — stacked beyond lane 1.
            # Only generate when lane 2 has a type but no geometry yet.
            if row.get(f'bikeway_{side}_2_type') is not None and row.get(f'bikeway_{side}_2_geometry') is None:
                # Base offset to the outer edge of lane 1
                offset_lane1 = compute_offset_distance(row, side, 'bikeway', sanity_buffer)
                # Width of lane 1 (default 1.5 m)
                w1 = 1.5
                try:
                    w1_raw = row.get(f'bikeway_{side}_1_width')
                    if w1_raw is not None:
                        w1 = float(w1_raw)
                except (ValueError, TypeError):
                    pass
                offset_lane2 = offset_lane1 + w1
                # Respect sanity buffer
                osmid = row.get('osmid')
                if osmid in sanity_buffer:
                    offset_lane2 = min(offset_lane2, sanity_buffer[osmid])
                offset_lane2 = max(offset_lane2, 0.5)
                geom2 = generate_offset_geometry(row['geometry'], offset_lane2, side)
                chunk.at[idx, f'bikeway_{side}_2_geometry'] = geom2
                chunk.at[idx, f'bikeway_{side}_buffered'] = True

    return chunk


def enrich_bikeway_geometries(
    network: gpd.GeoDataFrame,
    sanity_buffer: Dict[int, float],
    n_jobs: int = None
) -> gpd.GeoDataFrame:
    """Generate offset bikeway geometries for buffered bikeways."""
    print("  Generating bikeway offset geometries...")
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, sanity_buffer) for chunk in chunks]
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_enrich_bikeway_chunk, args),
            total=len(args),
            desc="  Bikeway geometries"
        ))
    
    # Concatenate results and preserve as GeoDataFrame
    result_df = pd.concat(results, ignore_index=False)
    
    # Convert back to GeoDataFrame with original geometry column
    if isinstance(network, gpd.GeoDataFrame):
        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs=network.crs)
        return result_gdf
    
    return result_df


def _enrich_sidewalk_chunk(chunk_data):
    """Worker function for parallel sidewalk geometry enrichment."""
    chunk, sanity_buffer = chunk_data
    
    for idx in chunk.index:
        row = chunk.loc[idx]
        
        # Left sidewalk - only generate offset if geometry not already assigned (from separate footway)
        if row.get('sidewalk_left_presence') is True and row.get('sidewalk_left_geometry') is None:
            offset_dist = compute_offset_distance(row, 'left', 'sidewalk', sanity_buffer)
            geom = generate_offset_geometry(row['geometry'], offset_dist, 'left')
            chunk.at[idx, 'sidewalk_left_geometry'] = geom
            chunk.at[idx, 'sidewalk_left_buffered'] = True
        
        # Right sidewalk - only generate offset if geometry not already assigned (from separate footway)
        if row.get('sidewalk_right_presence') is True and row.get('sidewalk_right_geometry') is None:
            offset_dist = compute_offset_distance(row, 'right', 'sidewalk', sanity_buffer)
            geom = generate_offset_geometry(row['geometry'], offset_dist, 'right')
            chunk.at[idx, 'sidewalk_right_geometry'] = geom
            chunk.at[idx, 'sidewalk_right_buffered'] = True
    
    return chunk


def enrich_sidewalk_geometries(
    network: gpd.GeoDataFrame,
    sanity_buffer: Dict[int, float],
    n_jobs: int = None
) -> gpd.GeoDataFrame:
    """Generate offset sidewalk geometries for buffered sidewalks."""
    print("  Generating sidewalk offset geometries...")
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    chunks = np.array_split(network, n_jobs)
    args = [(chunk, sanity_buffer) for chunk in chunks]
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_enrich_sidewalk_chunk, args),
            total=len(args),
            desc="  Sidewalk geometries"
        ))
    
    # Concatenate results and preserve as GeoDataFrame
    result_df = pd.concat(results, ignore_index=False)
    
    # Convert back to GeoDataFrame with original geometry column
    if isinstance(network, gpd.GeoDataFrame):
        result_gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs=network.crs)
        return result_gdf
    
    return result_df


# ============================================================================
# SECTION 9: STAGE 5 — VERTEX DEFLECTION SPLITTING (Step 5, new)
# ============================================================================

def _detect_deflection_worker(args):
    """Worker function for parallel deflection detection."""
    idx, row, threshold = args
    
    geom = row['geometry']
    
    if geom is None or geom.is_empty or not isinstance(geom, LineString):
        return idx, []
    
    coords = list(geom.coords)
    if len(coords) < 3:
        return idx, []
    
    # Check interior vertices for deflection
    split_indices = []
    
    for i in range(1, len(coords) - 1):
        p_prev = np.array(coords[i-1])
        p_curr = np.array(coords[i])
        p_next = np.array(coords[i+1])
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-9 or v2_norm < 1e-9:
            continue
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        deflection = 180.0 - angle_deg
        
        if deflection > threshold:
            split_indices.append(i)
    
    return idx, split_indices


def detect_and_split_deflections(
    edges: gpd.GeoDataFrame,
    split_node_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Detect and split edges at vertices with deflection > threshold.
    Parallelized for performance.
    
    Args:
        edges: GeoDataFrame of edges
        split_node_counter: Counter for negative split node IDs
    
    Returns:
        GeoDataFrame with split edges
    """
    print("  Detecting and splitting vertex deflections...")
    
    # Parallel deflection detection - keep using iterrows for worker compatibility
    n_jobs = max(1, mp.cpu_count() - 1)
    args = [(idx, row, DEFLECTION_THRESHOLD_DEG) for idx, row in edges.iterrows()]
    
    with mp.Pool(n_jobs) as pool:
        deflection_results = list(tqdm(
            pool.imap(_detect_deflection_worker, args),
            total=len(args),
            desc="  Detecting deflections"
        ))
    
    # Build map of edges that need splitting
    edges_to_split = {idx: split_indices for idx, split_indices in deflection_results if split_indices}
    
    if not edges_to_split:
        print(f"  No deflections detected")
        return edges
    
    # Track split vertices to reuse IDs
    coord_to_split_node: Dict[Tuple[float, float], int] = {}
    new_edges = []
    
    # Optimized: iterate by index instead of iterrows
    for idx in tqdm(edges.index, total=len(edges), desc="  Splitting edges"):
        row = edges.loc[idx]
        
        if idx not in edges_to_split:
            new_edges.append(row)
            continue
        
        geom = row['geometry']
        coords = list(geom.coords)
        split_indices = edges_to_split[idx]
        
        # Split the LineString at detected vertices
        segments = []
        start_idx = 0
        
        for split_idx in split_indices:
            seg_coords = coords[start_idx:split_idx+1]
            if len(seg_coords) >= 2:
                segments.append((seg_coords, split_idx))
            start_idx = split_idx
        
        # Final segment
        seg_coords = coords[start_idx:]
        if len(seg_coords) >= 2:
            segments.append((seg_coords, None))
        
        # Create new edge rows for each segment
        for seg_idx, (seg_coords, split_at) in enumerate(segments):
            new_row = row.copy()
            new_row['geometry'] = LineString(seg_coords)
            
            # Update start/end node IDs
            if seg_idx == 0:
                pass  # Keep original start_node_osmid
            else:
                prev_split_coord = tuple(seg_coords[0])
                if prev_split_coord not in coord_to_split_node:
                    coord_to_split_node[prev_split_coord] = split_node_counter.next_negative()
                new_row['start_node_osmid'] = coord_to_split_node[prev_split_coord]
            
            if seg_idx == len(segments) - 1:
                pass  # Keep original end_node_osmid
            else:
                split_coord = tuple(seg_coords[-1])
                if split_coord not in coord_to_split_node:
                    coord_to_split_node[split_coord] = split_node_counter.next_negative()
                new_row['end_node_osmid'] = coord_to_split_node[split_coord]
            
            new_edges.append(new_row)
    
    result = gpd.GeoDataFrame(new_edges, crs=edges.crs)
    result = result.reset_index(drop=True)
    print(f"  Split {len(result) - len(edges)} segments due to deflection")
    return result


# ============================================================================
# SECTION 10: STAGE 6 — BEARING NORMALIZATION AND TAG EXTRACTION (Step 6, adapted)
# ============================================================================

def normalize_left_right_tags(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize left/right tags based on bearing and compute normalized_bearing.

    normalized_bearing is folded to [0, 180) so that opposite-digitized segments
    on the same block side share the same undirected axis value. Use 'bearing'
    (full 0-360) for directional routing and turn computation.
    Vectorized for performance.
    """
    print("  Normalizing bearings and left/right tags...")

    # Compute bearings
    network['bearing'] = compute_bearings_vectorized(network['geometry'])
    # Fold to [0, 180): collapses N/S and E/W to the same undirected axis
    network['normalized_bearing'] = network['bearing'] % 180

    # Vectorized swap: identify rows where bearing is between 180 and 360
    swap_mask = (network['bearing'] >= 180) & (network['bearing'] < 360)
    
    if swap_mask.any():
        # Swap left and right tags for rows matching the mask
        for tag_base in ['sidewalk', 'cycleway']:
            left_col = f'{tag_base}:left'
            right_col = f'{tag_base}:right'
            
            if left_col in network.columns and right_col in network.columns:
                # Store original values
                left_vals = network[left_col].copy()
                right_vals = network[right_col].copy()
                
                # Swap where mask is True
                network.loc[swap_mask, left_col] = right_vals[swap_mask]
                network.loc[swap_mask, right_col] = left_vals[swap_mask]
    
    return network


def _recompute_bearing_column(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Recompute bearing and normalized_bearing after deflection splitting.

    Called in Stage 9 instead of normalize_left_right_tags so that bearing
    columns reflect the (potentially different) headings of split sub-segments
    WITHOUT re-swapping raw OSM tags or extracted facility columns — the swap
    already occurred before tag extraction in _stage_tag_extraction.

    normalized_bearing is folded to [0, 180) (undirected axis). Use 'bearing'
    for directional routing and turn computation.
    """
    network['bearing'] = compute_bearings_vectorized(network['geometry'])
    network['normalized_bearing'] = network['bearing'] % 180
    return network


def extract_sidewalk_tags(
    network: gpd.GeoDataFrame,
    sidewalk_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Extract sidewalk tags from OSM data and assign sequential IDs.
    """
    print("  Extracting sidewalk tags...")
    
    # Initialize columns
    for side in ['left', 'right']:
        network[f'sidewalk_{side}_ID'] = None
        network[f'sidewalk_{side}_block_ID'] = None
        network[f'sidewalk_{side}_presence'] = False
        network[f'public_data_id_sidewalk_{side}'] = None
        network[f'sidewalk_{side}_surface'] = None
        network[f'sidewalk_{side}_quality'] = None
        network[f'sidewalk_{side}_width'] = None
        network[f'sidewalk_{side}_incline'] = None
        network[f'sidewalk_{side}_buffered'] = False
        network[f'sidewalk_{side}_geometry'] = None
        
        # Feature columns - Feature 11: Initialize empty feature columns
        network[f'sidewalk_{side}_feature_ids'] = [[] for _ in range(len(network))]
        network[f'sidewalk_{side}_feature_types'] = [[] for _ in range(len(network))]
        network[f'public_data_id_sidewalk_{side}_feature'] = [[] for _ in range(len(network))]
        network[f'sidewalk_{side}_feature_geometry'] = [MultiPoint([]) for _ in range(len(network))]
        network[f'sidewalk_{side}_feature_geometry_projected'] = [MultiPoint([]) for _ in range(len(network))]
    
    for idx, row in network.iterrows():
        highway = row.get('highway')
        highway_val = normalize_tag(highway)
        
        # Handle separate footways (highway=footway, path, pedestrian, steps)
        if highway_val in ['footway', 'path', 'pedestrian', 'steps']:
            # Check if this is explicitly a sidewalk (footway=sidewalk tag)
            footway_tag = row.get('footway')
            footway_val = normalize_tag(footway_tag)
            
            # Skip crossing footways - they should only be in crossings_cache, not treated as sidewalks
            if footway_val == 'crossing':
                continue  # Skip - crossings are handled separately in extract_crosswalk_tags
            
            if footway_val == 'sidewalk':
                # This is a separate sidewalk - assign to both sides since it's the street itself
                # Per schema: separate sidewalks should have sidewalk presence
                for side in ['left', 'right']:
                    network.at[idx, f'sidewalk_{side}_presence'] = True
                    network.at[idx, f'sidewalk_{side}_ID'] = sidewalk_counter.next()
                    network.at[idx, f'sidewalk_{side}_buffered'] = False  # Separate, not buffered
                    
                    # Assign the street geometry as the sidewalk geometry
                    network.at[idx, f'sidewalk_{side}_geometry'] = row['geometry']
                    
                    # Extract attributes
                    surface = normalize_tag(row.get('surface'))
                    network.at[idx, f'sidewalk_{side}_surface'] = surface
                    
                    smoothness = normalize_tag(row.get('smoothness'))
                    network.at[idx, f'sidewalk_{side}_quality'] = smoothness
                    
                    width = normalize_tag(row.get('width'))
                    network.at[idx, f'sidewalk_{side}_width'] = width
                    
                    incline = normalize_tag(row.get('incline'))
                    network.at[idx, f'sidewalk_{side}_incline'] = incline
            
            # For other footways (paths, pedestrian areas, etc.), they're just pedestrian ways without sidewalk assignment
            continue  # Skip normal sidewalk tag processing for footways
        
        # Normal sidewalk tag processing for carways.
        # Tag priority per OSMNetworkDescription Phase 1 sidewalk format:
        #   sidewalk:{side}  >  sidewalk:both  >  sidewalk
        # Values: yes/no/separate/seperate (misspelling seen in OSM data)/none/left/right/both
        for side in ['left', 'right']:
            sidewalk_tag = row.get(
                f'sidewalk:{side}',
                row.get('sidewalk:both', row.get('sidewalk', None))
            )
            sidewalk_val = normalize_tag(sidewalk_tag)

            presence = False
            if sidewalk_val in ['yes', 'both', 'separate', 'seperate']:
                presence = True
            elif sidewalk_val == side:
                presence = True
            
            network.at[idx, f'sidewalk_{side}_presence'] = presence
            
            if presence:
                # Assign sequential ID
                network.at[idx, f'sidewalk_{side}_ID'] = sidewalk_counter.next()
                
                # Extract attributes
                surface = normalize_tag(row.get(f'sidewalk:{side}:surface', row.get('sidewalk:surface', None)))
                network.at[idx, f'sidewalk_{side}_surface'] = surface
                
                width = normalize_tag(row.get(f'sidewalk:{side}:width', row.get('sidewalk:width', None)))
                network.at[idx, f'sidewalk_{side}_width'] = width
                
                incline = normalize_tag(row.get(f'sidewalk:{side}:incline', row.get('sidewalk:incline', None)))
                network.at[idx, f'sidewalk_{side}_incline'] = incline
    
    return network


def extract_cycleway_tags(
    network: gpd.GeoDataFrame,
    bikeway_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Extract cycleway tags from OSM data and assign sequential IDs.
    """
    print("  Extracting cycleway tags...")
    
    # Initialize columns
    # oneway_bicycle: street-level tag indicating whether bicycle travel is
    # restricted to one direction (oneway:bicycle=yes/no/-1).  A value of 'no'
    # on an otherwise one-way street means cyclists may travel both ways.
    network['oneway_bicycle'] = None

    for side in ['left', 'right']:
        for n in [1, 2]:
            network[f'bikeway_{side}_{n}_id'] = None
            network[f'bikeway_{side}_{n}_block_id'] = None
            network[f'public_data_id_bikeway_{side}_{n}'] = None
            network[f'bikeway_{side}_{n}_type'] = None
            network[f'bikeway_{side}_{n}_surface'] = None
            network[f'bikeway_{side}_{n}_quality'] = None
            network[f'bikeway_{side}_{n}_permitted'] = None
            network[f'bikeway_{side}_{n}_width'] = None
            network[f'bikeway_{side}_{n}_incline'] = None
            network[f'bikeway_{side}_{n}_geometry'] = None

            # Feature columns - Feature 11: Initialize empty feature columns
            network[f'bikeway_{side}_{n}_feature_ids'] = [[] for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_types'] = [[] for _ in range(len(network))]
            network[f'public_data_id_bikeway_{side}_{n}_features'] = [[] for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_geometry'] = [MultiPoint([]) for _ in range(len(network))]
            network[f'bikeway_{side}_{n}_feature_geometry_projected'] = [MultiPoint([]) for _ in range(len(network))]

        network[f'bikeway_{side}_buffered'] = False
    
    for idx, row in network.iterrows():
        highway = row.get('highway')
        highway_val = normalize_tag(highway)
        
        # Handle separate cycleways (highway=cycleway)
        if highway_val == 'cycleway':
            # Separate cycleways are bike facilities — assign to both sides since it's the street itself
            for side in ['left', 'right']:
                network.at[idx, f'bikeway_{side}_1_id'] = bikeway_counter.next()
                network.at[idx, f'bikeway_{side}_1_type'] = 'track'  # Separate cycleways are typically tracks
                network.at[idx, f'bikeway_{side}_buffered'] = False  # Separate, not buffered

                # Assign the street geometry as the bikeway geometry
                network.at[idx, f'bikeway_{side}_1_geometry'] = row['geometry']

                # Extract attributes
                surface = normalize_tag(row.get('surface'))
                network.at[idx, f'bikeway_{side}_1_surface'] = surface

                smoothness = normalize_tag(row.get('smoothness'))
                network.at[idx, f'bikeway_{side}_1_quality'] = smoothness

                width = normalize_tag(row.get('width'))
                network.at[idx, f'bikeway_{side}_1_width'] = width

                incline = normalize_tag(row.get('incline'))
                network.at[idx, f'bikeway_{side}_1_incline'] = incline

                permitted = normalize_tag(row.get('bicycle', None))
                network.at[idx, f'bikeway_{side}_1_permitted'] = permitted

            # oneway:bicycle on a separate cycleway way (falls back to oneway)
            oneway_bicycle = normalize_tag(
                row.get('oneway:bicycle', row.get('oneway', None))
            )
            network.at[idx, 'oneway_bicycle'] = oneway_bicycle

            continue  # Skip normal cycleway tag processing for cycleways

        # Normal cycleway tag processing for carways.
        # Extract oneway:bicycle once per street segment (street-level attribute).
        oneway_bicycle = normalize_tag(row.get('oneway:bicycle', None))
        network.at[idx, 'oneway_bicycle'] = oneway_bicycle

        for side in ['left', 'right']:
            # Check cycleway presence
            cycleway_tag = row.get(f'cycleway:{side}', row.get('cycleway', None))
            cycleway_val = normalize_tag(cycleway_tag)

            if cycleway_val and cycleway_val not in ['no', 'none']:
                # Assign sequential ID for bikeway_1
                network.at[idx, f'bikeway_{side}_1_id'] = bikeway_counter.next()
                network.at[idx, f'bikeway_{side}_1_type'] = cycleway_val

                # Extract attributes
                surface = normalize_tag(row.get(f'cycleway:{side}:surface', row.get('cycleway:surface', None)))
                network.at[idx, f'bikeway_{side}_1_surface'] = surface

                width = normalize_tag(row.get(f'cycleway:{side}:width', row.get('cycleway:width', None)))
                network.at[idx, f'bikeway_{side}_1_width'] = width

                incline = normalize_tag(row.get(f'cycleway:{side}:incline', row.get('cycleway:incline', None)))
                network.at[idx, f'bikeway_{side}_1_incline'] = incline

                permitted = normalize_tag(row.get('bicycle', None))
                network.at[idx, f'bikeway_{side}_1_permitted'] = permitted

            # Check for cycleway:left:2 or cycleway:right:2
            cycleway_2_tag = row.get(f'cycleway:{side}:2', None)
            cycleway_2_val = normalize_tag(cycleway_2_tag)

            if cycleway_2_val and cycleway_2_val not in ['no', 'none']:
                network.at[idx, f'bikeway_{side}_2_id'] = bikeway_counter.next()
                network.at[idx, f'bikeway_{side}_2_type'] = cycleway_2_val

    return network


# ============================================================================
# SECTION 11: STAGE 7 — BLOCK STRUCTURE WITH SHOELACE FORMULA (Step 7, new)
# ============================================================================

def extract_nodes_from_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract nodes GeoDataFrame from edges.
    Reconstructs nodes from start/end points of edge geometries.
    
    Returns:
        GeoDataFrame with columns: osmid, x, y, geometry
    """
    nodes_dict = {}
    
    for idx, row in edges.iterrows():
        start_id = row['start_node_osmid']
        end_id = row['end_node_osmid']
        geom = row['geometry']
        
        if geom and hasattr(geom, 'coords'):
            # Start node
            if start_id not in nodes_dict:
                coords = list(geom.coords)
                nodes_dict[start_id] = {
                    'osmid': start_id,
                    'x': coords[0][0],
                    'y': coords[0][1],
                    'geometry': Point(coords[0])
                }
            
            # End node
            if end_id not in nodes_dict:
                coords = list(geom.coords)
                nodes_dict[end_id] = {
                    'osmid': end_id,
                    'x': coords[-1][0],
                    'y': coords[-1][1],
                    'geometry': Point(coords[-1])
                }
    
    return gpd.GeoDataFrame(list(nodes_dict.values()), crs=edges.crs)


def build_adjacency_graph(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame
) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Build adjacency graph from edges (vectorized).
    
    Returns:
        Dict mapping node_id -> [(edge_idx, other_node_id, outgoing_bearing), ...]
    """
    print("  Building adjacency graph...")
    
    adj = defaultdict(list)
    
    # Vectorized approach: process all edges at once
    start_nodes = edges['start_node_osmid'].values
    end_nodes = edges['end_node_osmid'].values
    bearings = edges['bearing'].values
    indices = edges.index.values
    
    # Build adjacency lists
    for i, (start_node, end_node, bearing, idx) in enumerate(zip(start_nodes, end_nodes, bearings, indices)):
        # Forward direction: start -> end
        adj[start_node].append((idx, end_node, bearing))
        
        # Reverse direction: end -> start (bearing + 180)
        reverse_bearing = (bearing + 180) % 360
        adj[end_node].append((idx, start_node, reverse_bearing))
    
    return dict(adj)


def create_grid_bounding_box(edges: gpd.GeoDataFrame, grid_size: float = 0.01) -> Tuple[List[Polygon], Dict]:
    """
    Create a square grid bounding box around the city's boundary.
    Grid cells are numbered with x-axis increasing left to right, y-axis increasing south to north.
    
    Args:
        edges: GeoDataFrame of street edges
        grid_size: Size of each grid cell in degrees (default 0.01 ~ 1km)
    
    Returns:
        Tuple of (grid_cells, grid_metadata)
        - grid_cells: List of Polygon geometries for each grid cell
        - grid_metadata: Dict with 'bounds', 'n_cols', 'n_rows', 'grid_size'
    """
    print("  Creating grid bounding box...")
    
    # Get bounds of all edges
    bounds = edges.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    n_cols = int(np.ceil((maxx - minx) / grid_size))
    n_rows = int(np.ceil((maxy - miny) / grid_size))
    
    print(f"  Grid: {n_cols} columns x {n_rows} rows ({n_cols * n_rows} cells)")
    
    # Create grid cells
    grid_cells = []
    for col in range(n_cols):
        for row in range(n_rows):
            x_min = minx + col * grid_size
            y_min = miny + row * grid_size
            x_max = x_min + grid_size
            y_max = y_min + grid_size
            
            cell = shapely_box(x_min, y_min, x_max, y_max)
            grid_cells.append((col, row, cell))
    
    grid_metadata = {
        'bounds': bounds,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'grid_size': grid_size,
        'minx': minx,
        'miny': miny
    }
    
    return grid_cells, grid_metadata


def _preassign_edges_to_grid(
    edges: gpd.GeoDataFrame,
    grid_cells: List[Tuple[int, int, Polygon]],
) -> Dict[Tuple[int, int], List[int]]:
    """
    Pre-assign edges to grid cells using STRtree spatial index.
    Returns dict mapping (col, row) -> [edge_indices].
    """
    print("  Pre-assigning edges to grid cells via spatial index...")
    edge_geoms = list(edges.geometry)
    edge_indices = list(edges.index)
    tree = STRtree(edge_geoms)

    cell_edge_map: Dict[Tuple[int, int], List[int]] = {}
    for col, row, cell_geom in grid_cells:
        hits = tree.query(cell_geom)
        matched = [edge_indices[i] for i in hits if edge_geoms[i].intersects(cell_geom)]
        if matched:
            cell_edge_map[(col, row)] = matched
    return cell_edge_map


def _preassign_nodes_to_grid(
    node_coords: Dict[int, Tuple[float, float]],
    grid_metadata: Dict,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Assign nodes to grid cells using arithmetic on grid bounds.
    node_coords values are (lat, lon) — y, x order.
    Returns dict mapping (col, row) -> [node_ids].
    """
    minx = grid_metadata['minx']
    miny = grid_metadata['miny']
    gs = grid_metadata['grid_size']
    n_cols = grid_metadata['n_cols']
    n_rows = grid_metadata['n_rows']

    cell_node_map: Dict[Tuple[int, int], List[int]] = {}
    for nid, (lat, lon) in node_coords.items():
        col = int((lon - minx) / gs)
        row_idx = int((lat - miny) / gs)
        col = min(max(col, 0), n_cols - 1)
        row_idx = min(max(row_idx, 0), n_rows - 1)
        cell_node_map.setdefault((col, row_idx), []).append(nid)
    return cell_node_map



def _detect_blocks_worker(args):
    """
    Phase A worker for parallel block detection in one grid cell.
    Operates on plain dicts/lists (no Shapely or GeoDataFrame serialization).

    Traversals that close within the cell are stored as complete blocks.
    Traversals that exit the cell (reach a node not in cell_node_ids_set)
    are returned as open chains for serial stitching in Phase B.

    Returns:
        (grid_col, grid_row, block_map_for_cell, visited_edge_keys, open_chains)
        open_chains: list of dicts with traversal state needed to resume:
            {
                'start_node': int,
                'start_edge_idx': int,
                'cycle': List[int],           # edge indices traversed so far
                'visited': Set of edge_keys,  # directed edges visited in this chain
                'current_node': int,           # node where traversal exited the cell
                'current_edge_idx': int,       # last edge traversed
                'incoming_bearing': float,     # bearing arriving at current_node
                'prev_node': int,              # node before current_node
                'next_node': int,              # next node to visit (outside cell)
            }
    """
    grid_col, grid_row, cell_node_ids, adj, node_coords = args

    cell_node_ids_set = set(cell_node_ids)
    visited_edges: Set = set()
    block_map: Dict[str, List[int]] = {}
    open_chains = []
    block_sequence = 1

    # Sort nodes south-to-north within cell
    sorted_nodes = sorted(cell_node_ids, key=lambda n: node_coords[n])

    for start_node in sorted_nodes:
        if start_node not in adj:
            continue

        for start_edge_idx, next_node, start_bearing in adj[start_node]:
            edge_key = (start_edge_idx, start_node, next_node)
            if edge_key in visited_edges:
                continue

            cycle = []
            chain_visited = set()
            current_node = start_node
            current_edge_idx = start_edge_idx
            incoming_bearing = start_bearing
            prev_node = None
            _next_node = next_node

            max_steps = 1000
            steps = 0
            exited_cell = False

            while steps < max_steps:
                cycle.append(current_edge_idx)
                ek = (current_edge_idx, current_node, _next_node)
                visited_edges.add(ek)
                chain_visited.add(ek)

                prev_node = current_node
                current_node = _next_node

                # Check if traversal has left the cell
                if current_node not in cell_node_ids_set:
                    # Save open chain for Phase B stitching
                    open_chains.append({
                        'start_node': start_node,
                        'start_edge_idx': start_edge_idx,
                        'cycle': list(cycle),
                        'visited': set(chain_visited),
                        'current_node': current_node,
                        'current_edge_idx': current_edge_idx,
                        'incoming_bearing': incoming_bearing,
                        'prev_node': prev_node,
                        'next_node': current_node,  # resumption point
                    })
                    exited_cell = True
                    break

                if current_node not in adj:
                    break

                candidates = [
                    (e_idx, other, ob)
                    for e_idx, other, ob in adj[current_node]
                    if not (e_idx == current_edge_idx and other == prev_node)
                ]
                if not candidates:
                    break

                turn_angles = sorted(
                    ((ob - incoming_bearing) % 360, e_idx, other, ob)
                    for e_idx, other, ob in candidates
                )
                _, next_edge_idx, next_node_id, next_bearing = turn_angles[0]

                if next_node_id == start_node and next_edge_idx == start_edge_idx:
                    block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                    block_map[block_id] = cycle
                    block_sequence += 1
                    break

                current_edge_idx = next_edge_idx
                _next_node = next_node_id
                incoming_bearing = next_bearing
                steps += 1
            else:
                # max_steps reached — treat as terminally bounded
                if cycle:
                    block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                    block_map[block_id] = cycle
                    block_sequence += 1

            # Terminally bounded blocks (dead-ends) that didn't exit the cell
            # and weren't stored by the cycle-close or max_steps branches.
            # The while loop's break-on-no-candidates and break-on-no-adj
            # paths land here. Only store if we have edges and didn't already
            # store this cycle above.
            if not exited_cell and cycle and steps < max_steps:
                already_stored = any(cycle == v for v in block_map.values())
                if not already_stored:
                    block_id = f"{grid_col}_{grid_row}_{block_sequence}"
                    block_map[block_id] = cycle
                    block_sequence += 1

    return grid_col, grid_row, block_map, visited_edges, open_chains




def _stitch_open_chains(
    open_chains: List[Dict],
    adj: Dict[int, List[Tuple[int, int, float]]],
    node_coords: Dict[int, Tuple[float, float]],
    global_visited: Set,
    grid_metadata: Dict,
) -> Dict[str, List[int]]:
    """
    Phase B: Serial boundary stitching.

    Takes open chains (traversals that exited their grid cell) and continues
    each one across the full adjacency graph until it either closes into a
    cycle or terminates at a dead-end.

    Block ownership is assigned to the grid cell containing the chain's
    start_node (the node where the traversal originally began in Phase A).

    Args:
        open_chains: list of chain state dicts from Phase A workers
        adj: full adjacency graph (node_id -> [(edge_idx, other_node, bearing)])
        node_coords: node_id -> (lat, lon)
        global_visited: set of directed edge keys already visited by Phase A
        grid_metadata: grid info for computing ownership cell from coordinates

    Returns:
        block_map: Dict mapping block_id -> [edge_ids] for stitched blocks
    """
    if not open_chains:
        return {}

    minx = grid_metadata['minx']
    miny = grid_metadata['miny']
    gs = grid_metadata['grid_size']
    n_cols = grid_metadata['n_cols']
    n_rows = grid_metadata['n_rows']

    def _node_to_cell(node_id):
        """Map a node to its owning grid cell."""
        lat, lon = node_coords[node_id]
        col = min(max(int((lon - minx) / gs), 0), n_cols - 1)
        row = min(max(int((lat - miny) / gs), 0), n_rows - 1)
        return col, row

    block_map: Dict[str, List[int]] = {}
    cell_seq_counters: Dict[Tuple[int, int], int] = {}

    # Track which directed edges have been consumed by stitching so we
    # don't produce duplicate blocks from chains that share edges.
    stitch_visited: Set = set(global_visited)

    for chain in open_chains:
        start_node = chain['start_node']
        start_edge_idx = chain['start_edge_idx']
        cycle = list(chain['cycle'])
        current_node = chain['current_node']
        current_edge_idx = chain['current_edge_idx']
        incoming_bearing = chain['incoming_bearing']
        prev_node = chain['prev_node']

        # Skip chains whose resumption edge was already consumed by a
        # previous stitch or by Phase A — avoids duplicate blocks.
        resume_ek = (current_edge_idx, prev_node, current_node)
        if resume_ek in stitch_visited:
            continue

        max_steps = 2000
        steps = 0

        while steps < max_steps:
            if current_node not in adj:
                # Dead-end terminal — store as terminally bounded block
                break

            candidates = [
                (e_idx, other, ob)
                for e_idx, other, ob in adj[current_node]
                if not (e_idx == current_edge_idx and other == prev_node)
            ]
            if not candidates:
                break

            turn_angles = sorted(
                ((ob - incoming_bearing) % 360, e_idx, other, ob)
                for e_idx, other, ob in candidates
            )
            _, next_edge_idx, next_node_id, next_bearing = turn_angles[0]

            # Check if this directed edge was already visited
            ek = (next_edge_idx, current_node, next_node_id)
            if ek in stitch_visited:
                # We've run into an already-traversed edge — can't continue
                break

            cycle.append(next_edge_idx)
            stitch_visited.add(ek)

            # Check if cycle closed
            if next_node_id == start_node and next_edge_idx == start_edge_idx:
                # Closed cycle — assign ownership to start_node's grid cell
                owner_col, owner_row = _node_to_cell(start_node)
                cell_key = (owner_col, owner_row)
                if cell_key not in cell_seq_counters:
                    cell_seq_counters[cell_key] = 1
                block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
                cell_seq_counters[cell_key] += 1
                block_map[block_id] = cycle
                break

            prev_node = current_node
            current_node = next_node_id
            current_edge_idx = next_edge_idx
            incoming_bearing = next_bearing
            steps += 1
        else:
            # max_steps exhausted — store as terminally bounded
            if cycle:
                owner_col, owner_row = _node_to_cell(start_node)
                cell_key = (owner_col, owner_row)
                if cell_key not in cell_seq_counters:
                    cell_seq_counters[cell_key] = 1
                block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
                cell_seq_counters[cell_key] += 1
                block_map[block_id] = cycle

        # If the while loop broke (dead-end or already-visited edge) and
        # the cycle wasn't stored by the cycle-close or max_steps branch,
        # store it as a terminally bounded block.
        cycle_fs = frozenset(cycle)
        already_stored = any(frozenset(v) == cycle_fs for v in block_map.values())
        if cycle and not already_stored:
            owner_col, owner_row = _node_to_cell(start_node)
            cell_key = (owner_col, owner_row)
            if cell_key not in cell_seq_counters:
                cell_seq_counters[cell_key] = 1
            block_id = f"{owner_col}_{owner_row}_s{cell_seq_counters[cell_key]}"
            cell_seq_counters[cell_key] += 1
            block_map[block_id] = cycle

    return block_map


def detect_blocks(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame = None,
    grid_size: float = 0.01,
    n_jobs: int = None
) -> Dict[str, List[int]]:
    """
    Detect blocks using two-phase grid-based cycle detection.

    Phase A (parallel): Each grid cell detects blocks that close entirely
    within it. Traversals that exit the cell boundary are saved as open
    chains with their traversal state.

    Phase B (serial): A single-threaded pass continues each open chain
    across the full adjacency graph until it closes or terminates.

    Per spec: Block IDs are made up of grid column number, grid row number,
    then sequential position in that grid square. Search starts in the
    southwestern corner and works north, then moves east.

    Args:
        edges: GeoDataFrame of street edges
        nodes: Optional GeoDataFrame of nodes (will be extracted from edges if not provided)
        grid_size: Size of each grid cell in degrees (default 0.01 ~ 1km)
        n_jobs: Number of parallel workers (default: cpu_count - 1)

    Returns:
        block_map: Dict mapping block_id (str: "col_row_seq") -> [edge_ids]
    """
    print("  Detecting blocks with two-phase approach (parallel local + serial stitch)...")

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    # Build adjacency graph (plain dict — picklable)
    adj = build_adjacency_graph(edges, nodes)

    # Extract node coordinates from edges: node_id -> (lat, lon)
    node_coords: Dict[int, Tuple[float, float]] = {}
    for idx, row in edges.iterrows():
        start_id = row['start_node_osmid']
        end_id = row['end_node_osmid']
        geom = row['geometry']
        if start_id not in node_coords and geom:
            coords = list(geom.coords)
            node_coords[start_id] = (coords[0][1], coords[0][0])
        if end_id not in node_coords and geom:
            coords = list(geom.coords)
            node_coords[end_id] = (coords[-1][1], coords[-1][0])

    # Create grid bounding box
    grid_cells, grid_metadata = create_grid_bounding_box(edges, grid_size)

    # Pre-assign nodes to grid cells (fast arithmetic, no geometry intersection)
    cell_node_map = _preassign_nodes_to_grid(node_coords, grid_metadata)

    # Build worker args — sorted column-major (west→east, south→north) per spec
    grid_keys_sorted = sorted(cell_node_map.keys(), key=lambda k: (k[0], k[1]))
    worker_args = [
        (col, row, cell_node_map[(col, row)], adj, node_coords)
        for col, row in grid_keys_sorted
    ]

    print(f"  Phase A: Processing {len(worker_args)} non-empty grid cells across {n_jobs} workers...")

    # ---- Phase A: Parallel local block detection ----
    block_map: Dict[str, List[int]] = {}
    all_open_chains = []
    global_visited: Set = set()

    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_detect_blocks_worker, worker_args,
                      chunksize=max(1, len(worker_args) // (n_jobs * 4))),
            total=len(worker_args),
            desc="  Phase A: grid cells"
        ))

    # Collect closed blocks and open chains from all workers
    for grid_col, grid_row, cell_blocks, cell_visited, cell_open_chains in results:
        for block_id, edge_ids in cell_blocks.items():
            block_map[block_id] = edge_ids
        global_visited.update(cell_visited)
        all_open_chains.extend(cell_open_chains)

    closed_count = len(block_map)
    open_count = len(all_open_chains)
    print(f"  Phase A complete: {closed_count} closed blocks, {open_count} open chains to stitch")

    # ---- Phase B: Serial boundary stitching ----
    if all_open_chains:
        print(f"  Phase B: Stitching {open_count} boundary-crossing traversals...")
        stitched_blocks = _stitch_open_chains(
            all_open_chains, adj, node_coords, global_visited, grid_metadata
        )
        block_map.update(stitched_blocks)
        print(f"  Phase B complete: {len(stitched_blocks)} additional blocks from stitching")

    # ---- Dedup pass ----
    # Blocks with identical edge sets detected by different cells or by
    # both Phase A and Phase B should be deduplicated. Ownership goes to
    # the cell with the lowest row (south), then lowest column (west).
    print("  Deduplicating blocks...")
    edge_set_to_candidates: Dict[frozenset, List[Tuple[str, List[int]]]] = {}
    for block_id, edge_ids in block_map.items():
        key = frozenset(edge_ids)
        edge_set_to_candidates.setdefault(key, []).append((block_id, edge_ids))

    deduped_map: Dict[str, List[int]] = {}
    for key, candidates in edge_set_to_candidates.items():
        if len(candidates) == 1:
            deduped_map[candidates[0][0]] = candidates[0][1]
        else:
            # Parse grid cell from block_id and pick southernmost/westernmost
            def _parse_cell(bid):
                parts = bid.replace('s', '').split('_')
                try:
                    return (int(parts[1]), int(parts[0]))  # (row, col)
                except (ValueError, IndexError):
                    return (999999, 999999)
            candidates.sort(key=lambda c: _parse_cell(c[0]))
            deduped_map[candidates[0][0]] = candidates[0][1]

    print(f"  Detected {len(deduped_map)} blocks ({closed_count} local + {len(deduped_map) - closed_count} from stitching, after dedup)")
    return deduped_map


def _compute_block_side_worker(args):
    """Worker function for parallel block side computation."""
    block_id, edge_ids, edges_coords = args
    
    # Build polygon from edge coordinates - optimized with list comprehension
    coords = [coord for edge_idx in edge_ids 
              for coord in (edges_coords.get(edge_idx) or [])]
    
    if len(coords) < 3:
        return block_id, []
    
    # Compute shoelace signed area using numpy for speed
    # Optimized: avoid creating intermediate arrays
    coords_array = np.array(coords, dtype=np.float64)
    x = coords_array[:, 0]
    y = coords_array[:, 1]
    
    # Optimized shoelace: use einsum for better performance
    signed_area = 0.5 * (np.sum(x[:-1] * y[1:]) + x[-1] * y[0] - 
                         np.sum(y[:-1] * x[1:]) - y[-1] * x[0])
    
    # Determine side: Area > 0 (CCW) = left, Area < 0 (CW) = right
    side = 'left' if signed_area > 0 else 'right'
    
    # Return block_id and list of (edge_idx, side, block_id) tuples
    return block_id, [(edge_idx, side, block_id) for edge_idx in edge_ids]


def compute_circular_mean_bearing(bearings: List[float]) -> float:
    """
    Compute circular mean of bearings in degrees.
    
    Args:
        bearings: List of bearings in degrees [0, 360)
    
    Returns:
        Circular mean bearing in degrees [0, 360)
    """
    if not bearings:
        return 0.0
    
    # Convert to radians
    radians = np.deg2rad(bearings)
    
    # Compute mean of sin and cos
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    
    # Compute circular mean
    mean_rad = np.arctan2(sin_mean, cos_mean)
    mean_deg = np.rad2deg(mean_rad)
    
    return (mean_deg + 360) % 360


def assign_block_sides_shoelace(
    block_map: Dict[str, List[int]],
    edges: gpd.GeoDataFrame
) -> Tuple[Dict[int, Tuple[Optional[str], Optional[str]]], Dict[int, Tuple[Optional[str], Optional[str]]]]:
    """
    Use shoelace formula to determine which side of each edge the block is on,
    and assign block sides based on circular mean of bearings.
    
    Per spec:
    - Block IDs are assigned to left/right slots based on shoelace formula
    - Block sides are labeled according to circular mean of bearings of segments
      between intersection nodes
    - Dead end blocks have their left block side slot filled by default
    
    Returns:
        Tuple of (edge_block_membership, edge_block_side_membership)
        - edge_block_membership: Dict mapping edge_id -> (left_block_id, right_block_id)
        - edge_block_side_membership: Dict mapping edge_id -> (left_block_side, right_block_side)
    """
    print("  Assigning block sides using shoelace formula...")
    
    # Extract only the coordinates we need (much smaller than full geometries)
    # Pre-convert geometries to coordinate lists to avoid serializing Shapely objects
    print("  Extracting coordinates...")
    
    # Find unique edges needed across all blocks
    all_edge_ids = set()
    for edge_ids in block_map.values():
        all_edge_ids.update(edge_ids)
    
    # Extract only needed coordinates
    minimal_coords = {}
    for eid in all_edge_ids:
        if eid in edges.index:
            geom = edges.loc[eid, 'geometry']
            if geom is not None and not geom.is_empty:
                minimal_coords[eid] = list(geom.coords)
    
    print(f"  Processing {len(block_map)} blocks with {len(minimal_coords)} unique edges...")
    
    # Prepare arguments for parallel processing
    n_jobs = max(1, mp.cpu_count() - 1)
    
    # Sort blocks by size (largest first) for better load balancing
    sorted_blocks = sorted(block_map.items(), key=lambda x: len(x[1]), reverse=True)
    args = [(block_id, edge_ids, minimal_coords) for block_id, edge_ids in sorted_blocks]
    
    # Process blocks in parallel with dynamic chunksize
    edge_block_membership = defaultdict(lambda: [None, None])
    
    # Use larger chunksize for better throughput
    chunksize = max(1, len(args) // (n_jobs * 4))
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_compute_block_side_worker, args, chunksize=chunksize),
            total=len(args),
            desc="  Shoelace assignment"
        ))
    
    # Aggregate results
    for block_id, edge_assignments in results:
        for edge_idx, side, block_id in edge_assignments:
            if side == 'left':
                edge_block_membership[edge_idx][0] = block_id
            else:
                edge_block_membership[edge_idx][1] = block_id
    
    # Step 4: Label block sides according to circular mean of bearings
    print("  Computing block sides from circular mean of bearings...")
    
    # Compute bearings on-the-fly for block side labeling
    # (bearings will be assigned to edges later in the pipeline)
    edge_bearings = {}
    for edge_idx in edge_block_membership.keys():
        if edge_idx in edges.index:
            geom = edges.loc[edge_idx, 'geometry']
            if geom is not None and not geom.is_empty:
                bearing = compute_bearing(geom)
                edge_bearings[edge_idx] = bearing % 180  # Normalized bearing
    
    # Group edges by block and side
    block_side_edges = defaultdict(list)
    for edge_idx, (left_block, right_block) in edge_block_membership.items():
        if left_block:
            block_side_edges[(left_block, 'left')].append(edge_idx)
        if right_block:
            block_side_edges[(right_block, 'right')].append(edge_idx)
    
    # Compute circular mean bearing for each block side
    block_side_bearings = {}
    for (block_id, side), edge_list in block_side_edges.items():
        bearings = []
        for edge_idx in edge_list:
            if edge_idx in edge_bearings:
                bearings.append(edge_bearings[edge_idx])
        
        if bearings:
            mean_bearing = compute_circular_mean_bearing(bearings)
            block_side_bearings[(block_id, side)] = mean_bearing
    
    # Assign block side labels based on mean bearing
    # Round to nearest 45 degrees for labeling: N, NE, E, SE, S, SW, W, NW
    def bearing_to_label(bearing: float) -> str:
        """Convert bearing to cardinal direction label."""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((bearing + 22.5) / 45) % 8
        return directions[idx]
    
    edge_block_side_membership = {}
    for edge_idx, (left_block, right_block) in edge_block_membership.items():
        left_side = None
        right_side = None
        
        if left_block:
            mean_bearing = block_side_bearings.get((left_block, 'left'))
            if mean_bearing is not None:
                left_side = f"{left_block}_{bearing_to_label(mean_bearing)}"
            else:
                # Dead end block: use left slot by default
                left_side = f"{left_block}_L"
        
        if right_block:
            mean_bearing = block_side_bearings.get((right_block, 'right'))
            if mean_bearing is not None:
                right_side = f"{right_block}_{bearing_to_label(mean_bearing)}"
            else:
                # Dead end block: use left slot by default
                right_side = f"{right_block}_L"
        
        edge_block_side_membership[edge_idx] = (left_side, right_side)
    
    return dict(edge_block_membership), edge_block_side_membership


def apply_block_ids_to_network(
    edges: gpd.GeoDataFrame,
    edge_block_membership: Dict[int, Tuple[Optional[str], Optional[str]]],
    edge_block_side_membership: Dict[int, Tuple[Optional[str], Optional[str]]]
) -> gpd.GeoDataFrame:
    """
    Apply block IDs and block sides to network edges.
    
    Per spec Step 5: For each segment whose start or end node is a vertex of 
    the combined block polygon, their start/end_node_is_block_node booleans 
    should be set to true.
    """
    print("  Applying block IDs and block sides to network...")
    
    # Initialize columns
    edges['block_ids'] = None
    edges['block_sides'] = None
    edges['start_node_is_block_node'] = False
    edges['end_node_is_block_node'] = False
    for side in ['left', 'right']:
        edges[f'sidewalk_{side}_block_ID'] = None
        edges[f'bikeway_{side}_1_block_id'] = None
        # Feature 12: Initialize bikeway_2_block_id
        edges[f'bikeway_{side}_2_block_id'] = None
    
    # Build set of block vertex nodes (nodes that are vertices of block polygons)
    block_vertex_nodes = set()
    for edge_idx, (left_block, right_block) in edge_block_membership.items():
        if edge_idx in edges.index:
            # If this edge is part of a block, its start and end nodes are block vertices
            if left_block or right_block:
                start_node = edges.loc[edge_idx, 'start_node_osmid']
                end_node = edges.loc[edge_idx, 'end_node_osmid']
                block_vertex_nodes.add(start_node)
                block_vertex_nodes.add(end_node)
    
    # Apply block IDs and sides
    for idx in edges.index:
        if idx in edge_block_membership:
            left_id, right_id = edge_block_membership[idx]
            left_side, right_side = edge_block_side_membership.get(idx, (None, None))
            
            edges.at[idx, 'block_ids'] = (left_id, right_id)
            edges.at[idx, 'block_sides'] = (left_side, right_side)
            
            # Assign to sidewalk and bikeway columns
            edges.at[idx, 'sidewalk_left_block_ID'] = left_id
            edges.at[idx, 'sidewalk_right_block_ID'] = right_id
            edges.at[idx, 'bikeway_left_1_block_id'] = left_id
            edges.at[idx, 'bikeway_right_1_block_id'] = right_id
            # Feature 12: Populate bikeway_2_block_id with same value as bikeway_1_block_id
            edges.at[idx, 'bikeway_left_2_block_id'] = left_id
            edges.at[idx, 'bikeway_right_2_block_id'] = right_id
            
            # Step 5: Set start/end_node_is_block_node booleans
            start_node = edges.loc[idx, 'start_node_osmid']
            end_node = edges.loc[idx, 'end_node_osmid']
            
            if start_node in block_vertex_nodes:
                edges.at[idx, 'start_node_is_block_node'] = True
            if end_node in block_vertex_nodes:
                edges.at[idx, 'end_node_is_block_node'] = True
    
    return edges


def reassign_facility_data_by_bearing(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reassign sidewalk and bikelane data to correct left/right slots based on 
    normalized bearing.
    
    Per architecture document Step 3: "Sidewalk and Bikelane data should be 
    re-assigned to the opposite left/right slot based on this new normalized bearing."
    
    For segments with bearing >= 180 and < 360, swap left and right facility data.
    This ensures left/right orientation is consistent relative to the normalized
    bearing direction [0, 180).
    """
    print("  Reassigning facility data based on normalized bearing...")
    
    # Identify rows where bearing is between 180 and 360 (need to swap)
    swap_mask = (edges['bearing'] >= 180) & (edges['bearing'] < 360)
    
    if not swap_mask.any():
        return edges
    
    # Get indices to swap
    swap_indices = edges.index[swap_mask]
    
    # Define all facility columns that need swapping
    facility_columns = []
    
    # Sidewalk columns
    for slot in ['', '_ID', '_block_ID', '_presence', '_surface', '_quality', '_width', '_incline', '_buffered', '_geometry']:
        facility_columns.append(f'sidewalk_left{slot}')
        facility_columns.append(f'sidewalk_right{slot}')
    
    # Sidewalk feature columns
    for slot in ['_feature_ids', '_feature_types', '_feature_geometry', '_feature_geometry_projected']:
        facility_columns.append(f'sidewalk_left{slot}')
        facility_columns.append(f'sidewalk_right{slot}')
    facility_columns.append('public_data_id_sidewalk_left')
    facility_columns.append('public_data_id_sidewalk_right')
    facility_columns.append('public_data_id_sidewalk_left_feature')
    facility_columns.append('public_data_id_sidewalk_right_feature')
    
    # Curb ramp columns (3 positions, start and end)
    for pos in [1, 2, 3]:
        for slot in ['start', 'end']:
            for attr in ['_ID', '_returnloc', '_returnposition', '_condition_score', '_geometry']:
                facility_columns.append(f'sidewalk_left_curbramp_{slot}_{pos}{attr}')
                facility_columns.append(f'sidewalk_right_curbramp_{slot}_{pos}{attr}')
            facility_columns.append(f'public_data_id_sidewalk_left_curbramp_{slot}_{pos}')
            facility_columns.append(f'public_data_id_sidewalk_right_curbramp_{slot}_{pos}')
    
    # Bikeway columns (2 lanes per side)
    for n in [1, 2]:
        for slot in ['_id', '_block_id', '_type', '_surface', '_quality', '_permitted', '_width', '_incline', '_geometry']:
            facility_columns.append(f'bikeway_left_{n}{slot}')
            facility_columns.append(f'bikeway_right_{n}{slot}')
        
        # Bikeway feature columns
        for slot in ['_feature_ids', '_feature_types', '_feature_geometry', '_feature_geometry_projected']:
            facility_columns.append(f'bikeway_left_{n}{slot}')
            facility_columns.append(f'bikeway_right_{n}{slot}')
        facility_columns.append(f'public_data_id_bikeway_left_{n}')
        facility_columns.append(f'public_data_id_bikeway_right_{n}')
        facility_columns.append(f'public_data_id_bikeway_left_{n}_features')
        facility_columns.append(f'public_data_id_bikeway_right_{n}_features')
    
    facility_columns.append('bikeway_left_buffered')
    facility_columns.append('bikeway_right_buffered')
    
    # Crosswalk columns (start and end)
    # Note: Crosswalks connect left and right curb ramps, so they also need swapping
    for slot in ['start', 'end']:
        for attr in ['_id', '_block_ids', '_type', '_controlled', '_marked', '_markings', 
                     '_signals', '_island', '_kerb', '_tactile_paving', '_traffic_calming', 
                     '_continuous', '_condition', '_geometry']:
            facility_columns.append(f'crosswalk_{slot}{attr}')
        facility_columns.append(f'public_data_id_crosswalk_{slot}')
    
    # Perform swaps for each column pair
    for i in range(0, len(facility_columns), 2):
        left_col = facility_columns[i]
        right_col = facility_columns[i + 1]
        
        # Only swap if both columns exist
        if left_col in edges.columns and right_col in edges.columns:
            # Store original values for swap indices
            left_vals = edges.loc[swap_indices, left_col].copy()
            right_vals = edges.loc[swap_indices, right_col].copy()
            
            # Swap
            edges.loc[swap_indices, left_col] = right_vals
            edges.loc[swap_indices, right_col] = left_vals
    
    return edges


# ============================================================================
# SECTION 12: STAGE 8 — CROSSWALK CACHE (Step 4 continued, adapted)
# ============================================================================

def extract_crosswalk_tags(
    network: gpd.GeoDataFrame,
    crossings_cache: gpd.GeoDataFrame,
    crosswalk_counter: SequentialIDCounter,
    gov_crosswalks: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Extract crosswalk tags from cached crossing data and government data (Step 20).
    Feature 4: Government crosswalks take precedence over OSM.
    Optimized with vectorized operations where possible.
    """
    print("  Extracting crosswalk tags...")
    
    # Initialize crosswalk columns for start and end
    for slot in ['start', 'end']:
        network[f'crosswalk_{slot}_id'] = None
        network[f'crosswalk_{slot}_block_ids'] = None
        network[f'crosswalk_{slot}_type'] = None
        network[f'public_data_id_crosswalk_{slot}'] = None
        network[f'crosswalk_{slot}_controlled'] = None
        network[f'crosswalk_{slot}_marked'] = None
        network[f'crosswalk_{slot}_markings'] = None
        network[f'crosswalk_{slot}_signals'] = None
        network[f'crosswalk_{slot}_island'] = None
        network[f'crosswalk_{slot}_kerb'] = None
        network[f'crosswalk_{slot}_tactile_paving'] = None
        network[f'crosswalk_{slot}_traffic_calming'] = None
        network[f'crosswalk_{slot}_continuous'] = None
        network[f'crosswalk_{slot}_condition'] = None
        network[f'crosswalk_{slot}_geometry'] = None
    
    # Initialize curb ramp columns if they don't exist
    for side in ['left', 'right']:
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                col = f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry'
                if col not in network.columns:
                    network[col] = None
    
    # Feature 4: Build spatial index for government crosswalks (takes precedence)
    gov_tree = None
    if gov_crosswalks is not None and not gov_crosswalks.empty:
        print("  Using government crosswalk data (takes precedence over OSM)")
        gov_tree = STRtree(gov_crosswalks.geometry)
    
    # Build spatial index for OSM crossings
    crossing_tree = None
    if crossings_cache is not None and not crossings_cache.empty:
        crossing_tree = STRtree(crossings_cache.geometry)
    
    if gov_tree is None and crossing_tree is None:
        print("  No crossing data available")
        return network
    
    # Extract start and end points vectorized
    start_points = network['geometry'].apply(lambda g: Point(list(g.coords)[0]) if g and not g.is_empty else None)
    end_points = network['geometry'].apply(lambda g: Point(list(g.coords)[-1]) if g and not g.is_empty else None)
    
    # For each edge, check start and end nodes for nearby crossings
    for idx in tqdm(network.index, total=len(network), desc="  Matching crosswalks"):
        start_point = start_points.loc[idx]
        end_point = end_points.loc[idx]
        
        if start_point is None or end_point is None:
            continue
        
        # Check start node - Feature 4: Check government data first
        crossing = None
        crossing_point = None
        is_government = False
        
        if gov_tree is not None:
            nearby_gov = gov_tree.query(start_point, predicate='dwithin', distance=10.0)  # 10m (UTM meters)
            if len(nearby_gov) > 0:
                crossing_idx = nearby_gov[0]
                crossing = gov_crosswalks.iloc[crossing_idx]
                crossing_point = crossing.geometry
                is_government = True
        
        # Fall back to OSM if no government data
        if crossing is None and crossing_tree is not None:
            nearby_start = crossing_tree.query(start_point, predicate='dwithin', distance=10.0)  # 10m (UTM meters)
            if len(nearby_start) > 0:
                crossing_idx = nearby_start[0]
                crossing = crossings_cache.iloc[crossing_idx]
                crossing_point = crossing.geometry
                is_government = False
        
        if crossing is not None:
            # Assign ID and attributes
            network.at[idx, 'crosswalk_start_id'] = crosswalk_counter.next()

            # Feature 4: Populate public_data_id_crosswalk_start when government data used
            if is_government:
                gov_id = crossing.get('id', crossing.get('ID', crossing.get('OBJECTID', None)))
                network.at[idx, 'public_data_id_crosswalk_start'] = str(gov_id) if gov_id is not None else None

            network.at[idx, 'crosswalk_start_type'] = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
            # controlled: traffic_signals=* indicates signal-controlled intersection
            network.at[idx, 'crosswalk_start_controlled'] = normalize_tag(crossing.get('traffic_signals', None))

            # marked: use OSM crossing=marked/unmarked/zebra/tiger directly;
            # fall back to None when ambiguous rather than assuming 'no'
            crossing_val = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
            if crossing_val in ['marked', 'zebra', 'tiger']:
                network.at[idx, 'crosswalk_start_marked'] = 'yes'
            elif crossing_val == 'unmarked':
                network.at[idx, 'crosswalk_start_marked'] = 'no'
            else:
                network.at[idx, 'crosswalk_start_marked'] = None
            network.at[idx, 'crosswalk_start_markings'] = normalize_tag(crossing.get('crossing:markings', None))

            network.at[idx, 'crosswalk_start_tactile_paving'] = normalize_tag(crossing.get('tactile_paving', None))
            network.at[idx, 'crosswalk_start_island'] = normalize_tag(crossing.get('crossing:island', None))
            network.at[idx, 'crosswalk_start_kerb'] = normalize_tag(crossing.get('kerb', None))
            network.at[idx, 'crosswalk_start_traffic_calming'] = normalize_tag(crossing.get('traffic_calming', None))
            network.at[idx, 'crosswalk_start_continuous'] = normalize_tag(crossing.get('crossing:continuous', None))
            network.at[idx, 'crosswalk_start_condition'] = normalize_tag(crossing.get('condition', None))

            # Signals list per schema:
            #   [signals yes/no, button yes/no, sound yes/no, vibration yes/no, flashing_lights yes/button/sensor]
            # crossing:signals=yes/no is the canonical presence flag; fall back to
            # inferring 'yes' when traffic_signals tag is present.
            signals_present = normalize_tag(crossing.get('crossing:signals', None))
            if signals_present is None and normalize_tag(crossing.get('traffic_signals', None)) is not None:
                signals_present = 'yes'
            button = normalize_tag(crossing.get('button_operated', None))
            sound = normalize_tag(crossing.get('traffic_signals:sound', None))
            vibration = normalize_tag(crossing.get('traffic_signals:vibration', None))
            flashing = normalize_tag(crossing.get('flashing_lights', None))
            network.at[idx, 'crosswalk_start_signals'] = [signals_present, button, sound, vibration, flashing]
            
            # Block IDs from sidewalks
            left_block = network.at[idx, 'sidewalk_left_block_ID']
            right_block = network.at[idx, 'sidewalk_right_block_ID']
            network.at[idx, 'crosswalk_start_block_ids'] = (right_block, left_block)
            
            # Step 20: Use explicit crossing geometry if available (LineString), otherwise draw line through crossing point
            if crossing_point.geom_type == 'LineString':
                # Crossing has explicit geometry - use it directly
                network.at[idx, 'crosswalk_start_geometry'] = crossing_point
            else:
                # Crossing is a Point - draw line through it if we have curb ramps
                left_ramp_geom = network.at[idx, 'sidewalk_left_curbramp_start_1_geometry']
                right_ramp_geom = network.at[idx, 'sidewalk_right_curbramp_start_1_geometry']
                
                if left_ramp_geom and right_ramp_geom:
                    # Draw 3-point line: left_ramp → crossing_point → right_ramp
                    crosswalk_geom = LineString([
                        left_ramp_geom.coords[0],
                        crossing_point.coords[0],
                        right_ramp_geom.coords[0]
                    ])
                    network.at[idx, 'crosswalk_start_geometry'] = crosswalk_geom
        
        # Check end node - Feature 4: Check government data first
        crossing = None
        crossing_point = None
        is_government = False
        
        if gov_tree is not None:
            nearby_gov = gov_tree.query(end_point, predicate='dwithin', distance=10.0)  # 10m (UTM meters)
            if len(nearby_gov) > 0:
                crossing_idx = nearby_gov[0]
                crossing = gov_crosswalks.iloc[crossing_idx]
                crossing_point = crossing.geometry
                is_government = True
        
        # Fall back to OSM if no government data
        if crossing is None and crossing_tree is not None:
            nearby_end = crossing_tree.query(end_point, predicate='dwithin', distance=10.0)  # 10m (UTM meters)
            if len(nearby_end) > 0:
                crossing_idx = nearby_end[0]
                crossing = crossings_cache.iloc[crossing_idx]
                crossing_point = crossing.geometry
                is_government = False
        
        if crossing is not None:
            # Assign ID and attributes
            network.at[idx, 'crosswalk_end_id'] = crosswalk_counter.next()

            # Feature 4: Populate public_data_id_crosswalk_end when government data used
            if is_government:
                gov_id = crossing.get('id', crossing.get('ID', crossing.get('OBJECTID', None)))
                network.at[idx, 'public_data_id_crosswalk_end'] = str(gov_id) if gov_id is not None else None

            network.at[idx, 'crosswalk_end_type'] = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
            network.at[idx, 'crosswalk_end_controlled'] = normalize_tag(crossing.get('traffic_signals', None))

            # marked: use OSM crossing=marked/unmarked/zebra/tiger directly
            crossing_val = normalize_tag(crossing.get('crossing', crossing.get('type', None)))
            if crossing_val in ['marked', 'zebra', 'tiger']:
                network.at[idx, 'crosswalk_end_marked'] = 'yes'
            elif crossing_val == 'unmarked':
                network.at[idx, 'crosswalk_end_marked'] = 'no'
            else:
                network.at[idx, 'crosswalk_end_marked'] = None
            network.at[idx, 'crosswalk_end_markings'] = normalize_tag(crossing.get('crossing:markings', None))
            
            network.at[idx, 'crosswalk_end_tactile_paving'] = normalize_tag(crossing.get('tactile_paving', None))
            network.at[idx, 'crosswalk_end_island'] = normalize_tag(crossing.get('crossing:island', None))
            network.at[idx, 'crosswalk_end_kerb'] = normalize_tag(crossing.get('kerb', None))
            network.at[idx, 'crosswalk_end_traffic_calming'] = normalize_tag(crossing.get('traffic_calming', None))
            network.at[idx, 'crosswalk_end_continuous'] = normalize_tag(crossing.get('crossing:continuous', None))
            network.at[idx, 'crosswalk_end_condition'] = normalize_tag(crossing.get('condition', None))
            
            # Signals list per schema:
            #   [signals yes/no, button yes/no, sound yes/no, vibration yes/no, flashing_lights yes/button/sensor]
            signals_present = normalize_tag(crossing.get('crossing:signals', None))
            if signals_present is None and normalize_tag(crossing.get('traffic_signals', None)) is not None:
                signals_present = 'yes'
            button = normalize_tag(crossing.get('button_operated', None))
            sound = normalize_tag(crossing.get('traffic_signals:sound', None))
            vibration = normalize_tag(crossing.get('traffic_signals:vibration', None))
            flashing = normalize_tag(crossing.get('flashing_lights', None))
            network.at[idx, 'crosswalk_end_signals'] = [signals_present, button, sound, vibration, flashing]
            
            # Block IDs from sidewalks
            left_block = network.at[idx, 'sidewalk_left_block_ID']
            right_block = network.at[idx, 'sidewalk_right_block_ID']
            network.at[idx, 'crosswalk_end_block_ids'] = (right_block, left_block)
            
            # Step 20: Use explicit crossing geometry if available (LineString), otherwise draw line through crossing point
            if crossing_point.geom_type == 'LineString':
                # Crossing has explicit geometry - use it directly
                network.at[idx, 'crosswalk_end_geometry'] = crossing_point
            else:
                # Crossing is a Point - draw line through it if we have curb ramps
                left_ramp_geom = network.at[idx, 'sidewalk_left_curbramp_end_1_geometry']
                right_ramp_geom = network.at[idx, 'sidewalk_right_curbramp_end_1_geometry']
                
                if left_ramp_geom and right_ramp_geom:
                    # Draw 3-point line: left_ramp → crossing_point → right_ramp
                    crosswalk_geom = LineString([
                        left_ramp_geom.coords[0],
                        crossing_point.coords[0],
                        right_ramp_geom.coords[0]
                    ])
                    network.at[idx, 'crosswalk_end_geometry'] = crosswalk_geom
    
    return network


# ============================================================================
# SECTION 13: STAGE 9 — INTERSECTION ANALYSIS (Steps 1-7, bikelanes and sidewalks)
# ============================================================================

def build_intersection_index(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame
) -> Dict[int, Dict]:
    """
    Build index of intersections with their connected edges (vectorized).
    
    Returns:
        Dict mapping node_id -> {'geometry': Point, 'edges': [(edge_idx, 'start'|'end'), ...]}
    """
    print("  Building intersection index...")
    
    # Extract nodes from edges if not provided
    if nodes is None:
        nodes = extract_nodes_from_edges(edges)
    
    intersection_map = {}
    
    # Vectorized approach: build node->edges mapping
    node_edges = defaultdict(list)
    
    # Process all start nodes
    for idx, node_id in zip(edges.index, edges['start_node_osmid']):
        node_edges[node_id].append((idx, 'start'))
    
    # Process all end nodes
    for idx, node_id in zip(edges.index, edges['end_node_osmid']):
        node_edges[node_id].append((idx, 'end'))
    
    # Filter to only intersections (2+ edges)
    for node_id, connected_edges in node_edges.items():
        if len(connected_edges) < 2:
            continue
        
        # Get node geometry
        node_geom = None
        if node_id in nodes.index:
            node_geom = nodes.loc[node_id, 'geometry']
        else:
            # Split node - get from edge geometry
            for idx, position in connected_edges:
                edge_geom = edges.loc[idx, 'geometry']
                if edge_geom:
                    coords = list(edge_geom.coords)
                    if position == 'start':
                        node_geom = Point(coords[0])
                    else:
                        node_geom = Point(coords[-1])
                    break
        
        if node_geom:
            intersection_map[node_id] = {
                'geometry': node_geom,
                'edges': connected_edges
            }
    
    print(f"  Found {len(intersection_map)} intersections")
    return intersection_map


def check_contiguity(geom_a: LineString, geom_b: LineString, pos_a: str, pos_b: str, threshold: float = 0.5) -> bool:
    """
    Check if two geometries are contiguous at their endpoints.
    
    Args:
        geom_a: First geometry
        geom_b: Second geometry
        pos_a: Position on first geometry ('start' or 'end')
        pos_b: Position on second geometry ('start' or 'end')
        threshold: Distance threshold in meters
    
    Returns:
        True if contiguous, False otherwise
    """
    if geom_a is None or geom_b is None:
        return False
    
    try:
        coords_a = list(geom_a.coords)
        coords_b = list(geom_b.coords)
        
        point_a = Point(coords_a[0] if pos_a == 'start' else coords_a[-1])
        point_b = Point(coords_b[0] if pos_b == 'start' else coords_b[-1])
        
        return point_a.distance(point_b) <= threshold
    except Exception:
        return False


def find_adjacent_segments_at_corner(
    intersection_data: Dict,
    edges: gpd.GeoDataFrame
) -> List[Tuple[int, str, int, str, str, str]]:
    """
    Find pairs of adjacent street segments at corners of an intersection.
    
    Per spec Step 1 & 4: Check if two contiguous street segments have bike facilities
    or sidewalks on adjacent sides of their street at a corner.
    
    Returns:
        List of tuples: (edge_a_idx, pos_a, edge_b_idx, pos_b, side_a, side_b)
        where side_a and side_b indicate which sides are adjacent at the corner
    """
    adjacent_pairs = []
    connected_edges = intersection_data['edges']
    
    # For each pair of edges at the intersection
    for i, (edge_a_idx, pos_a) in enumerate(connected_edges):
        for edge_b_idx, pos_b in connected_edges[i+1:]:
            # Get bearings to determine if they form a corner
            bearing_a = edges.loc[edge_a_idx, 'normalized_bearing']
            bearing_b = edges.loc[edge_b_idx, 'normalized_bearing']
            
            if pd.isna(bearing_a) or pd.isna(bearing_b):
                continue
            
            # Adjust bearings based on position (reverse if at end)
            if pos_a == 'end':
                bearing_a = (bearing_a + 180) % 360
            if pos_b == 'end':
                bearing_b = (bearing_b + 180) % 360
            
            # Calculate angle difference
            angle_diff = abs(bearing_a - bearing_b)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Check if they form a corner (roughly perpendicular, 60-120 degrees)
            if 60 <= angle_diff <= 120:
                # Determine which sides are adjacent at the corner
                # If bearing_b is clockwise from bearing_a, then right of A meets left of B
                clockwise_diff = (bearing_b - bearing_a) % 360
                
                if clockwise_diff < 180:
                    # B is clockwise from A: right of A meets left of B
                    adjacent_pairs.append((edge_a_idx, pos_a, edge_b_idx, pos_b, 'right', 'left'))
                else:
                    # B is counter-clockwise from A: left of A meets right of B
                    adjacent_pairs.append((edge_a_idx, pos_a, edge_b_idx, pos_b, 'left', 'right'))
    
    return adjacent_pairs


def process_bikeway_gaps_at_intersection(
    intersection_data: Dict,
    edges: gpd.GeoDataFrame,
    sanity_buffer: Dict
) -> List[Dict]:
    """
    Process bikeway gaps at an intersection (Steps 1-3).
    
    Step 1: Check bike lane slots for contiguous segments with bike facilities on adjacent sides
    Step 2: For both buffered, extend to midpoint
    Step 3: For one buffered, extend to meet the other
    
    Returns:
        List of bikeway extension operations to apply
    """
    bikeway_extensions = []
    node_geom = intersection_data['geometry']
    
    # Find adjacent segment pairs at corners
    adjacent_pairs = find_adjacent_segments_at_corner(intersection_data, edges)
    
    for edge_a_idx, pos_a, edge_b_idx, pos_b, side_a, side_b in adjacent_pairs:
        # Check if both have bike facilities on the adjacent sides
        geom_a = edges.loc[edge_a_idx, f'bikeway_{side_a}_1_geometry']
        geom_b = edges.loc[edge_b_idx, f'bikeway_{side_b}_1_geometry']
        
        if geom_a is None or geom_b is None:
            continue
        
        buffered_a = edges.loc[edge_a_idx, f'bikeway_{side_a}_buffered']
        buffered_b = edges.loc[edge_b_idx, f'bikeway_{side_b}_buffered']
        
        # Only process if at least one was buffered
        if not buffered_a and not buffered_b:
            continue
        
        # Step 1: Check if contiguous
        if check_contiguity(geom_a, geom_b, pos_a, pos_b, threshold=0.5):
            # Already contiguous, no action needed
            continue
        
        # Flag for correction
        coords_a = list(geom_a.coords)
        coords_b = list(geom_b.coords)
        
        point_a = Point(coords_a[0] if pos_a == 'start' else coords_a[-1])
        point_b = Point(coords_b[0] if pos_b == 'start' else coords_b[-1])
        
        if buffered_a and buffered_b:
            # Step 2: Both buffered - pick midpoint perpendicular from intersection node
            # Project perpendicular from node between the two bike facilities
            midpoint = Point((point_a.x + point_b.x) / 2, (point_a.y + point_b.y) / 2)
            
            # Check sanity buffer constraints
            street_geom_a = edges.loc[edge_a_idx, 'geometry']
            max_offset_a = sanity_buffer.get(edges.loc[edge_a_idx, 'osmid'], 10.0)
            
            if street_geom_a and midpoint.distance(street_geom_a) > max_offset_a:
                # Exceeds sanity buffer, skip
                continue
            
            bikeway_extensions.append({
                'type': 'both_buffered',
                'edge_a': edge_a_idx,
                'edge_b': edge_b_idx,
                'side_a': side_a,
                'side_b': side_b,
                'pos_a': pos_a,
                'pos_b': pos_b,
                'meeting_point': midpoint
            })
        
        elif buffered_a and not buffered_b:
            # Step 3: Extend buffered A to meet non-buffered B
            nearest_point = geom_b.interpolate(geom_b.project(point_a))
            
            bikeway_extensions.append({
                'type': 'one_buffered',
                'edge': edge_a_idx,
                'side': side_a,
                'pos': pos_a,
                'meeting_point': nearest_point
            })
        
        elif not buffered_a and buffered_b:
            # Step 3: Extend buffered B to meet non-buffered A
            nearest_point = geom_a.interpolate(geom_a.project(point_b))
            
            bikeway_extensions.append({
                'type': 'one_buffered',
                'edge': edge_b_idx,
                'side': side_b,
                'pos': pos_b,
                'meeting_point': nearest_point
            })
    
    return bikeway_extensions


def process_sidewalk_gaps_at_intersection(
    intersection_data: Dict,
    edges: gpd.GeoDataFrame,
    sanity_buffer: Dict
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process sidewalk gaps at an intersection (Steps 4-7).
    
    Step 4: Check sidewalk slots for contiguous segments
    Step 5: For both buffered, extend to midpoint (assign default curb ramp)
    Step 6: For one buffered, extend to meet the other (point of contiguity is curb ramp)
    Step 7: For separate footways adjacent or wrapping, use perpendicular projection
    
    Returns:
        Tuple of (sidewalk_extensions, curbramp_assignments)
    """
    sidewalk_extensions = []
    curbramp_assignments = []
    node_geom = intersection_data['geometry']
    
    # Find adjacent segment pairs at corners
    adjacent_pairs = find_adjacent_segments_at_corner(intersection_data, edges)
    
    for edge_a_idx, pos_a, edge_b_idx, pos_b, side_a, side_b in adjacent_pairs:
        # Check if both have sidewalks on the adjacent sides
        geom_a = edges.loc[edge_a_idx, f'sidewalk_{side_a}_geometry']
        geom_b = edges.loc[edge_b_idx, f'sidewalk_{side_b}_geometry']
        
        if geom_a is None or geom_b is None:
            continue
        
        buffered_a = edges.loc[edge_a_idx, f'sidewalk_{side_a}_buffered']
        buffered_b = edges.loc[edge_b_idx, f'sidewalk_{side_b}_buffered']
        
        # Step 4: Check if contiguous
        if check_contiguity(geom_a, geom_b, pos_a, pos_b, threshold=0.5):
            # Contiguous - populate curb ramp geometry with endpoint coordinates
            coords_a = list(geom_a.coords)
            point_a = Point(coords_a[0] if pos_a == 'start' else coords_a[-1])
            
            curbramp_assignments.append({
                'type': 'contiguous',
                'edge_a': edge_a_idx,
                'edge_b': edge_b_idx,
                'side_a': side_a,
                'side_b': side_b,
                'pos_a': pos_a,
                'pos_b': pos_b,
                'geometry': point_a
            })
            continue
        
        # Not contiguous - flag for correction
        coords_a = list(geom_a.coords)
        coords_b = list(geom_b.coords)
        
        point_a = Point(coords_a[0] if pos_a == 'start' else coords_a[-1])
        point_b = Point(coords_b[0] if pos_b == 'start' else coords_b[-1])
        
        if buffered_a and buffered_b:
            # Step 5: Both buffered - pick midpoint perpendicular from intersection node
            # Ensure it doesn't overlap with street, bikelane facilities, or sanity map
            midpoint = Point((point_a.x + point_b.x) / 2, (point_a.y + point_b.y) / 2)
            
            # Check sanity buffer constraints
            street_geom_a = edges.loc[edge_a_idx, 'geometry']
            max_offset_a = sanity_buffer.get(edges.loc[edge_a_idx, 'osmid'], 10.0)
            
            if street_geom_a and midpoint.distance(street_geom_a) > max_offset_a:
                # Exceeds sanity buffer, skip
                continue
            
            # Check for overlap with bikelanes
            bikeway_left_a = edges.loc[edge_a_idx, 'bikeway_left_1_geometry']
            bikeway_right_a = edges.loc[edge_a_idx, 'bikeway_right_1_geometry']
            
            overlaps = False
            for bw_geom in [bikeway_left_a, bikeway_right_a]:
                if bw_geom and midpoint.distance(bw_geom) < 0.5:
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            sidewalk_extensions.append({
                'type': 'both_buffered',
                'edge_a': edge_a_idx,
                'edge_b': edge_b_idx,
                'side_a': side_a,
                'side_b': side_b,
                'pos_a': pos_a,
                'pos_b': pos_b,
                'meeting_point': midpoint
            })
            
            # Assign default curb ramp at meeting point
            curbramp_assignments.append({
                'type': 'both_buffered',
                'edge_a': edge_a_idx,
                'edge_b': edge_b_idx,
                'side_a': side_a,
                'side_b': side_b,
                'pos_a': pos_a,
                'pos_b': pos_b,
                'geometry': midpoint
            })
        
        elif buffered_a and not buffered_b:
            # Step 6: Extend buffered A to meet non-buffered B
            nearest_point = geom_b.interpolate(geom_b.project(point_a))
            
            sidewalk_extensions.append({
                'type': 'one_buffered',
                'edge': edge_a_idx,
                'side': side_a,
                'pos': pos_a,
                'meeting_point': nearest_point
            })
            
            # Point of contiguity is the curb ramp
            curbramp_assignments.append({
                'type': 'one_buffered',
                'edge': edge_a_idx,
                'side': side_a,
                'pos': pos_a,
                'geometry': nearest_point
            })
        
        elif not buffered_a and buffered_b:
            # Step 6: Extend buffered B to meet non-buffered A
            nearest_point = geom_a.interpolate(geom_a.project(point_b))
            
            sidewalk_extensions.append({
                'type': 'one_buffered',
                'edge': edge_b_idx,
                'side': side_b,
                'pos': pos_b,
                'meeting_point': nearest_point
            })
            
            # Point of contiguity is the curb ramp
            curbramp_assignments.append({
                'type': 'one_buffered',
                'edge': edge_b_idx,
                'side': side_b,
                'pos': pos_b,
                'geometry': nearest_point
            })
    
    # Step 7: Handle separate footways (footway=sidewalk with separate geometry)
    # Check for adjacent separate footways or footways wrapping around corners
    for edge_idx, pos in intersection_data['edges']:
        for side in ['left', 'right']:
            geom = edges.loc[edge_idx, f'sidewalk_{side}_geometry']
            buffered = edges.loc[edge_idx, f'sidewalk_{side}_buffered']
            
            # Only process separate footways (not buffered)
            if geom is None or buffered:
                continue
            
            # Check if this is a separate footway
            # (In OSM, these would have footway=sidewalk tag)
            # For now, we use perpendicular projection from intersection node
            coords = list(geom.coords)
            endpoint = Point(coords[0] if pos == 'start' else coords[-1])
            
            # Check if endpoint is near intersection
            if endpoint.distance(node_geom) < 15:
                # Project perpendicular from intersection node onto footway
                projected_point = geom.interpolate(geom.project(node_geom))
                
                curbramp_assignments.append({
                    'type': 'separate_footway',
                    'edge': edge_idx,
                    'side': side,
                    'pos': pos,
                    'geometry': projected_point
                })
    
    return sidewalk_extensions, curbramp_assignments


def _intersection_analysis_worker(args):
    """
    Worker for parallel intersection analysis within a grid chunk.
    Processes a subset of intersection nodes and returns collected
    bikeway extensions, sidewalk extensions, and curb ramp assignments.

    Operates on serializable data only (no GeoDataFrame passed directly).
    """
    (
        node_ids,
        intersection_map_subset,
        edge_data,          # dict: edge_idx -> {col: value, ...} for needed columns
        edge_geometries,    # dict: edge_idx -> Shapely geometry (LineString)
        sanity_buffer,
    ) = args

    all_bikeway_ext = []
    all_sidewalk_ext = []
    all_curbramp_asgn = []

    for node_id in node_ids:
        idata = intersection_map_subset[node_id]
        node_geom = idata['geometry']
        connected_edges = idata['edges']

        # ---- find adjacent corner pairs (inline to avoid GDF dependency) ----
        adjacent_pairs = []
        for i, (ea_idx, pos_a) in enumerate(connected_edges):
            for eb_idx, pos_b in connected_edges[i + 1:]:
                ba = edge_data[ea_idx].get('normalized_bearing')
                bb = edge_data[eb_idx].get('normalized_bearing')
                if ba is None or bb is None:
                    continue
                if pos_a == 'end':
                    ba = (ba + 180) % 360
                if pos_b == 'end':
                    bb = (bb + 180) % 360
                angle_diff = abs(ba - bb)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if 60 <= angle_diff <= 120:
                    cw = (bb - ba) % 360
                    if cw < 180:
                        adjacent_pairs.append((ea_idx, pos_a, eb_idx, pos_b, 'right', 'left'))
                    else:
                        adjacent_pairs.append((ea_idx, pos_a, eb_idx, pos_b, 'left', 'right'))

        # ---- bikeway gaps (Steps 1-3) ----
        for ea_idx, pos_a, eb_idx, pos_b, side_a, side_b in adjacent_pairs:
            geom_a = edge_data[ea_idx].get(f'bikeway_{side_a}_1_geometry')
            geom_b = edge_data[eb_idx].get(f'bikeway_{side_b}_1_geometry')
            if geom_a is None or geom_b is None:
                continue
            buf_a = edge_data[ea_idx].get(f'bikeway_{side_a}_buffered', False)
            buf_b = edge_data[eb_idx].get(f'bikeway_{side_b}_buffered', False)
            if not buf_a and not buf_b:
                continue
            # contiguity check
            try:
                ca = list(geom_a.coords)
                cb = list(geom_b.coords)
                pa = Point(ca[0] if pos_a == 'start' else ca[-1])
                pb = Point(cb[0] if pos_b == 'start' else cb[-1])
                if pa.distance(pb) <= 0.5:
                    continue
            except Exception:
                continue

            if buf_a and buf_b:
                mp_pt = Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
                sg = edge_geometries.get(ea_idx)
                mo = sanity_buffer.get(edge_data[ea_idx].get('osmid'), 10.0)
                if sg and mp_pt.distance(sg) > mo:
                    continue
                all_bikeway_ext.append({
                    'type': 'both_buffered', 'edge_a': ea_idx, 'edge_b': eb_idx,
                    'side_a': side_a, 'side_b': side_b,
                    'pos_a': pos_a, 'pos_b': pos_b, 'meeting_point': mp_pt,
                })
            elif buf_a:
                np_pt = geom_b.interpolate(geom_b.project(pa))
                all_bikeway_ext.append({
                    'type': 'one_buffered', 'edge': ea_idx, 'side': side_a,
                    'pos': pos_a, 'meeting_point': np_pt,
                })
            elif buf_b:
                np_pt = geom_a.interpolate(geom_a.project(pb))
                all_bikeway_ext.append({
                    'type': 'one_buffered', 'edge': eb_idx, 'side': side_b,
                    'pos': pos_b, 'meeting_point': np_pt,
                })

        # ---- sidewalk gaps (Steps 4-7) ----
        for ea_idx, pos_a, eb_idx, pos_b, side_a, side_b in adjacent_pairs:
            geom_a = edge_data[ea_idx].get(f'sidewalk_{side_a}_geometry')
            geom_b = edge_data[eb_idx].get(f'sidewalk_{side_b}_geometry')
            if geom_a is None or geom_b is None:
                continue
            buf_a = edge_data[ea_idx].get(f'sidewalk_{side_a}_buffered', False)
            buf_b = edge_data[eb_idx].get(f'sidewalk_{side_b}_buffered', False)
            try:
                ca = list(geom_a.coords)
                cb = list(geom_b.coords)
                pa = Point(ca[0] if pos_a == 'start' else ca[-1])
                pb = Point(cb[0] if pos_b == 'start' else cb[-1])
            except Exception:
                continue

            if pa.distance(pb) <= 0.5:
                # contiguous — assign curb ramp at endpoint
                all_curbramp_asgn.append({
                    'type': 'contiguous', 'edge_a': ea_idx, 'edge_b': eb_idx,
                    'side_a': side_a, 'side_b': side_b,
                    'pos_a': pos_a, 'pos_b': pos_b, 'geometry': pa,
                })
                continue

            if buf_a and buf_b:
                mp_pt = Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
                sg = edge_geometries.get(ea_idx)
                mo = sanity_buffer.get(edge_data[ea_idx].get('osmid'), 10.0)
                if sg and mp_pt.distance(sg) > mo:
                    continue
                # check bikelane overlap
                overlaps = False
                for bk in ['bikeway_left_1_geometry', 'bikeway_right_1_geometry']:
                    bw = edge_data[ea_idx].get(bk)
                    if bw and mp_pt.distance(bw) < 0.5:
                        overlaps = True
                        break
                if overlaps:
                    continue
                all_sidewalk_ext.append({
                    'type': 'both_buffered', 'edge_a': ea_idx, 'edge_b': eb_idx,
                    'side_a': side_a, 'side_b': side_b,
                    'pos_a': pos_a, 'pos_b': pos_b, 'meeting_point': mp_pt,
                })
                all_curbramp_asgn.append({
                    'type': 'both_buffered', 'edge_a': ea_idx, 'edge_b': eb_idx,
                    'side_a': side_a, 'side_b': side_b,
                    'pos_a': pos_a, 'pos_b': pos_b, 'geometry': mp_pt,
                })
            elif buf_a and not buf_b:
                np_pt = geom_b.interpolate(geom_b.project(pa))
                all_sidewalk_ext.append({
                    'type': 'one_buffered', 'edge': ea_idx, 'side': side_a,
                    'pos': pos_a, 'meeting_point': np_pt,
                })
                all_curbramp_asgn.append({
                    'type': 'one_buffered', 'edge': ea_idx, 'side': side_a,
                    'pos': pos_a, 'geometry': np_pt,
                })
            elif not buf_a and buf_b:
                np_pt = geom_a.interpolate(geom_a.project(pb))
                all_sidewalk_ext.append({
                    'type': 'one_buffered', 'edge': eb_idx, 'side': side_b,
                    'pos': pos_b, 'meeting_point': np_pt,
                })
                all_curbramp_asgn.append({
                    'type': 'one_buffered', 'edge': eb_idx, 'side': side_b,
                    'pos': pos_b, 'geometry': np_pt,
                })

        # ---- Step 7: separate footways ----
        for edge_idx, pos in connected_edges:
            for side in ['left', 'right']:
                geom = edge_data[edge_idx].get(f'sidewalk_{side}_geometry')
                buffered = edge_data[edge_idx].get(f'sidewalk_{side}_buffered', False)
                if geom is None or buffered:
                    continue
                try:
                    coords = list(geom.coords)
                    endpoint = Point(coords[0] if pos == 'start' else coords[-1])
                except Exception:
                    continue
                if endpoint.distance(node_geom) < 15:
                    proj = geom.interpolate(geom.project(node_geom))
                    all_curbramp_asgn.append({
                        'type': 'separate_footway', 'edge': edge_idx,
                        'side': side, 'pos': pos, 'geometry': proj,
                    })

    return all_bikeway_ext, all_sidewalk_ext, all_curbramp_asgn


def _build_edge_data_for_intersection_workers(
    network: gpd.GeoDataFrame,
    needed_edge_indices: Set[int],
) -> Tuple[Dict[int, Dict], Dict[int, LineString]]:
    """
    Extract lightweight dicts from the GeoDataFrame for the columns that
    intersection analysis workers need.  Avoids pickling the full GDF.
    """
    cols_needed = [
        'osmid', 'normalized_bearing',
        'sidewalk_left_geometry', 'sidewalk_right_geometry',
        'sidewalk_left_buffered', 'sidewalk_right_buffered',
        'bikeway_left_1_geometry', 'bikeway_right_1_geometry',
        'bikeway_left_buffered', 'bikeway_right_buffered',
    ]
    # Only include columns that actually exist
    cols_present = [c for c in cols_needed if c in network.columns]

    edge_data: Dict[int, Dict] = {}
    edge_geometries: Dict[int, LineString] = {}

    for idx in needed_edge_indices:
        if idx not in network.index:
            continue
        row_dict = {}
        for c in cols_present:
            row_dict[c] = network.at[idx, c]
        edge_data[idx] = row_dict
        edge_geometries[idx] = network.at[idx, 'geometry']

    return edge_data, edge_geometries


def run_intersection_analysis(
    network: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    sanity_buffer: Dict[int, float],
    curbramp_counter: SequentialIDCounter,
    n_jobs: int = None
) -> gpd.GeoDataFrame:
    """
    Run full intersection analysis (Steps 1-7) for bikelanes and sidewalks.
    Parallelized across grid chunks using the block bounding box.

    Per spec: "Iterating through block corner nodes and parallelizing between
    grid boxes using the blocks bounding box."
    """
    print("  Running parallelized intersection analysis...")

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    intersection_map = build_intersection_index(nodes, network)

    if not intersection_map:
        print("  No intersections found — skipping.")
        return network

    # ---- partition intersections into grid chunks ----
    # Use the network bounding box to build a grid identical to block detection
    grid_cells, grid_metadata = create_grid_bounding_box(network, grid_size=0.01)
    gs = grid_metadata['grid_size']
    minx = grid_metadata['minx']
    miny = grid_metadata['miny']
    n_cols = grid_metadata['n_cols']
    n_rows = grid_metadata['n_rows']

    # Assign each intersection node to a grid cell
    grid_buckets: Dict[Tuple[int, int], List[int]] = {}
    for nid, idata in intersection_map.items():
        pt = idata['geometry']
        col = min(max(int((pt.x - minx) / gs), 0), n_cols - 1)
        row_idx = min(max(int((pt.y - miny) / gs), 0), n_rows - 1)
        grid_buckets.setdefault((col, row_idx), []).append(nid)

    # Collect all edge indices referenced by any intersection
    needed_edges: Set[int] = set()
    for idata in intersection_map.values():
        for eidx, _ in idata['edges']:
            needed_edges.add(eidx)

    # Build lightweight edge data dicts (avoids pickling full GDF)
    edge_data, edge_geometries = _build_edge_data_for_intersection_workers(network, needed_edges)

    # Build worker args — one per non-empty grid cell, sorted SW→NE
    sorted_keys = sorted(grid_buckets.keys(), key=lambda k: (k[0], k[1]))
    worker_args = []
    for key in sorted_keys:
        node_ids = grid_buckets[key]
        subset = {nid: intersection_map[nid] for nid in node_ids}
        worker_args.append((node_ids, subset, edge_data, edge_geometries, sanity_buffer))

    print(f"  Processing {len(intersection_map)} intersections across {len(worker_args)} grid chunks ({n_jobs} workers)...")

    # ---- parallel execution ----
    all_bikeway_extensions = []
    all_sidewalk_extensions = []
    all_curbramp_assignments = []

    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(_intersection_analysis_worker, worker_args,
                      chunksize=max(1, len(worker_args) // (n_jobs * 4))),
            total=len(worker_args),
            desc="  Intersections (parallel)"
        ))

    for bw_ext, sw_ext, cr_asgn in results:
        all_bikeway_extensions.extend(bw_ext)
        all_sidewalk_extensions.extend(sw_ext)
        all_curbramp_assignments.extend(cr_asgn)

    print(f"  Found {len(all_bikeway_extensions)} bikeway gaps to fix")
    print(f"  Found {len(all_sidewalk_extensions)} sidewalk gaps to fix")
    print(f"  Generated {len(all_curbramp_assignments)} curb ramp assignments")
    
    # Initialize curb ramp columns if they don't exist
    for side in ['left', 'right']:
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                col_id = f'sidewalk_{side}_curbramp_{slot}_{pos}_ID'
                col_geom = f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry'
                col_public = f'public_data_id_sidewalk_{side}_curbramp_{slot}_{pos}'
                
                if col_id not in network.columns:
                    network[col_id] = None
                if col_geom not in network.columns:
                    network[col_geom] = None
                if col_public not in network.columns:
                    network[col_public] = None
    
    # Apply bikeway extensions (Steps 2-3)
    if len(all_bikeway_extensions) > 0:
        print(f"  Applying {len(all_bikeway_extensions)} bikeway extensions...")
        
        for ext in all_bikeway_extensions:
            if ext['type'] == 'both_buffered':
                # Extend both bikeways to meeting point
                edge_a = ext['edge_a']
                edge_b = ext['edge_b']
                side_a = ext['side_a']
                side_b = ext['side_b']
                pos_a = ext['pos_a']
                pos_b = ext['pos_b']
                meeting_point = ext['meeting_point']
                
                # Extend edge A
                geom_a = network.loc[edge_a, f'bikeway_{side_a}_1_geometry']
                if geom_a:
                    coords_a = list(geom_a.coords)
                    if pos_a == 'end':
                        new_coords_a = coords_a + [meeting_point.coords[0]]
                    else:
                        new_coords_a = [meeting_point.coords[0]] + coords_a
                    network.at[edge_a, f'bikeway_{side_a}_1_geometry'] = LineString(new_coords_a)
                
                # Extend edge B
                geom_b = network.loc[edge_b, f'bikeway_{side_b}_1_geometry']
                if geom_b:
                    coords_b = list(geom_b.coords)
                    if pos_b == 'end':
                        new_coords_b = coords_b + [meeting_point.coords[0]]
                    else:
                        new_coords_b = [meeting_point.coords[0]] + coords_b
                    network.at[edge_b, f'bikeway_{side_b}_1_geometry'] = LineString(new_coords_b)
            
            elif ext['type'] == 'one_buffered':
                # Extend buffered bikeway to meeting point
                edge = ext['edge']
                side = ext['side']
                pos = ext['pos']
                meeting_point = ext['meeting_point']
                
                geom = network.loc[edge, f'bikeway_{side}_1_geometry']
                if geom:
                    coords = list(geom.coords)
                    if pos == 'end':
                        new_coords = coords + [meeting_point.coords[0]]
                    else:
                        new_coords = [meeting_point.coords[0]] + coords
                    network.at[edge, f'bikeway_{side}_1_geometry'] = LineString(new_coords)
    
    # Apply sidewalk extensions (Steps 5-6)
    if len(all_sidewalk_extensions) > 0:
        print(f"  Applying {len(all_sidewalk_extensions)} sidewalk extensions...")
        
        for ext in all_sidewalk_extensions:
            if ext['type'] == 'both_buffered':
                # Extend both sidewalks to meeting point
                edge_a = ext['edge_a']
                edge_b = ext['edge_b']
                side_a = ext['side_a']
                side_b = ext['side_b']
                pos_a = ext['pos_a']
                pos_b = ext['pos_b']
                meeting_point = ext['meeting_point']
                
                # Extend edge A
                geom_a = network.loc[edge_a, f'sidewalk_{side_a}_geometry']
                if geom_a:
                    coords_a = list(geom_a.coords)
                    if pos_a == 'end':
                        new_coords_a = coords_a + [meeting_point.coords[0]]
                    else:
                        new_coords_a = [meeting_point.coords[0]] + coords_a
                    network.at[edge_a, f'sidewalk_{side_a}_geometry'] = LineString(new_coords_a)
                
                # Extend edge B
                geom_b = network.loc[edge_b, f'sidewalk_{side_b}_geometry']
                if geom_b:
                    coords_b = list(geom_b.coords)
                    if pos_b == 'end':
                        new_coords_b = coords_b + [meeting_point.coords[0]]
                    else:
                        new_coords_b = [meeting_point.coords[0]] + coords_b
                    network.at[edge_b, f'sidewalk_{side_b}_geometry'] = LineString(new_coords_b)
            
            elif ext['type'] == 'one_buffered':
                # Extend buffered sidewalk to meeting point
                edge = ext['edge']
                side = ext['side']
                pos = ext['pos']
                meeting_point = ext['meeting_point']
                
                geom = network.loc[edge, f'sidewalk_{side}_geometry']
                if geom:
                    coords = list(geom.coords)
                    if pos == 'end':
                        new_coords = coords + [meeting_point.coords[0]]
                    else:
                        new_coords = [meeting_point.coords[0]] + coords
                    network.at[edge, f'sidewalk_{side}_geometry'] = LineString(new_coords)
    
    # Apply curb ramp assignments (Steps 4-7)
    if len(all_curbramp_assignments) > 0:
        print(f"  Applying {len(all_curbramp_assignments)} curb ramp assignments...")
        
        for assignment in all_curbramp_assignments:
            if assignment['type'] == 'contiguous':
                # Step 4: Contiguous sidewalks - populate curb ramp with endpoint
                edge_a = assignment['edge_a']
                edge_b = assignment['edge_b']
                side_a = assignment['side_a']
                side_b = assignment['side_b']
                pos_a = assignment['pos_a']
                pos_b = assignment['pos_b']
                geometry = assignment['geometry']
                
                # Assign to both edges
                ramp_id = curbramp_counter.next()
                
                network.at[edge_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_ID'] = ramp_id
                network.at[edge_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_geometry'] = geometry
                
                network.at[edge_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_ID'] = ramp_id
                network.at[edge_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_geometry'] = geometry
            
            elif assignment['type'] == 'both_buffered':
                # Step 5: Both buffered - assign meeting point as curb ramp
                edge_a = assignment['edge_a']
                edge_b = assignment['edge_b']
                side_a = assignment['side_a']
                side_b = assignment['side_b']
                pos_a = assignment['pos_a']
                pos_b = assignment['pos_b']
                geometry = assignment['geometry']
                
                # Assign to both edges
                ramp_id = curbramp_counter.next()
                
                network.at[edge_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_ID'] = ramp_id
                network.at[edge_a, f'sidewalk_{side_a}_curbramp_{pos_a}_1_geometry'] = geometry
                
                network.at[edge_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_ID'] = ramp_id
                network.at[edge_b, f'sidewalk_{side_b}_curbramp_{pos_b}_1_geometry'] = geometry
            
            elif assignment['type'] == 'one_buffered':
                # Step 6: One buffered - point of contiguity is curb ramp
                edge = assignment['edge']
                side = assignment['side']
                pos = assignment['pos']
                geometry = assignment['geometry']
                
                ramp_id = curbramp_counter.next()
                
                network.at[edge, f'sidewalk_{side}_curbramp_{pos}_1_ID'] = ramp_id
                network.at[edge, f'sidewalk_{side}_curbramp_{pos}_1_geometry'] = geometry
            
            elif assignment['type'] == 'separate_footway':
                # Step 7: Separate footway - perpendicular projection
                edge = assignment['edge']
                side = assignment['side']
                pos = assignment['pos']
                geometry = assignment['geometry']
                
                ramp_id = curbramp_counter.next()
                
                network.at[edge, f'sidewalk_{side}_curbramp_{pos}_1_ID'] = ramp_id
                network.at[edge, f'sidewalk_{side}_curbramp_{pos}_1_geometry'] = geometry
    
    return network


# ============================================================================
# SECTION 14: STAGE 10 — GOVERNMENT CURB RAMP PROCESSING (Steps 8-12)
# ============================================================================

def load_government_curb_ramps(city_config: CityConfig) -> Optional[gpd.GeoDataFrame]:
    """
    Load government curb ramp data if provided.
    Returns GeoDataFrame with curb ramp points and attributes.
    """
    path = city_config.government_data_paths.curb_ramps
    if path is None or path == '':
        print("  No curb ramp file specified")
        return None

    if not os.path.exists(path):
        print(f"  ⚠ Curb ramp file not found: {path}")
        return None

    print(f"  Loading curb ramps from {path}...")

    # Load using the standard geospatial loader
    gdf = load_geospatial_file(path)

    if gdf is None or gdf.empty:
        print(f"  ⚠ Could not load curb ramps from {path}")
        return None

    print(f"  Loaded {len(gdf)} curb ramp records")
    return gdf


def process_government_curb_ramps(
    network: gpd.GeoDataFrame,
    gov_curb_ramps: gpd.GeoDataFrame,
    city_config: CityConfig,
    global_config: GlobalConfig,
    curbramp_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Process government curb ramp data according to Steps 8-12.
    
    Step 8: If ramp within inner buffer, record public data ID and attributes but use default geometry
    Step 9: If ramp outside inner but inside outer buffer, flag for replacement/splitting
    Step 10: Replace single curb point and snap sidewalk segments
    Step 11: Split one ramp into 2, create orphan segment for curb return
    Step 12: Assign additional ramps to slots 2 and 3
    """
    if gov_curb_ramps is None or gov_curb_ramps.empty:
        return network
    
    if not city_config.curbramp_trustworthy:
        print("  Curb ramps marked as untrustworthy - skipping government curb ramp processing")
        return network
    
    print(f"  Processing {len(gov_curb_ramps)} government curb ramps...")
    
    inner_buffer = global_config.curb_ramp_trustworthiness_inner_buffer
    outer_buffer = global_config.curb_ramp_trustworthiness_outer_buffer
    
    print(f"  Inner buffer: {inner_buffer}m, Outer buffer: {outer_buffer}m")
    
    # Working CRS is already UTM — distances are in meters.
    # Ensure government ramps are in the same CRS.
    if gov_curb_ramps.crs != network.crs:
        gov_curb_ramps = gov_curb_ramps.to_crs(network.crs)
    
    # Column mappings
    col_map = city_config.column_mappings
    ramp_id_col = col_map.curbramp_id or 'id'
    return_loc_col = col_map.curbramp_return_loc or 'return_loc'
    return_pos_col = col_map.curbramp_position or 'return_position'
    condition_col = col_map.curbramp_condition or 'condition_score'
    
    # Process each default curb ramp in the network
    processed_count = 0
    inner_buffer_count = 0
    goldilocks_count = 0
    replaced_count = 0
    split_count = 0
    
    for idx in tqdm(network.index, desc="  Processing curb ramps"):
        for side in ['left', 'right']:
            for slot in ['start', 'end']:
                # Check if there's a default curb ramp at this slot
                default_ramp_id = network.loc[idx, f'sidewalk_{side}_curbramp_{slot}_1_ID']
                default_ramp_geom = network.loc[idx, f'sidewalk_{side}_curbramp_{slot}_1_geometry']
                
                if default_ramp_geom is None or pd.isna(default_ramp_id):
                    continue
                
                # Find government ramps within outer buffer (already in meters)
                nearby_ramps = []
                for ramp_idx, ramp_row in gov_curb_ramps.iterrows():
                    ramp_geom = ramp_row['geometry']
                    distance = default_ramp_geom.distance(ramp_geom)
                    
                    if distance <= outer_buffer:
                        nearby_ramps.append({
                            'distance': distance,
                            'geometry': ramp_row['geometry'],
                            'id': ramp_row.get(ramp_id_col, str(ramp_idx)),
                            'return_loc': ramp_row.get(return_loc_col, None),
                            'return_position': ramp_row.get(return_pos_col, None),
                            'condition_score': ramp_row.get(condition_col, None)
                        })
                
                if len(nearby_ramps) == 0:
                    continue
                
                # Sort by distance
                nearby_ramps.sort(key=lambda x: x['distance'])
                
                # Step 8: Inner buffer - record attributes but keep default geometry
                if nearby_ramps[0]['distance'] <= inner_buffer:
                    ramp = nearby_ramps[0]
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_1'] = ramp['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnloc'] = ramp['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnposition'] = ramp['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_condition_score'] = ramp['condition_score']
                    # Keep default geometry (don't replace)
                    inner_buffer_count += 1
                    processed_count += 1
                    continue
                
                # Step 9: Goldilocks zone (outside inner, inside outer)
                goldilocks_ramps = [r for r in nearby_ramps if inner_buffer < r['distance'] <= outer_buffer]
                
                if len(goldilocks_ramps) == 0:
                    continue
                
                goldilocks_count += 1
                
                if len(goldilocks_ramps) == 1:
                    # Step 10: Replace single curb point
                    ramp = goldilocks_ramps[0]
                    
                    # Replace geometry
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_geometry'] = ramp['geometry']
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_1'] = ramp['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnloc'] = ramp['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnposition'] = ramp['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_condition_score'] = ramp['condition_score']
                    
                    # Snap sidewalk segment to new point
                    sidewalk_geom = network.loc[idx, f'sidewalk_{side}_geometry']
                    if sidewalk_geom and not sidewalk_geom.is_empty:
                        coords = list(sidewalk_geom.coords)
                        if slot == 'start':
                            coords[0] = ramp['geometry'].coords[0]
                        else:
                            coords[-1] = ramp['geometry'].coords[0]
                        network.at[idx, f'sidewalk_{side}_geometry'] = LineString(coords)
                    
                    replaced_count += 1
                    processed_count += 1
                
                elif len(goldilocks_ramps) >= 2:
                    # Step 11: Split one ramp into 2
                    ramp_1 = goldilocks_ramps[0]
                    ramp_2 = goldilocks_ramps[1]
                    
                    sidewalk_geom = network.loc[idx, f'sidewalk_{side}_geometry']
                    if sidewalk_geom is None or sidewalk_geom.is_empty:
                        continue
                    
                    # Project both ramps onto sidewalk
                    proj_dist_1 = sidewalk_geom.project(ramp_1['geometry'])
                    proj_dist_2 = sidewalk_geom.project(ramp_2['geometry'])
                    
                    # Ensure ramp_1 is closer to start
                    if proj_dist_1 > proj_dist_2:
                        proj_dist_1, proj_dist_2 = proj_dist_2, proj_dist_1
                        ramp_1, ramp_2 = ramp_2, ramp_1
                    
                    # Get projected points on sidewalk
                    pt_1 = sidewalk_geom.interpolate(proj_dist_1)
                    pt_2 = sidewalk_geom.interpolate(proj_dist_2)
                    
                    # Create orphan segment (curb return) between the two ramps
                    orphan_geom = LineString([pt_1.coords[0], pt_2.coords[0]])
                    network.at[idx, 'curb_return_geometry'] = orphan_geom
                    
                    # Split sidewalk into segments
                    # For simplicity, we'll snap the sidewalk endpoints to the new ramp locations
                    coords = list(sidewalk_geom.coords)
                    
                    if slot == 'start':
                        # Adjust start of sidewalk
                        coords[0] = ramp_1['geometry'].coords[0]
                    else:
                        # Adjust end of sidewalk
                        coords[-1] = ramp_2['geometry'].coords[0]
                    
                    network.at[idx, f'sidewalk_{side}_geometry'] = LineString(coords)
                    
                    # Assign ramp 1
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_ID'] = curbramp_counter.next()
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_geometry'] = ramp_1['geometry']
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_1'] = ramp_1['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnloc'] = ramp_1['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_returnposition'] = ramp_1['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_1_condition_score'] = ramp_1['condition_score']
                    
                    # Assign ramp 2
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_ID'] = curbramp_counter.next()
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_geometry'] = ramp_2['geometry']
                    network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_2'] = ramp_2['id']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_returnloc'] = ramp_2['return_loc']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_returnposition'] = ramp_2['return_position']
                    network.at[idx, f'sidewalk_{side}_curbramp_{slot}_2_condition_score'] = ramp_2['condition_score']
                    
                    split_count += 1
                    processed_count += 1
                    
                    # Step 12: Assign additional ramps to slot 3
                    if len(goldilocks_ramps) >= 3:
                        ramp_3 = goldilocks_ramps[2]
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_ID'] = curbramp_counter.next()
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_geometry'] = ramp_3['geometry']
                        network.at[idx, f'public_data_id_sidewalk_{side}_curbramp_{slot}_3'] = ramp_3['id']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_returnloc'] = ramp_3['return_loc']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_returnposition'] = ramp_3['return_position']
                        network.at[idx, f'sidewalk_{side}_curbramp_{slot}_3_condition_score'] = ramp_3['condition_score']
    
    print(f"  Processed {processed_count} curb ramp slots:")
    print(f"    - {inner_buffer_count} within inner buffer (attributes only)")
    print(f"    - {replaced_count} replaced (single ramp in goldilocks zone)")
    print(f"    - {split_count} split (multiple ramps in goldilocks zone)")
    
    return network


# ============================================================================
# SECTION 14.5: CURB RAMP LOADING, ASSIGNMENT & CURB RETURN EXTRACTION
# ============================================================================

@dataclass
class CurbRamp:
    """
    Data model for curb ramp locations and attributes.
    """
    loc_id: str
    cnn: str
    curb_return_loc: str
    position_on_return: str
    latitude: float
    longitude: float
    condition_score: float
    geometry: Point


def _determine_ramp_slot(
    ramp: CurbRamp,
    segment: pd.Series,
    segment_bearing: float
) -> Optional[str]:
    """
    Determine which slot a curb ramp should be assigned to based on proximity
    and bearing analysis.

    Returns slot name ('left_start', 'left_end', 'right_start', 'right_end')
    or None if ramp is mid-block or unresolvable.
    """
    curb_loc = ramp.curb_return_loc.upper()
    position = ramp.position_on_return.upper()

    if position == 'MB':
        return None

    segment_geom = segment['geometry']
    coords = list(segment_geom.coords)
    start_point = Point(coords[0])
    end_point = Point(coords[-1])

    dist_to_start = ramp.geometry.distance(start_point)
    dist_to_end = ramp.geometry.distance(end_point)
    slot_position = 'start' if dist_to_start < dist_to_end else 'end'

    curb_bearing_map = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315
    }
    curb_bearing = curb_bearing_map.get(curb_loc, None)

    if curb_bearing is None:
        slot_side = 'left' if position == 'LEFT' else ('right' if position == 'RIGHT' else 'left')
    else:
        if position == 'LEFT':
            slot_side = 'left'
        elif position == 'RIGHT':
            slot_side = 'right'
        else:
            bearing_to_curb = (curb_bearing - segment_bearing) % 360
            slot_side = 'left' if bearing_to_curb < 180 else 'right'

    return f"{slot_side}_{slot_position}"


def _assign_curb_ramps_sequential(
    network: gpd.GeoDataFrame,
    network_projected: gpd.GeoDataFrame,
    curb_ramps: List[CurbRamp],
    tree: cKDTree,
    tree_indices: List,
    search_radius: float
) -> gpd.GeoDataFrame:
    """Sequential (single-threaded) curb ramp assignment."""
    curb_ramp_data = {
        f'sidewalk_{side}_curbramp_{slot}_{position}_{attr}': {}
        for side in ['left', 'right']
        for slot in ['start', 'end']
        for position in [1, 2, 3]
        for attr in ['locID', 'condition_score', 'geometry']
    }
    slot_assignments = {}
    assigned_count = 0

    for ramp in tqdm(curb_ramps, desc="  → Assigning curb ramps", leave=False):
        ramp_gdf = gpd.GeoDataFrame([{'geometry': ramp.geometry}], crs="EPSG:4326")
        ramp_projected = ramp_gdf.to_crs("EPSG:3857")
        ramp_point = ramp_projected.geometry.iloc[0]
        ramp_coords = (ramp_point.x, ramp_point.y)

        indices = tree.query_ball_point(ramp_coords, r=search_radius)
        if not indices:
            continue

        nearby_indices = list(set(tree_indices[i] for i in indices))

        for idx in nearby_indices:
            segment = network.loc[idx]
            segment_bearing = segment.get('bearing', 0.0)
            slot = _determine_ramp_slot(ramp, segment, segment_bearing)
            if slot is None:
                continue

            if idx not in slot_assignments:
                slot_assignments[idx] = {}
            if slot not in slot_assignments[idx]:
                slot_assignments[idx][slot] = []
            if len(slot_assignments[idx][slot]) >= 3:
                continue

            position = len(slot_assignments[idx][slot]) + 1
            slot_assignments[idx][slot].append(position)

            parts = slot.split('_')
            side, slot_position = parts[0], parts[1]

            curb_ramp_data[f'sidewalk_{side}_curbramp_{slot_position}_{position}_locID'][idx] = ramp.loc_id
            curb_ramp_data[f'sidewalk_{side}_curbramp_{slot_position}_{position}_condition_score'][idx] = ramp.condition_score
            curb_ramp_data[f'sidewalk_{side}_curbramp_{slot_position}_{position}_geometry'][idx] = geom_to_wkb(ramp.geometry)
            assigned_count += 1
            break

    curb_ramp_series = {col: pd.Series(data, index=network.index) for col, data in curb_ramp_data.items()}
    curb_ramp_df = pd.DataFrame(curb_ramp_series, index=network.index)
    network = pd.concat([network, curb_ramp_df], axis=1)

    if not isinstance(network, gpd.GeoDataFrame):
        network = gpd.GeoDataFrame(network, geometry='geometry', crs=network_projected.crs)

    return network


def load_curb_ramps(city_config: CityConfig) -> List:
    """
    Load curb ramps from the configured file path into CurbRamp objects.

    Args:
        city_config: CityConfig with government_data_paths.curb_ramps

    Returns:
        List of CurbRamp objects (or empty list if not available)
    """
    path = city_config.government_data_paths.curb_ramps
    if path is None or path == '':
        print("  No curb ramp file specified")
        return []

    if not os.path.exists(path):
        print(f"  ⚠ Curb ramp file not found: {path}")
        return []

    try:
        print(f"  Loading curb ramps from: {path}")
        df = pd.read_csv(path)

        curb_ramps = []
        for _, row in df.iterrows():
            if pd.isna(row.get('Latitude')) or pd.isna(row.get('Longitude')):
                continue
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
            except (ValueError, TypeError):
                continue

            curb_ramps.append(CurbRamp(
                loc_id=str(row.get('LocID', '')),
                cnn=str(row.get('CNN', '')),
                curb_return_loc=str(row.get('CurbReturnLoc', '')),
                position_on_return=str(row.get('PositionOnReturn', '')),
                latitude=lat,
                longitude=lon,
                condition_score=float(row.get('conditionScore', -2)),
                geometry=Point(lon, lat),
            ))

        print(f"  Loaded {len(curb_ramps)} curb ramps with valid coordinates")
        return curb_ramps
    except Exception as e:
        print(f"  Error loading curb ramps: {e}")
        return []


def assign_curb_ramps(
    edges: gpd.GeoDataFrame,
    curb_ramps: List,
    outer_buffer: float,
    inner_buffer: float,
    global_config: GlobalConfig,
    counter: SequentialIDCounter,
    trustworthy: bool = False
) -> gpd.GeoDataFrame:
    """
    Assign curb ramps to street segments using KD-tree spatial matching.

    Each segment has 4 slots (left_start, left_end, right_start, right_end),
    each holding up to 3 curb ramps.

    Args:
        edges: GeoDataFrame with street network
        curb_ramps: List of CurbRamp objects
        outer_buffer: Outer buffer distance (used as search radius)
        inner_buffer: Inner buffer distance (unused in spatial assignment)
        global_config: Global configuration
        counter: Sequential ID counter (unused in spatial assignment)
        trustworthy: Whether curb ramp data is trustworthy

    Returns:
        GeoDataFrame with assigned curb ramps
    """
    if len(curb_ramps) == 0:
        return edges

    search_radius = outer_buffer if outer_buffer > 0 else 25.0
    print(f"  Assigning {len(curb_ramps)} curb ramps (search radius: {search_radius}m)")

    # Build KD-tree from projected segment endpoints
    network_projected = edges.to_crs("EPSG:3857")
    tree_coords = []
    tree_indices = []

    for idx, row in network_projected.iterrows():
        geom = row['geometry']
        coords_list = list(geom.coords)
        tree_coords.append(coords_list[0])
        tree_indices.append(idx)
        tree_coords.append(coords_list[-1])
        tree_indices.append(idx)

    tree = cKDTree(tree_coords)

    try:
        return _assign_curb_ramps_sequential(
            edges, network_projected, curb_ramps, tree, tree_indices, search_radius
        )
    except Exception as e:
        print(f"  Error assigning curb ramps: {e}")
        traceback.print_exc()
        return edges


def extract_curb_return_geometries(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract curb return geometries from intersection endpoints.

    Identifies short footway-like connections at segment endpoints and assigns
    them to the curb_return_end_geometry column.

    Args:
        edges: GeoDataFrame with street network

    Returns:
        GeoDataFrame with curb_return_end_geometry column populated
    """
    print("  Extracting curb return geometries...")

    if 'curb_return_end_geometry' not in edges.columns:
        edges['curb_return_end_geometry'] = None

    try:
        network_projected = edges.to_crs("EPSG:3857")

        # Build KD-tree from segment endpoints
        tree_coords = []
        tree_indices = []

        for idx, row in network_projected.iterrows():
            geom = row['geometry']
            coords_list = list(geom.coords)
            tree_coords.append(coords_list[-1])  # end points only
            tree_indices.append(idx)

        tree = cKDTree(tree_coords)
        search_radius = 15.0  # meters
        assigned_count = 0

        # For each segment endpoint, look for nearby endpoints from other segments
        # that could form a curb return
        for i, (idx, row) in enumerate(network_projected.iterrows()):
            end_coord = list(row['geometry'].coords)[-1]
            nearby = tree.query_ball_point(end_coord, r=search_radius)

            # Filter to other segments
            nearby_other = [j for j in nearby if tree_indices[j] != idx]
            if len(nearby_other) < 1:
                continue

            # If multiple segments meet at this endpoint, it's an intersection
            # with potential curb returns — mark it
            if pd.isna(edges.loc[idx, 'curb_return_end_geometry']):
                # Create a small geometry representing the curb return area
                other_idx = tree_indices[nearby_other[0]]
                other_end = list(network_projected.loc[other_idx, 'geometry'].coords)[-1]

                # Only create curb return if endpoints are close but not identical
                dist = Point(end_coord).distance(Point(other_end))
                if 0.5 < dist < search_radius:
                    # Build curb return line in WGS84
                    end_4326 = list(edges.loc[idx, 'geometry'].coords)[-1]
                    other_4326 = list(edges.loc[other_idx, 'geometry'].coords)[-1]
                    curb_return_geom = LineString([end_4326, other_4326])
                    edges.at[idx, 'curb_return_end_geometry'] = geom_to_wkb(curb_return_geom)
                    assigned_count += 1

        print(f"  → Assigned {assigned_count} curb return geometries")
    except Exception as e:
        print(f"  Error extracting curb return geometries: {e}")
        traceback.print_exc()

    return edges


# ============================================================================
# SECTION 15: STAGE 11 — IMPLICIT CROSSWALK GEOMETRY (Step 21, new)
# ============================================================================

def generate_implicit_crosswalk_geometries(
    network: gpd.GeoDataFrame,
    crosswalk_counter: SequentialIDCounter
) -> gpd.GeoDataFrame:
    """
    Generate implicit crosswalk geometries where curb ramps exist but no explicit crosswalk (Step 21).
    Optimized with vectorized operations.
    """
    print("  Generating implicit crosswalk geometries...")
    
    # Pre-extract all columns we need to avoid repeated lookups
    updates = {}  # idx -> {col: value}
    
    for slot in ['start', 'end']:
        # Check if required columns exist
        required_cols = [
            f'crosswalk_{slot}_type',
            f'sidewalk_left_curbramp_{slot}_1_geometry',
            f'sidewalk_right_curbramp_{slot}_1_geometry',
            f'sidewalk_left_curbramp_{slot}_1_ID',
            f'public_data_id_sidewalk_left_curbramp_{slot}_1',
            f'sidewalk_right_curbramp_{slot}_1_ID',
            f'public_data_id_sidewalk_right_curbramp_{slot}_1'
        ]
        
        missing_cols = [col for col in required_cols if col not in network.columns]
        if missing_cols:
            print(f"  ⚠ Skipping {slot} slot - missing columns: {missing_cols}")
            continue
        
        # Vectorized checks for existing crosswalks
        has_crosswalk = network[f'crosswalk_{slot}_type'].notna()
        
        # Check for curb ramps on both sides
        left_ramp_geom = network[f'sidewalk_left_curbramp_{slot}_1_geometry']
        right_ramp_geom = network[f'sidewalk_right_curbramp_{slot}_1_geometry']
        
        left_has_attrs = (
            network[f'sidewalk_left_curbramp_{slot}_1_ID'].notna() |
            network[f'public_data_id_sidewalk_left_curbramp_{slot}_1'].notna()
        )
        right_has_attrs = (
            network[f'sidewalk_right_curbramp_{slot}_1_ID'].notna() |
            network[f'public_data_id_sidewalk_right_curbramp_{slot}_1'].notna()
        )
        
        # Find rows that need implicit crosswalks
        # Step 21: Only generate if no explicit crosswalk AND no existing geometry
        has_geometry = network[f'crosswalk_{slot}_geometry'].notna()
        needs_implicit = (
            ~has_crosswalk &
            ~has_geometry &  # Don't override explicit crossing geometries
            left_ramp_geom.notna() &
            right_ramp_geom.notna() &
            left_has_attrs &
            right_has_attrs
        )
        
        # Process only the rows that need implicit crosswalks
        for idx in tqdm(network[needs_implicit].index, desc=f"  Implicit crosswalks ({slot})", leave=False):
            left_geom = network.at[idx, f'sidewalk_left_curbramp_{slot}_1_geometry']
            right_geom = network.at[idx, f'sidewalk_right_curbramp_{slot}_1_geometry']
            
            # Draw LineString between left and right ramp points
            crosswalk_geom = LineString([
                left_geom.coords[0],
                right_geom.coords[0]
            ])
            
            if idx not in updates:
                updates[idx] = {}
            updates[idx][f'crosswalk_{slot}_id'] = crosswalk_counter.next()
            updates[idx][f'crosswalk_{slot}_geometry'] = crosswalk_geom
    
    # Batch apply all updates
    for idx, cols in updates.items():
        for col, value in cols.items():
            network.at[idx, col] = value
    
    return network


# ============================================================================
# SECTION 16: STAGE 12 — SCHEMA ASSEMBLY AND EXPORT (major rewrite)
# ============================================================================

def assemble_network_schema(network: gpd.GeoDataFrame, global_config: GlobalConfig, nodes: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """
    Assemble final network schema with all 152+ columns in spec order.
    
    Args:
        network: The edges GeoDataFrame
        global_config: Global configuration
        nodes: Optional nodes GeoDataFrame for populating start/end node geometries
    """
    print("  Assembling network schema...")
    
    # Define all columns in spec order
    columns = [
        # Block and street identification
        'block_ids',
        'street_osmid',
        'block_sides',
        'public_data_id_street',
        'start_node_osmid',
        'start_node_is_block_node',
        'end_node_osmid',
        'end_node_is_block_node',
        'public_data_id_start_end_nodes',
        
        # Street attributes
        'normalized_bearing',
        'name',
        'highway',
        'maxspeed',
        'oneway',
        'lanes',
        'lane_width',
        'surface',
        
        # Street features
        'street_feature_types',
        'public_data_id_street_feature',
        'street_feature_geometry',
        'street_feature_geometry_projected',
    ]
    
    # Sidewalk left (44 columns)
    sidewalk_left_cols = [
        'sidewalk_left_ID',
        'sidewalk_left_block_ID',
        'sidewalk_left_presence',
        'public_data_id_sidewalk_left',
        'sidewalk_left_surface',
        'sidewalk_left_quality',
        'sidewalk_left_width',
        'sidewalk_left_incline',
        'sidewalk_left_buffered',
    ]
    
    # Curb ramps left (6 slots × 6 attrs = 36 columns)
    for slot in ['start', 'end']:
        for pos in [1, 2, 3]:
            sidewalk_left_cols.extend([
                f'sidewalk_left_curbramp_{slot}_{pos}_ID',
                f'public_data_id_sidewalk_left_curbramp_{slot}_{pos}',
                f'sidewalk_left_curbramp_{slot}_{pos}_returnloc',
                f'sidewalk_left_curbramp_{slot}_{pos}_returnposition',
                f'sidewalk_left_curbramp_{slot}_{pos}_condition_score',
                f'sidewalk_left_curbramp_{slot}_{pos}_geometry',
            ])
    
    # Sidewalk left features
    sidewalk_left_cols.extend([
        'sidewalk_left_feature_ids',
        'sidewalk_left_feature_types',
        'public_data_id_sidewalk_left_feature',
        'sidewalk_left_feature_geometry',
        'sidewalk_left_feature_geometry_projected',
    ])
    
    columns.extend(sidewalk_left_cols)
    
    # Sidewalk right (44 columns) - same structure as left
    sidewalk_right_cols = [
        'sidewalk_right_ID',
        'sidewalk_right_block_ID',
        'sidewalk_right_presence',
        'public_data_id_sidewalk_right',
        'sidewalk_right_surface',
        'sidewalk_right_quality',
        'sidewalk_right_width',
        'sidewalk_right_incline',
        'sidewalk_right_buffered',
    ]
    
    for slot in ['start', 'end']:
        for pos in [1, 2, 3]:
            sidewalk_right_cols.extend([
                f'sidewalk_right_curbramp_{slot}_{pos}_ID',
                f'public_data_id_sidewalk_right_curbramp_{slot}_{pos}',
                f'sidewalk_right_curbramp_{slot}_{pos}_returnloc',
                f'sidewalk_right_curbramp_{slot}_{pos}_returnposition',
                f'sidewalk_right_curbramp_{slot}_{pos}_condition_score',
                f'sidewalk_right_curbramp_{slot}_{pos}_geometry',
            ])
    
    sidewalk_right_cols.extend([
        'sidewalk_right_feature_ids',
        'sidewalk_right_feature_types',
        'public_data_id_sidewalk_right_feature',
        'sidewalk_right_feature_geometry',
        'sidewalk_right_feature_geometry_projected',
    ])
    
    columns.extend(sidewalk_right_cols)
    
    # Crosswalks (2 × 15 = 30 columns)
    for slot in ['start', 'end']:
        columns.extend([
            f'crosswalk_{slot}_id',
            f'crosswalk_{slot}_block_ids',
            f'crosswalk_{slot}_type',
            f'public_data_id_crosswalk_{slot}',
            f'crosswalk_{slot}_controlled',
            f'crosswalk_{slot}_marked',
            f'crosswalk_{slot}_markings',
            f'crosswalk_{slot}_signals',
            f'crosswalk_{slot}_island',
            f'crosswalk_{slot}_kerb',
            f'crosswalk_{slot}_tactile_paving',
            f'crosswalk_{slot}_traffic_calming',
            f'crosswalk_{slot}_continuous',
            f'crosswalk_{slot}_condition',
            f'crosswalk_{slot}_geometry',
        ])
    
    # Bikeways left (2 lanes × 8 attrs + buffered + 2 × 5 features = 27 columns)
    columns.extend([
        'bikeway_left_1_id',
        'bikeway_left_1_block_id',
        'public_data_id_bikeway_left_1',
        'bikeway_left_1_type',
        'bikeway_left_1_surface',
        'bikeway_left_1_quality',
        'bikeway_left_1_permitted',
        'bikeway_left_1_width',
        'bikeway_left_1_incline',
        'bikeway_left_buffered',
        'bikeway_left_2_id',
        'bikeway_left_2_block_id',
        'public_data_id_bikeway_left_2',
        'bikeway_left_2_type',
        'bikeway_left_2_surface',
        'bikeway_left_2_quality',
        'bikeway_left_2_permitted',
        'bikeway_left_2_width',
        'bikeway_left_2_incline',
        'bikeway_left_1_feature_ids',
        'bikeway_left_1_feature_types',
        'public_data_id_bikeway_left_1_features',
        'bikeway_left_1_feature_geometry',
        'bikeway_left_1_feature_geometry_projected',
        'bikeway_left_2_feature_types',
        'public_data_id_bikeway_left_2_features',
        'bikeway_left_2_feature_geometry',
        'bikeway_left_2_feature_geometry_projected',
    ])
    
    # Bikeways right (columns per spec) - same structure as left
    columns.extend([
        'bikeway_right_1_id',
        'bikeway_right_1_block_id',
        'public_data_id_bikeway_right_1',
        'bikeway_right_1_type',
        'bikeway_right_1_surface',
        'bikeway_right_1_quality',
        'bikeway_right_1_permitted',
        'bikeway_right_1_width',
        'bikeway_right_1_incline',
        'bikeway_right_buffered',
        'bikeway_right_2_id',
        'bikeway_right_2_block_id',
        'public_data_id_bikeway_right_2',
        'bikeway_right_2_type',
        'bikeway_right_2_surface',
        'bikeway_right_2_quality',
        'bikeway_right_2_permitted',
        'bikeway_right_2_width',
        'bikeway_right_2_incline',
        'bikeway_right_1_feature_ids',
        'bikeway_right_1_feature_types',
        'public_data_id_bikeway_right_1_features',
        'bikeway_right_1_feature_geometry',
        'bikeway_right_1_feature_geometry_projected',
        'bikeway_right_2_feature_types',
        'public_data_id_bikeway_right_2_features',
        'bikeway_right_2_feature_geometry',
        'bikeway_right_2_feature_geometry_projected',
    ])
    
    # Geometry columns (WKB bytes)
    columns.extend([
        'street_geometry',
        'start_node_geometry',
        'end_node_geometry',
        'sidewalk_left_geometry',
        'sidewalk_right_geometry',
        'curb_return_geometry',
        'bikeway_left_1_geometry',
        'bikeway_left_2_geometry',
        'bikeway_right_1_geometry',
        'bikeway_right_2_geometry',
    ])
    
    # Initialize DataFrame with all columns
    result = pd.DataFrame(index=network.index)
    
    for col in columns:
        if col in network.columns:
            result[col] = network[col]
        else:
            result[col] = None
    
    # Map OSM columns to schema columns
    if 'osmid' in network.columns:
        result['street_osmid'] = network['osmid']
    
    # Compute lane_width for each edge
    if 'lane_width' not in network.columns:
        result['lane_width'] = network.apply(
            lambda row: resolve_lane_width(row, normalize_tag(row.get('highway', 'residential')) or 'residential'),
            axis=1
        )
    
    # Apply default maxspeed from config
    if 'maxspeed' in result.columns:
        # Fill missing maxspeed with default from global config
        result['maxspeed'] = result['maxspeed'].fillna(global_config.default_max_speed)
    
    # Ensure public_data_id columns exist
    for col in columns:
        if col.startswith('public_data_id_') and col not in result.columns:
            result[col] = None
    
    # Determine source CRS from the network GeoDataFrame for reprojection to 4326.
    # All working geometries are in UTM; output parquet must be EPSG:4326 per spec.
    source_crs = network.crs if hasattr(network, 'crs') and network.crs is not None else None
    need_reproject = source_crs is not None and str(source_crs) != 'EPSG:4326'
    if need_reproject:
        print(f"  Will reproject geometries from {source_crs} to EPSG:4326 for export...")
    
    def _reproject_and_wkb(geom):
        """Reproject a single geometry to 4326 and convert to WKB."""
        if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
            return None
        if need_reproject:
            reprojected = gpd.GeoSeries([geom], crs=source_crs).to_crs('EPSG:4326').iloc[0]
            return geom_to_wkb(reprojected)
        return geom_to_wkb(geom)
    
    # Populate node geometries from nodes dataframe if provided, or extract from edges
    if nodes is not None:
        # Use provided nodes dataframe
        node_geom_map = dict(zip(nodes['osmid'], nodes['geometry']))
    else:
        # Extract nodes from edges
        print("  Extracting nodes from edges for geometry population...")
        nodes_extracted = extract_nodes_from_edges(network)
        node_geom_map = dict(zip(nodes_extracted['osmid'], nodes_extracted['geometry']))
    
    # Populate start_node_geometry (reproject to 4326 for export)
    if 'start_node_osmid' in result.columns:
        result['start_node_geometry'] = result['start_node_osmid'].apply(
            lambda node_id: _reproject_and_wkb(node_geom_map.get(node_id)) if pd.notna(node_id) and node_id in node_geom_map else None
        )
    
    # Populate end_node_geometry (reproject to 4326 for export)
    if 'end_node_osmid' in result.columns:
        result['end_node_geometry'] = result['end_node_osmid'].apply(
            lambda node_id: _reproject_and_wkb(node_geom_map.get(node_id)) if pd.notna(node_id) and node_id in node_geom_map else None
        )
    
    # Convert geometries to WKB
    # All working geometries are in UTM.  Reproject to EPSG:4326 before WKB
    # serialization so the output parquet is in the spec-required CRS.
    geom_columns = [
        'street_geometry',
        'sidewalk_left_geometry',
        'sidewalk_right_geometry',
        'curb_return_geometry',
        'bikeway_left_1_geometry',
        'bikeway_left_2_geometry',
        'bikeway_right_1_geometry',
        'bikeway_right_2_geometry',
    ]
    
    print(f"  Converting geometries to WKB (EPSG:4326)...")
    for col in geom_columns:
        if col == 'street_geometry':
            source_col = 'geometry'
        else:
            source_col = col
        
        if source_col in network.columns:
            # Check if column has any non-null values
            non_null_count = network[source_col].notna().sum()
            print(f"    {col}: {non_null_count}/{len(network)} non-null geometries")
            result[col] = network[source_col].apply(_reproject_and_wkb)
        else:
            print(f"    {col}: Column not found in network")
    
    # Convert curb ramp geometries to WKB (with reprojection)
    for side in ['left', 'right']:
        for slot in ['start', 'end']:
            for pos in [1, 2, 3]:
                col = f'sidewalk_{side}_curbramp_{slot}_{pos}_geometry'
                if col in result.columns:
                    result[col] = result[col].apply(_reproject_and_wkb)
    
    # Convert crosswalk geometries to WKB (with reprojection)
    for slot in ['start', 'end']:
        col = f'crosswalk_{slot}_geometry'
        if col in result.columns:
            result[col] = result[col].apply(_reproject_and_wkb)
    
    print(f"  Assembled schema with {len(result.columns)} columns")
    return result


def export_network_parquet(
    network: pd.DataFrame,
    city_name: str,
    output_path: str
) -> None:
    """Export network to parquet file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Define list-type columns for PyArrow schema
    LIST_TYPE_COLUMNS = [
        'street_feature_types',
        'public_data_id_street_feature',
        'sidewalk_left_feature_ids',
        'sidewalk_left_feature_types',
        'public_data_id_sidewalk_left_feature',
        'sidewalk_right_feature_ids',
        'sidewalk_right_feature_types',
        'public_data_id_sidewalk_right_feature',
        'bikeway_left_1_feature_ids',
        'bikeway_left_1_feature_types',
        'public_data_id_bikeway_left_1_features',
        'bikeway_left_2_feature_types',
        'public_data_id_bikeway_left_2_features',
        'bikeway_right_1_feature_ids',
        'bikeway_right_1_feature_types',
        'public_data_id_bikeway_right_1_features',
        'bikeway_right_2_feature_types',
        'public_data_id_bikeway_right_2_features',
        'block_ids',
        'block_sides',
        'crosswalk_start_block_ids',
        'crosswalk_end_block_ids',
        'public_data_id_start_end_nodes',
        'crosswalk_start_signals',
        'crosswalk_end_signals',
    ]
    
    # Define geometry columns that may contain empty geometries
    GEOMETRY_COLUMNS = [
        'street_feature_geometry',
        'street_feature_geometry_projected',
        'sidewalk_left_feature_geometry',
        'sidewalk_left_feature_geometry_projected',
        'sidewalk_right_feature_geometry',
        'sidewalk_right_feature_geometry_projected',
        'bikeway_left_1_feature_geometry',
        'bikeway_left_1_feature_geometry_projected',
        'bikeway_left_2_feature_geometry',
        'bikeway_left_2_feature_geometry_projected',
        'bikeway_right_1_feature_geometry',
        'bikeway_right_1_feature_geometry_projected',
        'bikeway_right_2_feature_geometry',
        'bikeway_right_2_feature_geometry_projected',
    ]
    
    # Convert list columns to proper format
    for col in LIST_TYPE_COLUMNS:
        if col in network.columns:
            network[col] = network[col].apply(
                lambda x: list(x) if isinstance(x, (list, tuple)) else None
            )
    
    # Handle empty geometries - convert to None for PyArrow compatibility
    for col in GEOMETRY_COLUMNS:
        if col in network.columns:
            network[col] = network[col].apply(
                lambda x: None if (x is None or (hasattr(x, 'is_empty') and x.is_empty)) else x
            )
    
    # Ensure string columns are properly typed (prevent PyArrow auto-conversion)
    STRING_COLUMNS = ['maxspeed', 'name', 'highway', 'surface', 'oneway']
    for col in STRING_COLUMNS:
        if col in network.columns:
            network[col] = network[col].astype(str).replace('nan', None)
    
    # Handle inf and NaN values in numeric columns
    for col in network.columns:
        if network[col].dtype in [np.float64, np.float32, np.float16]:
            network[col] = network[col].replace([np.inf, -np.inf], None)
            # Also handle NaN values
            network[col] = network[col].where(pd.notna(network[col]), None)
        # Handle object dtype columns that might have mixed types
        elif network[col].dtype == 'object':
            # Skip if it's a known list or geometry column
            if col not in LIST_TYPE_COLUMNS and col not in GEOMETRY_COLUMNS and col not in STRING_COLUMNS:
                # Check if column contains bytes (WKB geometries are fine)
                sample = network[col].dropna().head(1)
                if len(sample) > 0 and not isinstance(sample.iloc[0], bytes):
                    # Convert to string for safety, handling None values
                    network[col] = network[col].apply(
                        lambda x: str(x) if x is not None and not isinstance(x, bytes) else x
                    )
    
    # Write to parquet
    table = pa.Table.from_pandas(network)
    pq.write_table(table, output_path)
    print(f"  Exported network: {output_path}")


def _write_network_checkpoint(
    edges: gpd.GeoDataFrame,
    global_config: 'GlobalConfig',
    city_name: str,
    output_path: str,
    label: str = '',
) -> None:
    """Assemble schema from current edges state and write to the network parquet."""
    if label:
        print(f"\n  Writing network parquet [{label}]...")
    df = assemble_network_schema(edges, global_config)
    export_network_parquet(df, city_name, output_path)


# ============================================================================
# SECTION 17: PIPELINE STAGE FUNCTIONS
# ============================================================================

def _display_gpu_status() -> None:
    """Print GPU acceleration status."""
    if GPU_AVAILABLE:
        try:
            gpu_info = get_gpu_info()
            if gpu_info['available'] and gpu_info['gpu_names']:
                print(f"[GPU] Acceleration: ENABLED ({gpu_info['gpu_names'][0]})")
                if gpu_info['total_memory_gb']:
                    print(f"      Memory: {gpu_info['total_memory_gb'][0]:.1f} GB")
            else:
                print("GPU Acceleration: DISABLED (using CPU)")
        except Exception as e:
            print("GPU Acceleration: DISABLED (using CPU)")
            print(f"   GPU detection error: {e}")
    else:
        print("GPU Acceleration: NOT AVAILABLE (using CPU)")
    print()


def _reset_pipeline_counters() -> None:
    """Reset all sequential ID counters for a fresh city run."""
    global sidewalk_counter, bikeway_counter, curbramp_counter, crosswalk_counter, split_node_counter, sidewalk_feature_counter
    sidewalk_counter = SequentialIDCounter()
    sidewalk_feature_counter = SequentialIDCounter()
    bikeway_counter = SequentialIDCounter()
    curbramp_counter = SequentialIDCounter()
    crosswalk_counter = SequentialIDCounter()
    split_node_counter = SequentialIDCounter()


def _stage_load_and_init(
    city_config: CityConfig,
    global_config: GlobalConfig,
    network_output: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Stage 1: Fetch OSM data, project to UTM for accurate metric calculations,
    and write initial network parquet.
    Returns (edges, crossings_cache) — both in UTM CRS.
    """
    print("\n[Stage 1/14] Fetching OSM network...")
    edges, crossings_cache = fetch_osm_network(city_config)

    # Project to UTM for all intermediate processing.  All distance, offset,
    # buffer, and bearing calculations require a metric CRS.  The final
    # conversion back to EPSG:4326 happens in assemble_network_schema before
    # WKB serialization and parquet export.
    utm_crs = edges.estimate_utm_crs()
    print(f"  Projecting working data to {utm_crs} for metric calculations...")
    edges = edges.to_crs(utm_crs)
    if not crossings_cache.empty:
        crossings_cache = crossings_cache.to_crs(utm_crs)

    _write_network_checkpoint(edges, global_config, city_config.name, network_output, label='OSM load')
    return edges, crossings_cache


def _stage_sanity_buffer(
    edges: gpd.GeoDataFrame,
    city_config: CityConfig,
    global_config: GlobalConfig,
) -> dict:
    """
    Stage 2: Compute sanity buffer and write sanity parquet.
    Returns sanity_buffer dict {osmid: max_offset_width}.
    """
    print("\n[Stage 2/14] Computing sanity buffer...")
    parcel_path = city_config.government_data_paths.parcels
    if parcel_path and os.path.exists(parcel_path):
        sanity_df = compute_sanity_buffer_from_parcels(edges, parcel_path)
    else:
        sanity_df = compute_sanity_buffer_from_highway(edges)

    sanity_output = os.path.join(
        global_config.output_dir,
        f"{city_config.name.replace(', ', '_').replace(' ', '_')}_sanity.parquet"
    )
    export_sanity_buffer(sanity_df, city_config.name, sanity_output)

    sanity_buffer = dict(zip(sanity_df['osmid'], sanity_df['max_offset_width']))
    del sanity_df
    gc.collect()
    return sanity_buffer


def _stage_government_data(
    edges: gpd.GeoDataFrame,
    city_config: CityConfig,
    global_config: GlobalConfig,
    network_output: str,
) -> Tuple[gpd.GeoDataFrame, Any]:
    """
    Stage 3: Merge government centerlines, sidewalks, bikeways, and intersection nodes.
    Writes a parquet checkpoint after all government layers are applied.
    Returns (edges, gov_intersection_nodes).
    """
    print("\n[Stage 3/14] Merging government data...")
    gov_centerlines = load_government_centerlines(city_config)
    if gov_centerlines is not None:
        edges = merge_government_centerlines(edges, gov_centerlines, city_config)
        del gov_centerlines
        gc.collect()

    gov_sidewalks = load_government_sidewalks(city_config)
    if gov_sidewalks is not None:
        edges = merge_government_sidewalks(edges, gov_sidewalks, city_config)
        del gov_sidewalks
        gc.collect()

    gov_bikelanes = load_government_bikelanes(city_config)
    if gov_bikelanes is not None:
        edges = merge_government_bikelanes(edges, gov_bikelanes, city_config)
        del gov_bikelanes
        gc.collect()

    gov_intersection_nodes = load_government_intersection_nodes(city_config)
    if gov_intersection_nodes is not None:
        edges = merge_government_intersection_nodes(edges, gov_intersection_nodes)
        del gov_intersection_nodes
        gc.collect()

    _write_network_checkpoint(edges, global_config, city_config.name, network_output, label='government data')
    # gov_intersection_nodes has been applied; return None so the caller
    # signature stays unchanged but no stale reference is kept.
    return edges, None


def _stage_tag_extraction(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Stages 4-6: Sort network and extract cycleway and sidewalk tags from OSM attributes.

    Per architecture document: Raw OSM left/right tags are extracted as-is.
    Bearing assignment and left/right normalization happens later in Stage 10
    during block detection via the shoelace formula.
    
    Returns edges.
    """
    print("\n[Stage 4/14] Sorting network...")
    edges = edges.sort_values(['start_node_osmid', 'end_node_osmid']).reset_index(drop=True)

    print("\n[Stage 5/14] Extracting cycleway tags...")
    edges = extract_cycleway_tags(edges, bikeway_counter)

    print("\n[Stage 6/14] Extracting sidewalk tags...")
    edges = extract_sidewalk_tags(edges, sidewalk_counter)
    return edges


def _stage_deflection_split(
    edges: gpd.GeoDataFrame,
    global_config: GlobalConfig,
    city_config: CityConfig,
    network_output: str,
) -> gpd.GeoDataFrame:
    """
    Stage 7: Split edges at high vertex deflection angles.
    Rewrites the parquet since row count may change after splits.
    Returns edges.
    """
    print("\n[Stage 7/14] Detecting vertex deflections...")
    edges = detect_and_split_deflections(edges, split_node_counter)
    _write_network_checkpoint(edges, global_config, city_config.name, network_output, label='deflection split')
    return edges


def _stage_offset_and_features(
    edges: gpd.GeoDataFrame,
    sanity_buffer: dict,
    city_config: CityConfig,
    global_config: GlobalConfig,
) -> gpd.GeoDataFrame:
    """
    Stage 8: Generate offset geometries for bikeways and sidewalks, then
    populate street, sidewalk, and bikeway features from government data.
    Returns edges.
    """
    print("\n[Stage 8/14] Generating offset geometries...")
    edges = enrich_bikeway_geometries(edges, sanity_buffer)
    edges = enrich_sidewalk_geometries(edges, sanity_buffer)

    street_features = load_street_features(city_config)
    if street_features is not None:
        edges = populate_street_features(edges, street_features, city_config)
        del street_features
        gc.collect()
    else:
        edges['street_feature_types'] = [[] for _ in range(len(edges))]
        edges['public_data_id_street_feature'] = [[] for _ in range(len(edges))]
        edges['street_feature_geometry'] = [MultiPoint([]) for _ in range(len(edges))]
        edges['street_feature_geometry_projected'] = [MultiPoint([]) for _ in range(len(edges))]

    sidewalk_features = load_sidewalk_features(city_config)
    if sidewalk_features is not None:
        edges = populate_sidewalk_features(edges, sidewalk_features, city_config, sidewalk_feature_counter)
        del sidewalk_features
        gc.collect()

    bikeway_features = load_bikeway_features(city_config)
    if bikeway_features is not None:
        edges = populate_bikeway_features(edges, bikeway_features, city_config)
        del bikeway_features
        gc.collect()
    return edges


def _stage_bearing_and_blocks(
    edges: gpd.GeoDataFrame,
    gov_intersection_nodes: Any,
) -> gpd.GeoDataFrame:
    """
    Stages 9-10: Detect block structure via shoelace formula, assign bearings
    relative to block orientation, and reassign sidewalk/bikelane data to 
    correct left/right slots.
    
    Per architecture document Step 3: "For each detected block, use the shoelace 
    formula to assign bearings to each segment that makes up the block relative 
    to 0degrees north. Block ids should be assigned to left or right slots of 
    the street segments that make them up based on this shoelacing. Sidewalk 
    and Bikelane data should be re-assigned to the opposite left/right slot 
    based on this new normalized bearing."
    
    Returns edges.
    """
    print("\n[Stage 9/14] Detecting blocks via shoelace formula...")
    block_map = detect_blocks(edges)
    
    print("\n[Stage 10/14] Assigning bearings and normalizing left/right facility data...")
    edge_block_membership, edge_block_side_membership = assign_block_sides_shoelace(block_map, edges)
    edges = apply_block_ids_to_network(edges, edge_block_membership, edge_block_side_membership)
    
    # Compute bearings from geometry after block assignment
    edges['bearing'] = compute_bearings_vectorized(edges['geometry'])
    edges['normalized_bearing'] = edges['bearing'] % 180
    
    # Reassign sidewalk and bikelane data based on normalized bearing
    edges = reassign_facility_data_by_bearing(edges)
    
    return edges


def _stage_crosswalks(
    edges: gpd.GeoDataFrame,
    crossings_cache: gpd.GeoDataFrame,
    city_config: CityConfig,
) -> gpd.GeoDataFrame:
    """
    Stage 11: Extract crosswalk tags from the cached OSM crossings and
    any government crosswalk data.
    Returns edges.
    """
    print("\n[Stage 11/14] Extracting crosswalk tags...")
    gov_crosswalks = load_government_crosswalks(city_config)
    edges = extract_crosswalk_tags(edges, crossings_cache, crosswalk_counter, gov_crosswalks)
    del crossings_cache
    if gov_crosswalks is not None:
        del gov_crosswalks
    gc.collect()
    return edges


def _stage_intersection_analysis(
    edges: gpd.GeoDataFrame,
    sanity_buffer: dict,
    global_config: GlobalConfig,
) -> gpd.GeoDataFrame:
    """
    Stage 12: Run intersection analysis for bikeways and sidewalks (Steps 8-14).
    Returns edges.
    """
    print("\n[Stage 12/14] Running intersection analysis...")
    edges = run_intersection_analysis(edges, None, sanity_buffer, curbramp_counter, n_jobs=None)
    return edges


def _stage_curb_ramps(
    edges: gpd.GeoDataFrame,
    city_config: CityConfig,
    global_config: GlobalConfig,
    network_output: str,
) -> gpd.GeoDataFrame:
    """
    Stage 13: Assign government curb ramps to sidewalk slots and write
    a parquet checkpoint.
    Returns edges.
    """
    print("\n[Stage 13/14] Processing government curb ramps...")
    curb_ramps = load_curb_ramps(city_config)
    print(f"  Curb ramp trustworthy setting: {city_config.curbramp_trustworthy}")

    if len(curb_ramps) > 0:
        print("  Assigning curb ramps to network...")
        edges = assign_curb_ramps(
            edges,
            curb_ramps,
            global_config.curb_ramp_trustworthiness_outer_buffer,
            global_config.curb_ramp_trustworthiness_inner_buffer,
            global_config,
            curbramp_counter,
            trustworthy=city_config.curbramp_trustworthy,
        )
        if city_config.curbramp_trustworthy:
            edges = extract_curb_return_geometries(edges)
    else:
        print("  No curb ramps loaded from file")

    del curb_ramps
    gc.collect()
    _write_network_checkpoint(edges, global_config, city_config.name, network_output, label='curb ramps')
    return edges


def _stage_finalize(
    edges: gpd.GeoDataFrame,
    city_config: CityConfig,
    global_config: GlobalConfig,
    network_output: str,
) -> None:
    """
    Stage 14 + Final: Generate implicit crosswalk geometries and write the
    completed network parquet.
    """
    print("\n[Stage 14/14] Generating implicit crosswalk geometries...")
    edges = generate_implicit_crosswalk_geometries(edges, crosswalk_counter)
    _write_network_checkpoint(edges, global_config, city_config.name, network_output, label='final')


def main():
    """Main entry point."""
    # Create output directory
    os.makedirs("Notebooks/Karna/Proximity Model/Output", exist_ok=True)

    # Define global configuration
    GLOBAL_CONFIG = GlobalConfig(
        output_dir="Notebooks/Karna/Proximity Model/Output",
        default_max_speed=25,
        curb_ramp_trustworthiness_outer_buffer=15.0,
        curb_ramp_trustworthiness_inner_buffer=5.0,
        use_gpu=True,
        python_env="C:/Users/karna/miniconda3/envs/ParkximityENV/python.exe",
        cities=[
            CityConfig(
                name="San Francisco County, CA",
                government_data_paths=GovernmentDataPaths(
                    parcels="C:/Users/karna/Documents/Berkeley/Year_2/Sem 1/Human Mobility and Network Science/Assignments/Final Project.Parkximity/Parkximity/Notebooks/Karna/Proximity Model/Data/sf_parcels_2.16.26.geojson",
                    curb_ramps="C:/Users/karna/Documents/Berkeley/Year_2/Sem 1/Human Mobility and Network Science/Assignments/Final Project.Parkximity/Parkximity/Notebooks/Karna/Proximity Model/Data/Curb_Ramps_20260217.csv",
                ),
                column_mappings=ColumnMappingConfig(
                    curbramp_id="CNN",
                    curbramp_return_loc="curbReturnLoc",
                    curbramp_position="positionOnReturn",
                    curbramp_condition="conditionScore",
                ),
                curbramp_trustworthy=False,
            ),
            CityConfig(
                name="Alameda County, CA",
                government_data_paths=GovernmentDataPaths(
                ),
                column_mappings=ColumnMappingConfig(
                ),
                curbramp_trustworthy=False,
            ),
        ]
    )

    # Run pipeline for each city
    for city in GLOBAL_CONFIG.cities:
        print(f"\n{'='*80}")
        print(f"Processing: {city.name}")
        print(f"{'='*80}\n")
        _display_gpu_status()
        _reset_pipeline_counters()
        network_output = os.path.join(
            GLOBAL_CONFIG.output_dir,
            f"{city.name.replace(', ', '_').replace(' ', '_')}_network.parquet"
        )
        try:
            edges, crossings_cache = _stage_load_and_init(city, GLOBAL_CONFIG, network_output)
            sanity_buffer          = _stage_sanity_buffer(edges, city, GLOBAL_CONFIG)
            edges, gov_nodes       = _stage_government_data(edges, city, GLOBAL_CONFIG, network_output)
            edges                  = _stage_tag_extraction(edges)
            edges                  = _stage_deflection_split(edges, GLOBAL_CONFIG, city, network_output)
            edges                  = _stage_offset_and_features(edges, sanity_buffer, city, GLOBAL_CONFIG)
            edges                  = _stage_bearing_and_blocks(edges, gov_nodes)
            edges                  = _stage_crosswalks(edges, crossings_cache, city)
            edges                  = _stage_intersection_analysis(edges, sanity_buffer, GLOBAL_CONFIG)
            edges                  = _stage_curb_ramps(edges, city, GLOBAL_CONFIG, network_output)
            _stage_finalize(edges, city, GLOBAL_CONFIG, network_output)
            print(f"\n{'='*80}")
            print(f"✓ Successfully processed: {city.name}")
            print(f"{'='*80}\n")
        except Exception:
            print(f"\n{'='*80}")
            print(f"✗ Error processing {city.name}:")
            print(f"{'='*80}")
            traceback.print_exc()
            print()



if __name__ == "__main__":
    main()
