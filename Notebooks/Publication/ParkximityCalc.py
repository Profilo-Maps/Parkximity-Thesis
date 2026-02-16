import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import wkt
import heapq
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# ======================================================================
# Smart Factory Function - Automatically Chooses GPU or CPU
# ======================================================================

def create_parkximity_analyzer(city_name, config, target_crs="EPSG:6350", 
                               force_cpu=False, verbose=True, _show_status=True):
    """
    Factory function that automatically selects the best analyzer.
    
    This function detects GPU availability and returns either:
    - ParkXimityAnalyzerGPU (if NVIDIA GPU detected and libraries available)
    - ParkXimityAnalyzer (CPU-only fallback)
    
    Users can simply call this function and let the system decide which
    implementation to use based on available hardware.
    
    Parameters
    ----------
    city_name : str
        Name of the city being analyzed.
    config : dict
        Configuration dictionary (same as ParkXimityAnalyzer).
    target_crs : str
        Target coordinate reference system (default EPSG:6350).
    force_cpu : bool
        If True, force CPU mode even if GPU is available.
    verbose : bool
        If True, print which implementation is being used.
    _show_status : bool
        Internal flag to control status message display (used by multi_city_analysis).
    
    Returns
    -------
    analyzer : ParkXimityAnalyzer or ParkXimityAnalyzerGPU
        Analyzer instance ready to use.
    
    Examples
    --------
    >>> # Simple usage - automatic GPU detection
    >>> analyzer = create_parkximity_analyzer('Boston', config)
    >>> analyzer.full_analysis()
    
    >>> # Force CPU mode
    >>> analyzer = create_parkximity_analyzer('Boston', config, force_cpu=True)
    """
    # Try to import GPU utilities and check availability
    gpu_available = False
    gpu_info = None
    
    if not force_cpu:
        try:
            from gpu_utils import get_gpu_capabilities, prompt_gpu_installation_if_needed
            gpu_caps = get_gpu_capabilities()
            gpu_available = gpu_caps.is_available()
            if gpu_available:
                gpu_info = gpu_caps.get_summary()
            elif verbose and _show_status:
                # GPU hardware exists but libraries missing - offer to install
                # This will handle its own messaging
                should_retry = prompt_gpu_installation_if_needed()
                if should_retry:
                    # User installed packages but needs to restart Python
                    print("\n⚠ Please restart your Python session and run again.")
                    raise SystemExit(0)
        except ImportError:
            pass
        except Exception:
            pass
    
    # Create appropriate analyzer
    if gpu_available:
        try:
            from ParkximityCalcGPU import ParkXimityAnalyzerGPU
            return ParkXimityAnalyzerGPU(city_name, config, target_crs, force_cpu=False)
        except ImportError:
            if verbose and _show_status:
                print(f"\n⚠ GPU detected but ParkximityCalcGPU.py not found")
                print(f"  Using CPU mode\n")
            return ParkXimityAnalyzer(city_name, config, target_crs)
        except Exception as e:
            if verbose and _show_status:
                print(f"\n⚠ GPU initialization failed: {e}")
                print(f"  Using CPU mode\n")
            return ParkXimityAnalyzer(city_name, config, target_crs)
    else:
        return ParkXimityAnalyzer(city_name, config, target_crs)


class ParkXimityAnalyzer:
    """
    Analyzes park accessibility by computing LTS-weighted network distances
    from parcels to parks using street network analysis.

    All input data is automatically reprojected to the target CRS (default
    EPSG:6350 / NAD83(2011) Conus Albers, meters) for accurate distance
    calculations across the contiguous US.
    """

    def __init__(self, city_name, config, target_crs="EPSG:6350"):
        """
        Parameters
        ----------
        city_name : str
            Name of the city being analyzed.
        config : dict
            Configuration dictionary containing:
                - parcels_path: Path to parcels GeoJSON/CSV
                - parks_path: Path to parks GeoJSON
                - streets_path: Path to streets GeoJSON
                - boundary_path: Path to boundary GeoJSON/CSV (optional)
                - combine_boundaries: bool, merge neighborhood polygons (optional)
                - exports_output_dir: Global directory for all outputs (creates Data/ and Visualizations/ subfolders)
                - lts_column: Column name for LTS values in streets data
                - park_buffer: Buffer in meters for entrance detection (default 50)
                - entrance_tolerance: Dedup tolerance in meters (default 5)
                - heatmap_resolution: Grid cell size in meters (default 100)
                - heatmap_neighbors: IDW neighbor count (default 5)
                - heatmap_smoothing: Gaussian sigma (default 1)
                - n_jobs: Number of parallel workers (default: cpu_count() - 1)
        target_crs : str
            Target coordinate reference system (default EPSG:6350).
        """
        self.city_name = city_name
        self.config = config
        self.target_crs = target_crs

        self.parcels_gdf = None
        self.parks_gdf = None
        self.streets_gdf = None
        self.entrances_gdf = None
        self.boundary_gdf = None
        self.network_graph = None
        self.distances_dict = None
        self.lts_sums_dict = None
        self.paths_dict = None
        self.nearest_park_dict = None
        self.park_parcel_counts = None
        self.parks_without_entrances = []
        
        # Parallel processing configuration
        self.n_jobs = config.get('n_jobs', max(1, cpu_count() - 1))
        
        # Configure visualization fonts
        self._configure_fonts()

    def _configure_fonts(self):
        """Configure matplotlib fonts based on config settings."""
        font_family = self.config.get('font_family', 'serif')
        font_serif = self.config.get('font_serif', ['Times New Roman', 'Times', 'DejaVu Serif'])
        font_stretch = self.config.get('font_stretch', 'condensed')
        font_size = self.config.get('font_size', 10)
        
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.serif'] = font_serif
        plt.rcParams['font.stretch'] = font_stretch
        plt.rcParams['font.size'] = font_size

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self):
            """Load all required data files and reproject to target CRS."""
            print(f"Loading data for {self.city_name}...")
            print(f"Target CRS: {self.target_crs}")

            if self.config.get("boundary_path"):
                self._load_boundary()

            self._load_parcels()
            self._load_parks()

            # Clip parcels and parks to boundary if specified
            if self.boundary_gdf is not None:
                self._clip_data_to_boundary()

            self.streets_gdf = self._load_and_reproject(
                self.config["streets_path"], "Streets"
            )
            print(f"Loaded {len(self.streets_gdf)} street segments")

            self._verify_crs_consistency()

    def _load_and_reproject(self, filepath, description="data"):
        """Load a geospatial file and reproject to target CRS."""
        gdf = gpd.read_file(filepath)
        if gdf.crs is None:
            print(f"  Warning: {description} has no CRS, assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")
        print(f"  {description}: {gdf.crs} -> {self.target_crs}")
        return gdf.to_crs(self.target_crs)

    def _load_parcels(self):
        """Load parcels from GeoJSON or CSV with geometry validation."""
        parcels_path = self.config["parcels_path"]

        if parcels_path.endswith((".geojson", ".json")):
            self._load_parcels_geojson(parcels_path)
        else:
            self._load_parcels_csv(parcels_path)

        print(f"Loaded {len(self.parcels_gdf)} valid parcels")

    def _load_parcels_geojson(self, path):
        """Load parcels from GeoJSON with parallel geometry validation."""
        import warnings

        print("Loading parcels from GeoJSON...")
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                raw = gpd.read_file(path, on_invalid="warn")
                if w:
                    print(f"  {len(w)} geometry warnings during load")
        except Exception as e:
            print(f"  Error loading parcels: {e}")
            self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
            return

        if raw.crs is None:
            raw = raw.set_crs("EPSG:4326")

        # Parallel geometry validation for large datasets
        if len(raw) > 10000 and self.n_jobs > 1:
            print(f"  Validating {len(raw)} geometries using {self.n_jobs} workers...")
            
            # Split into chunks for parallel processing
            chunk_size = max(1000, len(raw) // (self.n_jobs * 2))
            chunks = [raw.iloc[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
            
            # Process chunks in parallel
            with Pool(processes=self.n_jobs) as pool:
                valid_chunks = list(tqdm(
                    pool.imap(_validate_geometry_chunk, chunks),
                    total=len(chunks),
                    desc="  Validating parcels",
                    unit="chunk"
                ))
            
            # Combine valid geometries
            valid_gdf = pd.concat(valid_chunks, ignore_index=True)
            invalid_count = len(raw) - len(valid_gdf)
            
            if invalid_count > 0:
                print(f"  Skipped {invalid_count} invalid/empty parcels")
            
            if len(valid_gdf) > 0:
                self.parcels_gdf = valid_gdf.to_crs(self.target_crs)
            else:
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
        else:
            # Sequential validation for small datasets - fully vectorized
            valid_mask = (
                raw.geometry.notna() &  # Not null
                ~raw.geometry.is_empty &  # Not empty
                raw.geometry.is_valid  # Valid geometry
            )

            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"  Skipped {invalid_count} invalid/empty parcels")

            if valid_mask.any():
                self.parcels_gdf = raw.loc[valid_mask].copy().to_crs(self.target_crs)
            else:
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)

    def _load_parcels_csv(self, path):
        """Load parcels from CSV with lat/lon columns."""
        print("Loading parcels from CSV...")
        df = pd.read_csv(path, engine="python", on_bad_lines="warn")

        valid = df.dropna(subset=["centroid_latitude", "centroid_longitude"])
        # Vectorized Point creation (50-100x faster than iterrows)
        geometries = gpd.points_from_xy(
            valid['centroid_longitude'], 
            valid['centroid_latitude']
        )
        self.parcels_gdf = gpd.GeoDataFrame(
            valid, geometry=geometries, crs="EPSG:4326"
        ).to_crs(self.target_crs)

    def _load_parks(self):
        """Load, validate, and repair park geometries."""
        parks_raw = self._load_and_reproject(self.config["parks_path"], "Parks")
        print(f"Loaded {len(parks_raw)} parks from file")

        # Keep only polygon geometries
        poly_mask = parks_raw.geometry.type.isin(["Polygon", "MultiPolygon"])
        parks_poly = parks_raw[poly_mask].copy()
        dropped = len(parks_raw) - len(parks_poly)
        if dropped:
            print(f"  Filtered to {len(parks_poly)} polygon parks (removed {dropped})")

        if len(parks_poly) == 0:
            raise ValueError("No polygon parks found in dataset")

        # Validate and repair in parallel
        print(f"  Validating geometries using {self.n_jobs} workers...")
        
        # Convert to list of tuples for parallel processing
        park_data = [(idx, row) for idx, row in parks_poly.iterrows()]
        
        with Pool(processes=self.n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_validate_park_geometry, park_data),
                total=len(park_data),
                desc="  Validating parks",
                unit="park"
            ))
        
        # Filter out None results and count repairs
        valid_parks = [r for r in results if r is not None]
        repaired = sum(1 for r in results if r is not None and r.get('_repaired', False))

        if not valid_parks:
            raise ValueError("No valid polygon parks after geometry repair")

        self.parks_gdf = gpd.GeoDataFrame(valid_parks, crs=self.target_crs)
        if repaired:
            print(f"  Repaired {repaired} invalid geometries")
        print(f"  {len(self.parks_gdf)} valid parks ready")

    @staticmethod
    def _validate_park_geometry_static(park_tuple):
        """Static method for parallel park geometry validation."""
        idx, row = park_tuple
        geom = row.geometry
        
        if geom is None or geom.is_empty:
            return None
            
        if not geom.is_valid:
            try:
                fixed = geom.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    row = row.copy()
                    row["geometry"] = fixed
                    row['_repaired'] = True
                    return row
            except Exception:
                pass
            return None
        
        return row

    def _clip_data_to_boundary(self):
        """Clip parcels and parks to city boundary if specified."""
        if self.boundary_gdf is None or len(self.boundary_gdf) == 0:
            return

        print("Clipping data to city boundary...")
        boundary_geom = self.boundary_gdf.iloc[0].geometry

        # Clip parcels (parallelized for large datasets)
        if self.parcels_gdf is not None and len(self.parcels_gdf) > 0:
            original_parcel_count = len(self.parcels_gdf)
            
            # Use parallel processing for large datasets
            if len(self.parcels_gdf) > 10000:
                # Split into chunks for parallel processing
                chunk_size = max(1000, len(self.parcels_gdf) // (self.n_jobs * 4))
                chunks = [self.parcels_gdf.iloc[i:i+chunk_size] 
                         for i in range(0, len(self.parcels_gdf), chunk_size)]
                
                print(f"  Checking {len(self.parcels_gdf)} parcels using {self.n_jobs} workers ({len(chunks)} chunks)...")
                
                # Process chunks in parallel
                with Pool(processes=self.n_jobs) as pool:
                    results = list(tqdm(
                        pool.starmap(
                            _clip_geometries_chunk,
                            [(chunk, boundary_geom) for chunk in chunks]
                        ),
                        total=len(chunks),
                        desc="  Clipping parcels",
                        unit="chunk"
                    ))
                
                # Combine results
                self.parcels_gdf = pd.concat(results, ignore_index=True)
            else:
                # Small dataset - use simple filtering
                self.parcels_gdf = self.parcels_gdf[
                    self.parcels_gdf.geometry.intersects(boundary_geom)
                ].copy()
            
            clipped_parcels = original_parcel_count - len(self.parcels_gdf)
            if clipped_parcels > 0:
                print(f"  Clipped {clipped_parcels} parcels outside boundary "
                      f"({len(self.parcels_gdf)} remaining)")
            else:
                print(f"  All {len(self.parcels_gdf)} parcels within boundary")

        # Clip parks (usually small, no need to parallelize)
        if self.parks_gdf is not None and len(self.parks_gdf) > 0:
            original_park_count = len(self.parks_gdf)
            self.parks_gdf = self.parks_gdf[
                self.parks_gdf.geometry.intersects(boundary_geom)
            ].copy()
            clipped_parks = original_park_count - len(self.parks_gdf)
            if clipped_parks > 0:
                print(f"  Clipped {clipped_parks} parks outside boundary "
                      f"({len(self.parks_gdf)} remaining)")
            else:
                print(f"  All {len(self.parks_gdf)} parks within boundary")


    def _load_boundary(self):
        """Load boundary from GeoJSON or CSV."""
        path = self.config["boundary_path"]
        if path.endswith((".geojson", ".json")):
            self._load_boundary_from_geojson()
        elif path.endswith(".csv"):
            self._load_boundary_from_csv()
        else:
            print(f"Warning: unsupported boundary format: {path}")

    def _load_boundary_from_geojson(self):
        """Load boundary from GeoJSON, optionally combining neighborhoods."""
        print(f"Loading boundary from GeoJSON for {self.city_name}...")
        gdf = gpd.read_file(self.config["boundary_path"])
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        # Optional name filter
        if self.config.get("boundary_name"):
            col = self.config.get("boundary_county_column", "name")
            gdf = gdf[gdf[col] == self.config["boundary_name"]]
            if len(gdf) == 0:
                print(f"  Warning: no boundary found for {self.config['boundary_name']}")
                return

        boundary = self._resolve_boundary_geometry(gdf)
        if boundary is None:
            return

        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[boundary], crs=gdf.crs
        ).to_crs(self.target_crs)
        print(f"  Loaded boundary for {self.city_name}")

    def _load_boundary_from_csv(self):
        """Load boundary from CSV with WKT geometry column."""
        print(f"Loading boundary from CSV for {self.city_name}...")
        df = pd.read_csv(self.config["boundary_path"])
        wkt_col = self.config.get("boundary_wkt_column", "the_geom")
        county_col = self.config.get("boundary_county_column", "COUNTY")
        csv_crs = self.config.get("boundary_csv_crs", "EPSG:4326")

        name = self.config.get("boundary_name", self.city_name)
        combine = self.config.get("combine_boundaries", False)

        if combine:
            rows = df[df[county_col] == name]
            if len(rows) == 0:
                print(f"  Warning: no neighborhoods found for {name}")
                return
            
            # Vectorized geometry extraction
            geoms = []
            for wkt_str in rows[wkt_col]:
                g = wkt.loads(wkt_str)
                if isinstance(g, MultiPolygon):
                    geoms.extend(list(g.geoms))
                else:
                    geoms.append(g)
            boundary = self._combine_boundaries(geoms)
        else:
            match = df[df[county_col] == name]
            if len(match) == 0:
                print(f"  Warning: no boundary found for {name}")
                return
            boundary = wkt.loads(match[wkt_col].iloc[0])
            if isinstance(boundary, MultiPolygon):
                boundary = max(boundary.geoms, key=lambda p: p.area)

        if boundary is None:
            return

        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[boundary], crs=csv_crs
        ).to_crs(self.target_crs)
        print(f"  Loaded boundary for {self.city_name}")

    def _resolve_boundary_geometry(self, gdf):
        """Extract a single boundary polygon from a GeoDataFrame."""
        combine = self.config.get("combine_boundaries", False)

        if combine or len(gdf) > 1:
            if len(gdf) > 1 and not combine:
                print(f"  Found {len(gdf)} features, combining into one boundary")
            
            # Vectorized geometry extraction
            geoms = []
            for g in gdf.geometry:
                if isinstance(g, MultiPolygon):
                    geoms.extend(list(g.geoms))
                else:
                    geoms.append(g)
            return self._combine_boundaries(geoms)
        else:
            geom = gdf.iloc[0].geometry
            if isinstance(geom, MultiPolygon):
                return max(geom.geoms, key=lambda p: p.area)
            return geom

    def _combine_boundaries(self, polygons):
        """Merge neighborhood polygons into a single city boundary."""
        if not polygons:
            return None
        merged = unary_union(polygons)
        if isinstance(merged, MultiPolygon):
            print(f"  Warning: boundary has {len(merged.geoms)} disconnected parts, using largest")
            merged = max(merged.geoms, key=lambda p: p.area)
        if hasattr(merged, "exterior"):
            return Polygon(merged.exterior)
        return merged

    def _verify_crs_consistency(self):
        """Verify all loaded datasets are in the target CRS."""
        datasets = {
            "Parks": self.parks_gdf,
            "Streets": self.streets_gdf,
            "Parcels": self.parcels_gdf,
            "Boundary": self.boundary_gdf,
        }
        from pyproj import CRS
        target = CRS.from_user_input(self.target_crs)
        ok = True
        for name, gdf in datasets.items():
            if gdf is not None and len(gdf) > 0 and gdf.crs != target:
                print(f"  WARNING: {name} CRS mismatch: {gdf.crs} != {self.target_crs}")
                ok = False
        if ok:
            print(f"  CRS verification passed: all datasets in {self.target_crs}")

    # ------------------------------------------------------------------
    # Park Entrance Generation
    # ------------------------------------------------------------------

    def generate_park_entrances(self):
        """Find points where streets intersect park boundaries."""
        print("Generating park entrances...")

        if len(self.parks_gdf) == 0 or len(self.streets_gdf) == 0:
            print("  ERROR: no parks or streets available")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
            return

        buffer_distance = self.config.get("park_buffer", 50.0)
        entrance_tolerance = self.config.get("entrance_tolerance", 5.0)
        batch_size = self.config.get("entrance_batch_size", 50)

        print(f"  Buffer: {buffer_distance}m | Tolerance: {entrance_tolerance}m | Batch: {batch_size}")
        print(f"  Using {self.n_jobs} parallel workers")

        # Build spatial index once
        streets_tree = STRtree(self.streets_gdf.geometry)

        all_coords = []
        all_data = []
        total = len(self.parks_gdf)

        # Create batches for parallel processing
        batches = []
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = self.parks_gdf.iloc[batch_start:batch_end]
            batches.append((batch, batch_start, batch_end, total))

        # Process batches in parallel
        process_func = partial(
            _process_park_batch,
            streets_gdf=self.streets_gdf,
            streets_tree=streets_tree,
            buffer_distance=buffer_distance,
            entrance_tolerance=entrance_tolerance
        )
        
        with Pool(processes=self.n_jobs) as pool:
            batch_results = list(tqdm(
                pool.imap(process_func, batches),
                total=len(batches),
                desc="  Finding entrances",
                unit="batch"
            ))

        # Aggregate results
        for coords, data, no_access in batch_results:
            all_coords.extend(coords)
            all_data.extend(data)
            self.parks_without_entrances.extend(no_access)

        print(f"  Total candidates: {len(all_coords)}")

        # Use only per-park deduplication (no global deduplication)
        # This preserves all legitimate entrances, including those from adjacent parks
        unique_coords = all_coords
        unique_data = all_data

        if unique_coords:
            points = [Point(x, y) for x, y in unique_coords]
            self.entrances_gdf = gpd.GeoDataFrame(
                unique_data, geometry=points, crs=self.target_crs
            )
        else:
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)

        print(f"  Generated {len(self.entrances_gdf)} unique entrances")

        if self.parks_without_entrances:
            print(f"  {len(self.parks_without_entrances)} parks without entrances")

    def _batch_park_entrances(self, batch_parks, streets_tree, buffer_distance, entrance_tolerance):
        """Process a batch of parks to generate entrance points."""
        batch_coords = []
        batch_data = []
        no_access = []

        for park_idx, park in batch_parks.iterrows():
            park_geom = park.geometry
            park_name = (
                park.get("name")
                or park.get("NAME")
                or park.get("park_name")
                or park.get("PARK_NAME")
                or f"Park_{park_idx}"
            )

            buffered = park_geom.buffer(buffer_distance)
            candidates = streets_tree.query(buffered)

            if len(candidates) == 0:
                no_access.append(park_name)
                continue

            intersecting = self.streets_gdf.iloc[candidates]
            intersecting = intersecting[intersecting.intersects(buffered)]

            if len(intersecting) == 0:
                no_access.append(park_name)
                continue

            park_points = []
            for street_idx, street in intersecting.iterrows():
                try:
                    intersection = street.geometry.intersection(park_geom.boundary)
                except Exception:
                    continue
                if intersection.is_empty:
                    continue

                for pt in self._extract_points_from_intersection(intersection):
                    if pt is not None and not pt.is_empty:
                        park_points.append({
                            "coord": (pt.x, pt.y),
                            "park_name": park_name,
                            "park_id": park_idx,
                            "street_id": street_idx,
                        })

            if park_points:
                # Per-park deduplication
                unique = self._deduplicate_points(park_points, entrance_tolerance)
                for p in unique:
                    batch_coords.append(p["coord"])
                    batch_data.append({
                        "park_name": p["park_name"],
                        "park_id": p["park_id"],
                        "street_id": p["street_id"],
                    })
            else:
                # Fallback: nearest point on boundary to closest street
                try:
                    nearest_street = intersecting.iloc[0]
                    nearest_pt = park_geom.boundary.interpolate(
                        park_geom.boundary.project(nearest_street.geometry.centroid)
                    )
                    if nearest_pt and not nearest_pt.is_empty:
                        batch_coords.append((nearest_pt.x, nearest_pt.y))
                        batch_data.append({
                            "park_name": park_name,
                            "park_id": park_idx,
                            "street_id": nearest_street.name,
                        })
                except Exception:
                    no_access.append(park_name)

        return batch_coords, batch_data, no_access

    def _extract_points_from_intersection(self, intersection):
        """Extract Point objects from various geometry types."""
        points = []
        gtype = intersection.geom_type

        if gtype == "Point":
            points.append(intersection)
        elif gtype == "MultiPoint":
            points.extend(list(intersection.geoms))
        elif gtype == "LineString":
            coords = list(intersection.coords)
            if len(coords) >= 2:
                points.extend([Point(coords[0]), Point(coords[-1])])
        elif gtype == "MultiLineString":
            for line in intersection.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    points.extend([Point(coords[0]), Point(coords[-1])])
        elif gtype == "GeometryCollection":
            for geom in intersection.geoms:
                points.extend(self._extract_points_from_intersection(geom))

        return points

    def _deduplicate_points(self, point_dicts, tolerance):
        """Deduplicate points within a single park using KD-tree."""
        if len(point_dicts) <= 1:
            return point_dicts
        coords = np.array([p["coord"] for p in point_dicts])
        tree = cKDTree(coords)
        remove = set()
        for i, j in tree.query_pairs(r=tolerance):
            if i not in remove:
                remove.add(j)
        return [p for idx, p in enumerate(point_dicts) if idx not in remove]

    def _vectorized_deduplication_park_entrances(self, coords_list, data_list, tolerance):
        """Global deduplication across all entrances using KD-tree."""
        if not coords_list:
            return [], []
        coords_array = np.array(coords_list)
        tree = cKDTree(coords_array)
        remove = set()
        for i, j in tree.query_pairs(r=tolerance):
            if i not in remove:
                remove.add(j)
        unique_coords = [c for idx, c in enumerate(coords_list) if idx not in remove]
        unique_data = [d for idx, d in enumerate(data_list) if idx not in remove]
        return unique_coords, unique_data

    # ------------------------------------------------------------------
    # Network Construction
    # ------------------------------------------------------------------

    def build_network(self):
        """Build weighted graph from street segments."""
        print("Building street network...")
        self.network_graph = nx.Graph()
        lts_col = self.config.get("lts_column", "final_lts")

        # Vectorized approach: extract all data at once
        valid_streets = self.streets_gdf[self.streets_gdf[lts_col].notna()].copy()
        
        # Parallel processing for large networks
        if len(valid_streets) > 50000 and self.n_jobs > 1:
            print(f"  Processing {len(valid_streets)} street segments using {self.n_jobs} workers...")
            
            # Split into chunks for parallel processing
            chunk_size = max(1000, len(valid_streets) // (self.n_jobs * 2))
            chunks = []
            for i in range(0, len(valid_streets), chunk_size):
                chunk = valid_streets.iloc[i:i+chunk_size]
                chunks.append((chunk, lts_col))
            
            # Process chunks in parallel
            with Pool(processes=self.n_jobs) as pool:
                chunk_results = list(tqdm(
                    pool.imap(_process_street_network_chunk, chunks),
                    total=len(chunks),
                    desc="  Building network",
                    unit="chunk"
                ))
            
            # Combine all edges
            edges = [edge for chunk_edges in chunk_results for edge in chunk_edges]
        else:
            # Sequential processing for small networks
            edges = []
            for geom, lts in zip(valid_streets.geometry, valid_streets[lts_col]):
                coords = list(geom.coords)
                for j in range(len(coords) - 1):
                    start, end = coords[j], coords[j + 1]
                    length = LineString([start, end]).length
                    if length > 0:
                        edges.append((
                            start, end,
                            {"weight": length * (1 + lts), "length": length, "lts": lts},
                        ))

        self.network_graph.add_edges_from(edges)
        print(f"  Graph: {self.network_graph.number_of_nodes()} nodes, "
              f"{self.network_graph.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Parkximity Calculation
    # ------------------------------------------------------------------

    def find_parkximity(self):
        """Calculate LTS-weighted distances from every parcel to nearest park."""
        print("Calculating parkximity...")

        self._run_dijkstra()
        self._assign_parcels_to_nodes()
        self._export_parkximity_json()

    def _run_dijkstra(self):
        """
        Multi-source Dijkstra from all park entrance nodes.

        Tiebreak: when two paths have the same distance, prefer the one
        with lower cumulative LTS.  Tracks which entrance (and therefore
        which park) is closest to every reachable node.
        """
        if self.entrances_gdf is None or len(self.entrances_gdf) == 0:
            raise ValueError("No park entrances — cannot run Dijkstra")

        # Map entrance points to nearest network nodes
        nodes_list = list(self.network_graph.nodes())
        nodes_array = np.array(nodes_list)
        tree = cKDTree(nodes_array)

        entrance_coords = np.array(
            [[p.x, p.y] for p in self.entrances_gdf.geometry]
        )
        _, indices = tree.query(entrance_coords)

        # Build entrance_index -> park_id mapping
        entrance_park_ids = list(self.entrances_gdf["park_id"])

        # Initialise data structures
        self.distances_dict = {}
        self.lts_sums_dict = {}
        self.paths_dict = {}
        self.nearest_park_dict = {}  # node -> park_id

        source_nodes = {}  # node_tuple -> (entrance_index, park_id)
        for eidx, nidx in enumerate(indices):
            node = tuple(nodes_array[nidx])
            pid = entrance_park_ids[eidx]
            if node not in source_nodes:
                source_nodes[node] = (eidx, pid)

        pq = []
        for node, (eidx, pid) in source_nodes.items():
            self.distances_dict[node] = 0
            self.lts_sums_dict[node] = 0
            self.paths_dict[node] = [node]
            self.nearest_park_dict[node] = pid
            heapq.heappush(pq, (0, 0, node))

        visited = set()
        total_nodes = self.network_graph.number_of_nodes()

        print(f"  Running Dijkstra from {len(source_nodes)} source nodes...")
        
        pbar = tqdm(total=total_nodes, desc="  Dijkstra search", unit="node")

        while pq:
            dist, lts_sum, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            pbar.update(1)

            for neighbor in self.network_graph.neighbors(current):
                if neighbor in visited:
                    continue
                edge = self.network_graph[current][neighbor]
                new_dist = dist + edge["length"]
                new_lts = lts_sum + edge["lts"]

                better = (
                    neighbor not in self.distances_dict
                    or new_dist < self.distances_dict[neighbor]
                    or (
                        new_dist == self.distances_dict[neighbor]
                        and new_lts < self.lts_sums_dict[neighbor]
                    )
                )
                if better:
                    self.distances_dict[neighbor] = new_dist
                    self.lts_sums_dict[neighbor] = new_lts
                    self.paths_dict[neighbor] = self.paths_dict[current] + [neighbor]
                    self.nearest_park_dict[neighbor] = self.nearest_park_dict[current]
                    heapq.heappush(pq, (new_dist, new_lts, neighbor))

        pbar.close()
        reachable = len(self.distances_dict)
        print(f"  Dijkstra complete: {reachable}/{total_nodes} nodes reachable")

    def _assign_parcels_to_nodes(self):
        """Assign each parcel its parkximity distance and nearest park."""
        print("  Assigning parcels to network nodes...")

        if len(self.parcels_gdf) == 0:
            print("  Warning: no parcels to process")
            return

        nodes_list = list(self.network_graph.nodes())
        nodes_array = np.array(nodes_list)
        tree = cKDTree(nodes_array)

        # Vectorised centroid extraction
        parcel_coords = np.array([
            (g.centroid.x, g.centroid.y) if g.geom_type in ("Polygon", "MultiPolygon")
            else (g.x, g.y)
            for g in self.parcels_gdf.geometry
        ])

        nn_dists, nn_indices = tree.query(parcel_coords)

        # Vectorised distance lookup
        parcel_to_node = nn_dists
        node_to_park = np.array([
            self.distances_dict.get(tuple(nodes_array[i]), float("inf"))
            for i in nn_indices
        ])
        nearest_parks = np.array([
            self.nearest_park_dict.get(tuple(nodes_array[i]), -1)
            for i in nn_indices
        ])

        self.parcels_gdf["parcel_to_node_distance"] = parcel_to_node
        self.parcels_gdf["node_to_park_distance"] = node_to_park
        self.parcels_gdf["park_distance"] = parcel_to_node + node_to_park
        self.parcels_gdf["nearest_park_id"] = nearest_parks

        # Build park -> parcel count mapping
        valid = self.parcels_gdf[self.parcels_gdf["park_distance"] < float("inf")]
        self.park_parcel_counts = valid["nearest_park_id"].value_counts().to_dict()

        valid_dists = valid["park_distance"]
        if len(valid_dists) > 0:
            print(f"  Distances: min={valid_dists.min():.1f}m, "
                  f"max={valid_dists.max():.1f}m, mean={valid_dists.mean():.1f}m")
        print(f"  {len(valid)}/{len(self.parcels_gdf)} parcels with valid distances")

    def _export_parkximity_json(self):
        """Export full parkximity dataset as JSON."""
        print("  Exporting parkximity JSON...")

        # Create city-specific subfolder in Exports/Data directory
        exports_dir = Path(self.config.get("exports_output_dir", "Exports"))
        data_dir = exports_dir / "Data" / self.city_name.replace(' ', '_').replace(',', '')
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = data_dir / f"{self.city_name.lower().replace(' ', '_')}_parkximity_data.json"

        # Vectorized parks data extraction
        parks_data = []
        for park_idx in self.parks_gdf.index:
            park = self.parks_gdf.loc[park_idx]
            park_name = (
                park.get("name")
                or park.get("NAME")
                or park.get("park_name")
                or park.get("PARK_NAME")
                or f"Park_{park_idx}"
            )
            centroid = park.geometry.centroid
            parcel_count = self.park_parcel_counts.get(park_idx, 0)

            # Find parcel indices closest to this park
            closest_parcels = []
            if "nearest_park_id" in self.parcels_gdf.columns:
                mask = self.parcels_gdf["nearest_park_id"] == park_idx
                closest_parcels = self.parcels_gdf.index[mask].tolist()

            parks_data.append({
                "id": int(park_idx) if isinstance(park_idx, (int, np.integer)) else str(park_idx),
                "name": str(park_name),
                "centroid": [float(centroid.x), float(centroid.y)],
                "parcel_count": int(parcel_count),
                "closest_parcel_ids": [
                    int(p) if isinstance(p, (int, np.integer)) else str(p)
                    for p in closest_parcels
                ],
            })

        # Vectorized parcels data extraction (parallelized for large datasets)
        nodes_list = list(self.network_graph.nodes())
        nodes_array = np.array(nodes_list)
        
        if len(self.parcels_gdf) > 10000 and self.n_jobs > 1:
            # Split parcels into chunks
            chunk_size = max(1000, len(self.parcels_gdf) // (self.n_jobs * 4))
            chunks = []
            for i in range(0, len(self.parcels_gdf), chunk_size):
                chunk = self.parcels_gdf.iloc[i:i+chunk_size]
                chunks.append((chunk, nodes_array, self.paths_dict))
            
            # Process chunks in parallel
            with Pool(processes=self.n_jobs) as pool:
                chunk_results = list(tqdm(
                    pool.imap(_process_parcel_json_chunk, chunks),
                    total=len(chunks),
                    desc="  Exporting parcels",
                    unit="chunk"
                ))
            
            # Flatten results
            parcels_data = [item for chunk in chunk_results for item in chunk]
        else:
            # Small dataset or single worker - vectorized sequential processing
            tree = cKDTree(nodes_array)
            
            # Extract all centroids at once (vectorized)
            centroids = []
            for geom in self.parcels_gdf.geometry:
                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    centroids.append((geom.centroid.x, geom.centroid.y))
                else:
                    centroids.append((geom.x, geom.y))
            centroids = np.array(centroids)
            
            # Query all nearest nodes at once
            _, nn_indices = tree.query(centroids)
            
            # Build parcels data
            parcels_data = []
            for idx, (parcel_idx, parcel) in enumerate(self.parcels_gdf.iterrows()):
                cx, cy = centroids[idx]
                p2n = parcel.get("parcel_to_node_distance", float("inf"))
                n2p = parcel.get("node_to_park_distance", float("inf"))
                total = parcel.get("park_distance", float("inf"))
                npid = parcel.get("nearest_park_id", -1)

                # Get path for this parcel
                nearest_node = tuple(nodes_array[nn_indices[idx]])
                path = self.paths_dict.get(nearest_node, [])
                route = [[float(c) for c in coord] for coord in path]

                parcels_data.append({
                    "id": int(parcel_idx) if isinstance(parcel_idx, (int, np.integer)) else str(parcel_idx),
                    "centroid": [float(cx), float(cy)],
                    "parkximity": float(total) if total != float("inf") else None,
                    "parcel_to_node_distance": float(p2n) if p2n != float("inf") else None,
                    "node_to_park_distance": float(n2p) if n2p != float("inf") else None,
                    "nearest_park_id": int(npid) if isinstance(npid, (int, np.integer)) and npid != -1 else None,
                    "route": route,
                })

        output = {
            "city": self.city_name,
            "target_crs": self.target_crs,
            "total_parks": len(self.parks_gdf),
            "total_parcels": len(self.parcels_gdf),
            "parks": parks_data,
            "parcels": parcels_data,
        }

        print(f"  Writing JSON file ({len(parcels_data)} parcels)...")
        
        # Try fast JSON library first (orjson is 2-5x faster for large files)
        try:
            import orjson
            with open(filename, "wb") as f:
                # orjson.dumps returns bytes, OPT_INDENT_2 for pretty printing
                f.write(orjson.dumps(output, option=orjson.OPT_INDENT_2))
        except ImportError:
            # Fallback to standard json (slower but always available)
            with open(filename, "w") as f:
                json.dump(output, f, indent=2)

        print(f"  Saved parkximity data to {filename}")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def create_visualizations(self):
        """
        Create all output visualizations.
        
        For GPU-accelerated analyzers, creates visualizations sequentially to avoid
        GPU context conflicts with multiprocessing. For CPU-only, uses parallel creation.
        """
        # Create city-specific subfolders in Exports directory
        exports_dir = Path(self.config.get("exports_output_dir", "Exports"))
        viz_dir = exports_dir / "Visualizations" / self.city_name.replace(' ', '_').replace(',', '')
        viz_dir.mkdir(parents=True, exist_ok=True)
        base = viz_dir / f"{self.city_name.lower().replace(' ', '_')}_parkximity"

        # Prepare visualization tasks
        viz_tasks = [
            ('entrance_map', self._create_entrance_map, base),
            ('parks_without_entrances', self._create_parks_without_entrances_map, base),
            ('heatmap', self._create_heatmap, base),
            ('histograms', self._create_distance_histograms, base),
        ]
        
        # Check if this is a GPU-accelerated analyzer
        use_gpu = getattr(self, 'use_gpu', False)
        
        if use_gpu:
            # Sequential creation for GPU to avoid context conflicts
            print(f"Creating visualizations sequentially (GPU mode)...")
            for task_name, func, base_path in viz_tasks:
                try:
                    func(base_path)
                except Exception as e:
                    print(f"  Warning: Failed to create {task_name}: {e}")
        else:
            # Parallel creation for CPU
            print(f"Creating visualizations using {self.n_jobs} workers...")
            with Pool(processes=min(self.n_jobs, 4)) as pool:
                pool.starmap(_create_visualization_wrapper, [
                    (task_name, func, base, self) for task_name, func, base in viz_tasks
                ])

    def _create_entrance_map(self, base_filename):
        """Map of parks and their generated entrance points."""
        print("Creating entrance map...")
        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot streets network first (background layer)
        self.streets_gdf.plot(
            ax=ax, color="lightgray", linewidth=0.5, alpha=0.6, zorder=0,
        )
        
        self.parks_gdf.plot(
            ax=ax, color="lightgreen", edgecolor="darkgreen",
            alpha=0.5, linewidth=1, zorder=1,
        )
        if len(self.entrances_gdf) > 0:
            self.entrances_gdf.plot(
                ax=ax, color="red", markersize=10, alpha=0.7, zorder=5,
            )
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax, facecolor="none", edgecolor="black", linewidth=2, zorder=6,
            )

        ax.set_title(f"Park Entrances - {self.city_name}", fontsize=16, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        fname = f"{base_filename}_entrances.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    def _create_parks_without_entrances_map(self, base_filename):
        """Diagnostic map highlighting parks without entrances."""
        if not self.parks_without_entrances:
            print("  All parks have entrances — skipping diagnostic map")
            return

        print("Creating parks-without-entrances map...")
        fig, ax = plt.subplots(figsize=(20, 20))

        no_entrance_set = set(self.parks_without_entrances)

        def _park_name(row, idx):
            return (
                row.get("name") or row.get("NAME")
                or row.get("park_name") or row.get("PARK_NAME")
                or f"Park_{idx}"
            )

        has_mask = self.parks_gdf.apply(
            lambda r: _park_name(r, r.name) not in no_entrance_set, axis=1
        )

        if has_mask.any():
            self.parks_gdf[has_mask].plot(
                ax=ax, color="lightgreen", edgecolor="darkgreen",
                alpha=0.5, linewidth=1, zorder=1,
            )
        if (~has_mask).any():
            self.parks_gdf[~has_mask].plot(
                ax=ax, color="red", edgecolor="darkred",
                alpha=0.7, linewidth=2, zorder=2,
            )
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax, facecolor="none", edgecolor="black", linewidth=2, zorder=3,
            )

        ax.set_title(
            f"Parks Without Entrances (Red) - {self.city_name}",
            fontsize=16, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="darkgreen", alpha=0.5,
                  label=f"With entrances ({has_mask.sum()})"),
            Patch(facecolor="red", edgecolor="darkred", alpha=0.7,
                  label=f"Without entrances ({(~has_mask).sum()})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

        plt.tight_layout()
        fname = f"{base_filename}_parks_without_entrances.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    def _calculate_density_cv(self, parcels_gdf, cell_size=500):
        """
        Calculate coefficient of variation (CV) of parcel density across grid cells.
        
        Parameters
        ----------
        parcels_gdf : GeoDataFrame
            Parcels to analyze
        cell_size : float
            Grid cell size in meters (default 500m)
            
        Returns
        -------
        float
            Coefficient of variation (std/mean) of parcels per cell
        """
        if self.boundary_gdf is None or len(parcels_gdf) == 0:
            return 0.0
            
        # Get boundary bounds
        xmin, ymin, xmax, ymax = self.boundary_gdf.total_bounds
        
        # Create grid cells
        x_cells = int(np.ceil((xmax - xmin) / cell_size))
        y_cells = int(np.ceil((ymax - ymin) / cell_size))
        
        # Count parcels in each cell
        parcels_per_cell = []
        for i in range(x_cells):
            for j in range(y_cells):
                cell_xmin = xmin + i * cell_size
                cell_ymin = ymin + j * cell_size
                cell_xmax = cell_xmin + cell_size
                cell_ymax = cell_ymin + cell_size
                
                # Create cell polygon
                from shapely.geometry import box
                cell = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
                
                # Count parcels intersecting this cell
                count = parcels_gdf[parcels_gdf.geometry.intersects(cell)].shape[0]
                if count > 0:  # Only include non-empty cells
                    parcels_per_cell.append(count)
        
        if len(parcels_per_cell) < 2:
            return 0.0
            
        # Calculate CV
        mean_density = np.mean(parcels_per_cell)
        std_density = np.std(parcels_per_cell)
        cv = std_density / mean_density if mean_density > 0 else 0.0
        
        return cv

    def _calculate_adaptive_alpha(self, cv):
        """
        Calculate adaptive alpha parameter based on density coefficient of variation.
        
        Parameters
        ----------
        cv : float
            Coefficient of variation of parcel density
            
        Returns
        -------
        float
            Alpha value between 0.3 and 1.0
            - 1.0 = fully proportional sampling (no density bias)
            - 0.3 = maximum equity compensation (strong oversampling of sparse areas)
        """
        # Cap CV to prevent extreme values
        cv_capped = min(cv, 2.5)
        
        # Calculate alpha: higher CV -> lower alpha -> more equity compensation
        alpha = max(0.3, min(1.0, 1.0 - (cv_capped * 0.28)))
        
        return alpha

    def _stratified_spatial_sample(self, parcels_gdf, n_samples, alpha, cell_size=500):
        """
        Perform stratified spatial sampling with density compensation.
        
        Uses vectorized spatial join and parallel processing for efficiency.
        
        Parameters
        ----------
        parcels_gdf : GeoDataFrame
            Parcels to sample from
        n_samples : int
            Total number of samples to draw
        alpha : float
            Density compensation factor (0.3-1.0)
            - 1.0 = proportional to density
            - <1.0 = oversample sparse areas
        cell_size : float
            Grid cell size in meters (default 500m)
            
        Returns
        -------
        GeoDataFrame
            Sampled parcels with spatial stratification
        """
        if len(parcels_gdf) <= n_samples:
            return parcels_gdf.copy()
            
        if self.boundary_gdf is None:
            # Fallback to random sampling if no boundary
            return parcels_gdf.sample(n=n_samples, random_state=42)
        
        # Get boundary bounds
        xmin, ymin, xmax, ymax = self.boundary_gdf.total_bounds
        
        # Create grid cells
        x_cells = int(np.ceil((xmax - xmin) / cell_size))
        y_cells = int(np.ceil((ymax - ymin) / cell_size))
        
        # Vectorized: Create all grid cells at once
        from shapely.geometry import box
        cells = []
        cell_ids = []
        
        for i in range(x_cells):
            for j in range(y_cells):
                cell_xmin = xmin + i * cell_size
                cell_ymin = ymin + j * cell_size
                cell_xmax = cell_xmin + cell_size
                cell_ymax = cell_ymin + cell_size
                
                cells.append(box(cell_xmin, cell_ymin, cell_xmax, cell_ymax))
                cell_ids.append((i, j))
        
        # Create GeoDataFrame of cells
        cells_gdf = gpd.GeoDataFrame(
            {'cell_id': range(len(cells))}, 
            geometry=cells, 
            crs=self.target_crs
        )
        
        # Vectorized spatial join: assign parcels to cells
        print(f"  Performing spatial join ({len(parcels_gdf)} parcels, {len(cells)} cells)...")
        parcels_with_cells = gpd.sjoin(
            parcels_gdf, 
            cells_gdf, 
            how='left', 
            predicate='intersects'
        )
        
        # Group by cell and calculate weights
        print(f"  Calculating sampling weights...")
        cell_groups = parcels_with_cells.groupby('cell_id')
        cell_samples = []
        total_weight = 0.0
        
        for cell_id, group in cell_groups:
            if len(group) > 0:
                # Weight = (parcel_count)^alpha
                weight = len(group) ** alpha
                cell_samples.append({
                    'cell_id': cell_id,
                    'indices': group.index.tolist(),
                    'weight': weight,
                    'count': len(group)
                })
                total_weight += weight
        
        if total_weight == 0 or len(cell_samples) == 0:
            # Fallback if spatial join failed
            return parcels_gdf.sample(n=n_samples, random_state=42)
        
        # Parallelize sampling from cells if many cells
        if len(cell_samples) > 100 and self.n_jobs > 1:
            # Allocate samples to cells
            for cell_data in cell_samples:
                cell_n = int(np.round(n_samples * cell_data['weight'] / total_weight))
                cell_data['target_n'] = max(1, min(cell_n, cell_data['count']))
            
            # Process cells in parallel
            with Pool(processes=self.n_jobs) as pool:
                sampled_indices = pool.starmap(_sample_cell_chunk, [
                    (cell_data['indices'], cell_data['target_n'])
                    for cell_data in cell_samples
                ])
            
            # Flatten results
            all_sampled_indices = [idx for chunk in sampled_indices for idx in chunk]
        else:
            # Sequential sampling for small grids
            all_sampled_indices = []
            for cell_data in cell_samples:
                # Calculate number of samples for this cell
                cell_n = int(np.round(n_samples * cell_data['weight'] / total_weight))
                cell_n = max(1, min(cell_n, cell_data['count']))
                
                # Sample from this cell
                if cell_n >= cell_data['count']:
                    all_sampled_indices.extend(cell_data['indices'])
                else:
                    sampled = np.random.choice(cell_data['indices'], size=cell_n, replace=False)
                    all_sampled_indices.extend(sampled.tolist())
        
        # Get sampled parcels
        result = parcels_gdf.loc[all_sampled_indices]
        
        # Adjust if we overshot or undershot due to rounding
        if len(result) > n_samples:
            result = result.sample(n=n_samples, random_state=42)
        elif len(result) < n_samples and len(result) < len(parcels_gdf):
            # Sample additional parcels to reach target
            remaining = parcels_gdf[~parcels_gdf.index.isin(result.index)]
            additional = min(n_samples - len(result), len(remaining))
            if additional > 0:
                result = pd.concat([result, remaining.sample(n=additional, random_state=42)])
        
        print(f"  Sampled {len(result)} parcels from {len(cell_samples)} grid cells")
        return result

    def _create_heatmap(self, base_filename):
        """IDW-interpolated heatmap with parks colored by parcel count."""
        print("Creating heatmap...")

        valid = self.parcels_gdf[self.parcels_gdf["park_distance"] < float("inf")].copy()
        if len(valid) == 0:
            print("  Warning: no valid distances for heatmap")
            return

        # Sample large datasets for performance
        # Higher values = more granular heatmap but slower processing
        # Recommended: 50k for good balance, up to 100k for maximum detail
        max_parcels = self.config.get("max_parcels_for_heatmap", 50000)
        if len(valid) > max_parcels:
            # Calculate density variation to determine sampling strategy
            cv = self._calculate_density_cv(valid, cell_size=500)
            alpha = self._calculate_adaptive_alpha(cv)
            
            print(f"  Sampling {max_parcels} of {len(valid)} parcels")
            print(f"  Density CV: {cv:.2f}, Alpha: {alpha:.2f} ({'proportional' if alpha > 0.8 else 'equity-compensated'})")
            
            # Use stratified spatial sampling with density compensation
            valid = self._stratified_spatial_sample(valid, max_parcels, alpha, cell_size=500)
        else:
            print(f"  Using all {len(valid)} parcels for heatmap")

        valid_km = valid["park_distance"] / 1000.0
        vmin_km = valid_km.min()
        vmax_km = np.percentile(valid_km, 95)

        # Build interpolation grid
        xmin, ymin, xmax, ymax = self.streets_gdf.total_bounds
        res = self.config.get("heatmap_resolution", 100)
        x_range = np.arange(xmin, xmax, res)
        y_range = np.arange(ymin, ymax, res)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        # Parcel coordinates
        coords = np.array([
            (g.centroid.x, g.centroid.y) if g.geom_type in ("Polygon", "MultiPolygon")
            else (g.x, g.y)
            for g in valid.geometry
        ])
        values = valid["park_distance"].values

        # IDW interpolation - parallelized for large grids
        k = self.config.get("heatmap_neighbors", 5)
        ptree = cKDTree(coords)
        
        # Parallelize IDW if grid is large
        if len(grid_points) > 100000 and self.n_jobs > 1:
            print(f"  Parallel IDW interpolation ({len(grid_points)} grid points, {self.n_jobs} workers)...")
            
            # Split grid into chunks for parallel processing
            n_chunks = min(self.n_jobs * 2, len(grid_points) // 10000)  # Optimal chunk size
            chunks = np.array_split(grid_points, n_chunks)
            
            # Process chunks in parallel
            with Pool(processes=self.n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(_idw_interpolation_chunk_wrapper, [
                        (chunk, coords, values, k) for chunk in chunks
                    ]),
                    total=len(chunks),
                    desc="  IDW interpolation",
                    unit="chunk"
                ))
            
            grid_vals = np.concatenate(results)
        else:
            print(f"  IDW interpolation ({len(grid_points)} grid points)...")
            dists, idxs = ptree.query(grid_points, k=k)
            weights = 1.0 / (dists + 1e-10)
            grid_vals = np.sum(weights * values[idxs], axis=1) / np.sum(weights, axis=1)
        
        grid_km = (grid_vals / 1000.0).reshape(len(y_range), len(x_range))

        # Gaussian smoothing
        print(f"  Applying Gaussian smoothing (sigma={self.config.get('heatmap_smoothing', 1)})...")
        sigma = self.config.get("heatmap_smoothing", 1)
        grid_smooth = gaussian_filter(grid_km, sigma=sigma)

        # Boundary mask - use vectorized approach (fastest)
        if self.boundary_gdf is not None:
            print(f"  Applying boundary mask...")
            try:
                from shapely.vectorized import contains
                boundary_geom = self.boundary_gdf.iloc[0].geometry
                inside = contains(boundary_geom, grid_x.ravel(), grid_y.ravel())
                grid_smooth[~inside.reshape(grid_smooth.shape)] = np.nan
            except ImportError:
                # Fallback: parallel boundary masking if vectorized not available
                print("  Warning: shapely.vectorized not available, using slower fallback")
                from shapely.prepared import prep
                boundary_geom = self.boundary_gdf.iloc[0].geometry
                prepared = prep(boundary_geom)
                
                if len(grid_points) > 100000 and self.n_jobs > 1:
                    # Parallel masking for large grids
                    chunks = np.array_split(grid_points, self.n_jobs * 2)
                    with Pool(processes=self.n_jobs) as pool:
                        results = list(tqdm(
                            pool.starmap(_boundary_mask_chunk, [
                                (chunk, prepared) for chunk in chunks
                            ]),
                            total=len(chunks),
                            desc="  Boundary masking",
                            unit="chunk"
                        ))
                    mask = np.concatenate(results).reshape(grid_smooth.shape)
                else:
                    # Sequential for small grids
                    mask = np.array([
                        prepared.contains(Point(x, y))
                        for x, y in grid_points
                    ]).reshape(grid_smooth.shape)
                
                grid_smooth[~mask] = np.nan

        # Plot
        print(f"  Rendering heatmap visualization...")
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot boundary first as background with light grey fill
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax, facecolor="lightgrey", edgecolor="none", alpha=1.0, zorder=0,
            )

        im = ax.imshow(
            grid_smooth,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="viridis_r",
            norm=mcolors.Normalize(vmin=vmin_km, vmax=vmax_km),
            alpha=0.7,
            zorder=1,
        )

        self.streets_gdf.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4, zorder=2)

        # Color parks by parcel count - vectorized for performance
        if self.park_parcel_counts:
            counts_list = [
                self.park_parcel_counts.get(idx, 0)
                for idx in self.parks_gdf.index
            ]
            
            # Ensure we have valid counts
            if not counts_list or all(c == 0 for c in counts_list):
                # All parks have zero parcels - just plot them in gray
                self.parks_gdf.plot(
                    ax=ax, color="lightgray", edgecolor="darkgreen",
                    alpha=1.0, linewidth=1.5, zorder=3,
                )
            else:
                max_count = max(counts_list)
                mean_count = sum(counts_list) / len(counts_list)
                
                # Create simple gradient colormap: pale light green -> dark green
                from matplotlib.colors import LinearSegmentedColormap
                import matplotlib.colors as mcolors
                colors = ['#E8F5E9', '#006400']  # Pale light green to dark green
                cmap_parks = LinearSegmentedColormap.from_list('park_usage', colors, N=256)
                
                # Simple normalization from 0 to max
                norm_parks = mcolors.Normalize(vmin=0, vmax=max_count)

                # Vectorized: compute all colors at once, then plot once
                # Ensure all values are valid before normalization
                park_colors = []
                for idx, c in zip(self.parks_gdf.index, counts_list):
                    if c == 0:
                        park_colors.append("lightgray")
                    else:
                        try:
                            # Get RGBA tuple and ensure it's a proper tuple
                            rgba = cmap_parks(norm_parks(c))
                            # Convert to standard tuple format (R, G, B, A)
                            if hasattr(rgba, '__iter__') and len(rgba) >= 3:
                                park_colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]) if len(rgba) > 3 else 1.0))
                            else:
                                park_colors.append("lightgray")
                        except (ValueError, TypeError, IndexError):
                            # Fallback to gray if normalization fails
                            park_colors.append("lightgray")
                
                # Convert to pandas Series with proper index alignment
                import pandas as pd
                park_colors_series = pd.Series(park_colors, index=self.parks_gdf.index)
                
                # Single plot call for all parks (much faster than loop)
                try:
                    self.parks_gdf.plot(
                        ax=ax, 
                        color=park_colors_series,
                        edgecolor="darkgreen",
                        alpha=1.0, 
                        linewidth=1.5, 
                        zorder=3,
                    )

                    sm = plt.cm.ScalarMappable(cmap=cmap_parks, norm=norm_parks)
                    sm.set_array([])
                    cbar_parks = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.01, location="left")
                    cbar_parks.set_label("Parcels closest to park")
                except Exception as e:
                    # If coloring fails, fall back to simple green
                    print(f"  Warning: Park coloring failed ({e}), using simple colors")
                    self.parks_gdf.plot(
                        ax=ax, color="green", edgecolor="darkgreen",
                        alpha=1.0, linewidth=1.5, zorder=3,
                    )
        else:
            self.parks_gdf.plot(
                ax=ax, color="green", edgecolor="darkgreen",
                alpha=1.0, linewidth=1.5, zorder=3,
            )

        if self.boundary_gdf is not None:
            # Plot boundary outline on top
            self.boundary_gdf.plot(
                ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, zorder=5,
            )

        cbar = fig.colorbar(im, ax=ax, extend="max", shrink=0.7)
        cbar.set_label("Distance to nearest park (km, LTS-weighted)")

        ax.set_title(
            f"Park Accessibility Heatmap - {self.city_name}", fontsize=16, fontweight="bold"
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Remove axis frame for cleaner look

        plt.tight_layout(pad=0.1)  # Minimal padding
        fname = f"{base_filename}_heatmap.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.05)  # Tight crop with minimal padding
        plt.close()
        print(f"  Saved {fname}")

    def _create_distance_histograms(self, base_filename):
        """Two-panel histogram: distance distribution + parcels-per-park."""
        print("Creating histograms...")

        valid = self.parcels_gdf[self.parcels_gdf["park_distance"] < float("inf")].copy()
        if len(valid) == 0:
            print("  Warning: no valid distances for histograms")
            return

        # Pre-compute all statistics at once (vectorized)
        dist_km = valid["park_distance"] / 1000.0
        dist_stats = {
            'mean': dist_km.mean(),
            'median': dist_km.median(),
            'p95': dist_km.quantile(0.95),
            'min': dist_km.min(),
            'max': dist_km.max(),
            'std': dist_km.std(),
            'count': len(valid)
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # --- Panel 1: Distance distribution ---
        ax1.hist(dist_km, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax1.axvline(dist_stats['mean'], color="red", ls="--", lw=2, label=f"Mean: {dist_stats['mean']:.2f} km")
        ax1.axvline(dist_stats['median'], color="orange", ls="--", lw=2, label=f"Median: {dist_stats['median']:.2f} km")
        ax1.axvline(dist_stats['p95'], color="purple", ls="--", lw=2, label=f"95th %ile: {dist_stats['p95']:.2f} km")
        ax1.set_xlabel("Parkximity Distance (km)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Parcels", fontsize=12, fontweight="bold")
        ax1.set_title("Distribution of Parkximity Distances", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, ls=":", lw=0.5)

        stats_text = (
            f"Parcels: {dist_stats['count']:,}\n"
            f"Mean: {dist_stats['mean']:.2f} km\n"
            f"Median: {dist_stats['median']:.2f} km\n"
            f"Min: {dist_stats['min']:.2f} km\n"
            f"Max: {dist_stats['max']:.2f} km\n"
            f"Std: {dist_stats['std']:.2f} km"
        )
        ax1.text(
            0.97, 0.97, stats_text, transform=ax1.transAxes, va="top", ha="right",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.5"),
        )

        # --- Panel 2: Parcels-per-park distribution ---
        if self.park_parcel_counts:
            counts = np.array(list(self.park_parcel_counts.values()))
            
            # Pre-compute park statistics
            park_stats = {
                'mean': counts.mean(),
                'median': np.median(counts),
                'max': counts.max(),
                'with_parcels': len(counts),
                'without_parcels': len(self.parks_gdf) - len(counts)
            }
            
            ax2.hist(counts, bins=min(50, max(len(counts) // 2, 10)),
                     color="forestgreen", alpha=0.7, edgecolor="black", linewidth=0.5)
            ax2.axvline(park_stats['mean'], color="red", ls="--", lw=2, label=f"Mean: {park_stats['mean']:.1f}")
            ax2.axvline(park_stats['median'], color="orange", ls="--", lw=2, label=f"Median: {park_stats['median']:.1f}")
            ax2.set_xlabel("Number of Closest Parcels", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Number of Parks", fontsize=12, fontweight="bold")
            ax2.set_title("Parcels per Nearest Park", fontsize=14, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, ls=":", lw=0.5)

            stats_text2 = (
                f"Parks (with parcels): {park_stats['with_parcels']}\n"
                f"Parks (no parcels): {park_stats['without_parcels']}\n"
                f"Mean: {park_stats['mean']:.1f}\n"
                f"Median: {park_stats['median']:.1f}\n"
                f"Max: {park_stats['max']}"
            )
            ax2.text(
                0.97, 0.97, stats_text2, transform=ax2.transAxes, va="top", ha="right",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.5"),
            )
        else:
            ax2.text(0.5, 0.5, "No park-parcel data available",
                     transform=ax2.transAxes, ha="center", fontsize=14)

        fig.suptitle(
            f"Parkximity Analysis - {self.city_name}",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fname = f"{base_filename}_histograms.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_parcels_geojson(self):
        """Export parcels with parkximity columns as WGS84 GeoJSON."""
        print(f"Exporting parcel GeoJSON for {self.city_name}...")

        if len(self.parcels_gdf) == 0:
            print("  Warning: no parcels to export")
            return None

        # Create city-specific subfolder in Exports/Data directory
        exports_dir = Path(self.config.get("exports_output_dir", "Exports"))
        data_dir = exports_dir / "Data" / self.city_name.replace(' ', '_').replace(',', '')
        data_dir.mkdir(parents=True, exist_ok=True)

        if "park_distance" in self.parcels_gdf.columns:
            self.parcels_gdf["park_distance_km"] = self.parcels_gdf["park_distance"] / 1000.0

        export_cols = ["geometry"]
        for col in [
            "park_distance", "park_distance_km",
            "parcel_to_node_distance", "node_to_park_distance",
            "nearest_park_id",
        ]:
            if col in self.parcels_gdf.columns:
                export_cols.append(col)

        export_gdf = self.parcels_gdf[export_cols].copy().to_crs("EPSG:4326")
        
        # Replace inf values with None for valid GeoJSON
        for col in export_gdf.columns:
            if col != "geometry" and export_gdf[col].dtype in [np.float64, np.float32]:
                export_gdf[col] = export_gdf[col].replace([np.inf, -np.inf], None)
        
        fname = output_dir / f"{self.city_name.lower().replace(' ', '_')}_parcels_parkximity.geojson"
        export_gdf.to_file(fname, driver="GeoJSON")

        valid = export_gdf[export_gdf["park_distance_km"].notna() & (export_gdf["park_distance_km"] < float("inf"))]["park_distance_km"]
        if len(valid) > 0:
            print(f"  Stats (km): min={valid.min():.3f}, max={valid.max():.3f}, "
                  f"mean={valid.mean():.3f}, median={valid.median():.3f}")
        print(f"  Exported {len(export_gdf)} parcels to {fname}")
        return str(fname)

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def full_analysis(self):
        """Run the complete parkximity analysis pipeline."""
        # Guard against re-running analysis
        if hasattr(self, '_analysis_complete') and self._analysis_complete:
            print(f"\n⚠ Analysis already complete for {self.city_name}, skipping...")
            return self.parcels_gdf
        
        print(f"\n{'=' * 60}")
        print(f"  ParkXimity Analysis: {self.city_name}")
        print(f"{'=' * 60}\n")

        self.load_data()
        self.generate_park_entrances()
        self.build_network()
        self.find_parkximity()
        self.create_visualizations()
        self.export_parcels_geojson()

        self._analysis_complete = True
        print(f"\nAnalysis complete for {self.city_name}!")
        return self.parcels_gdf


# ======================================================================
# Multi-City Runner
# ======================================================================

def multi_city_analysis(cities_config, target_crs="EPSG:6350", global_settings=None,
                       force_cpu=False, verbose=True, parallel_cities=False):
    """
    Run parkximity analysis for multiple cities with automatic GPU detection.
    Optionally run census analysis for cities with census configuration.
    
    This function automatically detects GPU availability and uses the fastest
    implementation available. No manual configuration needed!

    Parameters
    ----------
    cities_config : dict
        Mapping of city names to config dicts. Each city config can include:
        
        Parkximity settings:
            - parcels_path, parks_path, streets_path, boundary_path, etc.
        
        Census analysis settings (optional):
            - run_census_analysis: bool, enable census analysis for this city
            - census_state_fips: str, state FIPS code (e.g., '25')
            - census_county_fips: list, county FIPS codes (e.g., ['025'])
            - census_block_shapefile: str, path to census block shapefile
            - census_parcels_path: str, path to parkximity output for census join
    
    target_crs : str
        Target CRS for all cities.
    
    global_settings : dict, optional
        Global settings applied to all cities. Individual city configs
        can override these. Common settings include:
        
        Parkximity settings:
            - park_buffer: Buffer in meters for entrance detection (default 50)
            - entrance_tolerance: Dedup tolerance in meters (default 5)
            - entrance_batch_size: Parks per batch for parallel processing (default 50)
            - heatmap_resolution: Grid cell size in meters (default 100)
            - heatmap_neighbors: IDW neighbor count (default 5)
            - heatmap_smoothing: Gaussian sigma (default 1)
            - n_jobs: Number of parallel workers per city (default: cpu_count() - 1)
        
        Census analysis settings (optional):
            - census_api_key: str, Census API key
            - census_acs_year: int, ACS year (default 2022)
            - census_n_jobs: int, parallel workers for census (default: same as n_jobs)
            - census_output_dir: str, output directory (default: 'Data/Processed/Demographics')
    
    force_cpu : bool
        If True, force CPU mode even if GPU is available.
    
    verbose : bool
        If True, print which implementation is being used for each city.
    
    parallel_cities : bool
        If True, process multiple cities in parallel (default: False).
        WARNING: Parallel city processing disables within-city parallelization,
        which is typically much less efficient. Only use for small datasets
        where within-city operations are fast.

    Returns
    -------
    tuple : (parkximity_results, census_results)
        - parkximity_results: dict of city_name -> parcels GeoDataFrame
        - census_results: dict of city_name -> census-joined GeoDataFrame (or empty dict)
    
    Examples
    --------
    >>> # Sequential cities with parallel processing within each (RECOMMENDED)
    >>> parkximity_results, census_results = multi_city_analysis(cities_config)
    
    >>> # With census analysis enabled
    >>> cities_config = {
    ...     'Boston': {
    ...         'parcels_path': '...',
    ...         'parks_path': '...',
    ...         'streets_path': '...',
    ...         'run_census_analysis': True,
    ...         'census_state_fips': '25',
    ...         'census_county_fips': ['025'],
    ...         'census_block_shapefile': '...',
    ...         'census_parcels_path': 'Data/Processed/ParkXimity/boston_parcels_parkximity.geojson'
    ...     }
    ... }
    >>> global_settings = {
    ...     'census_api_key': 'your_key_here',
    ...     'census_acs_year': 2022
    ... }
    >>> parkximity_results, census_results = multi_city_analysis(
    ...     cities_config, global_settings=global_settings
    ... )
    """
    if global_settings is None:
        global_settings = {}
    
    # Print overall status banner
    if verbose:
        print("\n" + "="*70)
        print("  PARKXIMITY MULTI-CITY ANALYSIS")
        print("="*70)
        print(f"  Cities to process: {len(cities_config)}")
        print(f"  Target CRS: {target_crs}")
        if parallel_cities and len(cities_config) > 1:
            print(f"  Mode: Parallel city processing (within-city parallelization disabled)")
        else:
            print(f"  Mode: Sequential city processing (within-city parallelization enabled)")
        print("="*70 + "\n")
    
    # Check GPU status once at the beginning
    gpu_available = False
    gpu_info = None
    show_status_per_city = False  # Only show status once
    
    if not force_cpu:
        try:
            from gpu_utils import get_gpu_capabilities, prompt_gpu_installation_if_needed
            gpu_caps = get_gpu_capabilities()
            gpu_available = gpu_caps.is_available()
            
            if gpu_available:
                gpu_info = gpu_caps.get_summary()
                if verbose:
                    print("="*70)
                    print("🚀 GPU ACCELERATION ENABLED")
                    print("="*70)
                    print(f"  Device: {gpu_info['gpu_names'][0]}")
                    print(f"  Memory: {gpu_info['total_memory_gb'][0]:.1f} GB VRAM")
                    print(f"  Mode: GPU-accelerated processing")
                    print("="*70 + "\n")
            else:
                # Check if GPU hardware exists but libraries missing
                should_retry = prompt_gpu_installation_if_needed()
                if should_retry:
                    print("\n⚠ Please restart your Python session and run again.")
                    raise SystemExit(0)
                
                # If we get here, user declined installation or no GPU hardware
                if verbose:
                    from gpu_utils import check_nvidia_gpu_hardware
                    has_gpu_hw, _ = check_nvidia_gpu_hardware()
                    
                    print("="*70)
                    print("💻 CPU MODE")
                    print("="*70)
                    if has_gpu_hw:
                        print("  GPU libraries not installed")
                    else:
                        print("  No GPU detected")
                    print("  Mode: CPU-only processing")
                    print("="*70 + "\n")
        except:
            if verbose:
                print("="*70)
                print("💻 CPU MODE")
                print("="*70)
                print("  Mode: CPU-only processing")
                print("="*70 + "\n")
    else:
        if verbose:
            print("="*70)
            print("💻 CPU MODE (forced)")
            print("="*70)
            print("  Mode: CPU-only processing")
            print("="*70 + "\n")

    # Process cities sequentially (RECOMMENDED) or in parallel
    if parallel_cities and len(cities_config) > 1:
        # Parallel processing of cities (disables within-city parallelization)
        # This is less efficient but may be useful for small datasets
        print(f"⚠ Warning: Parallel city mode disables within-city parallelization")
        print(f"  Processing {len(cities_config)} cities using {min(len(cities_config), max(1, cpu_count() // 2))} parallel workers\n")
        
        city_tasks = [
            (city_name, {**global_settings, **config}, target_crs, force_cpu, verbose)
            for city_name, config in cities_config.items()
        ]
        
        n_city_workers = min(len(cities_config), max(1, cpu_count() // 2))
        
        with Pool(processes=n_city_workers) as pool:
            city_results = pool.starmap(_process_single_city, city_tasks)
        
        results = {}
        errors = {}
        for city_name, parcels_gdf, error_msg in city_results:
            results[city_name] = parcels_gdf
            if error_msg:
                errors[city_name] = error_msg
    else:
        # Sequential processing (RECOMMENDED - enables within-city parallelization)
        if verbose and len(cities_config) > 1:
            print(f"Processing {len(cities_config)} cities sequentially with full parallelization within each city\n")
        
        results = {}
        errors = {}
        for i, (city_name, config) in enumerate(cities_config.items(), 1):
            if verbose and len(cities_config) > 1:
                print(f"\n{'='*70}")
                print(f"  CITY {i}/{len(cities_config)}: {city_name.upper()}")
                print(f"{'='*70}\n")
            
            try:
                merged_config = {**global_settings, **config}
                analyzer = create_parkximity_analyzer(
                    city_name, merged_config, target_crs, 
                    force_cpu=force_cpu, verbose=verbose, _show_status=False
                )
                results[city_name] = analyzer.full_analysis()
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"❌ ERROR processing {city_name}")
                print(f"{'='*70}")
                print(f"Error: {str(e)}")
                print(f"{'='*70}\n")
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                errors[city_name] = error_msg
                traceback.print_exc()
                print(f"\nSkipping {city_name} and continuing with next city...\n")
                results[city_name] = None
    
    # Print summary of results
    if verbose and len(cities_config) > 1:
        print(f"\n{'='*70}")
        print("  MULTI-CITY ANALYSIS SUMMARY")
        print(f"{'='*70}")
        successful = [city for city, result in results.items() if result is not None]
        failed = [city for city, result in results.items() if result is None]
        
        print(f"  Total cities: {len(cities_config)}")
        print(f"  ✓ Successful: {len(successful)}")
        if successful:
            for city in successful:
                print(f"    - {city}")
        
        if failed:
            print(f"  ✗ Failed: {len(failed)}")
            for city in failed:
                error_msg = errors.get(city, "Unknown error")
                print(f"    - {city}")
                print(f"      Error: {error_msg}")
        print(f"{'='*70}\n")
    
    # Run census analysis if requested
    census_results = _run_census_analysis_if_requested(
        cities_config, global_settings, results, verbose
    )
    
    return results, census_results


# ======================================================================
# Census Analysis Integration
# ======================================================================

def _run_census_analysis_if_requested(cities_config, global_settings, parkximity_results, verbose):
    """
    Run census analysis for cities that have census configuration and successful parkximity results.
    
    Parameters
    ----------
    cities_config : dict
        City configurations with optional census settings
    global_settings : dict
        Global settings that may include census parameters
    parkximity_results : dict
        Results from parkximity analysis
    verbose : bool
        Whether to print progress messages
    
    Returns
    -------
    dict : city_name -> census analysis results (or None if not requested/failed)
    """
    # Check if any cities have census analysis enabled
    cities_with_census = {
        city_name: config 
        for city_name, config in cities_config.items()
        if config.get('run_census_analysis', False) and parkximity_results.get(city_name) is not None
    }
    
    if not cities_with_census:
        return {}
    
    # Import census analysis module
    try:
        from CensusAnalysis import multi_city_census_analysis
    except ImportError:
        if verbose:
            print("\n⚠ Warning: CensusAnalysis.py not found. Skipping census analysis.")
        return {}
    
    if verbose:
        print(f"\n{'='*70}")
        print("  CENSUS ANALYSIS")
        print(f"{'='*70}")
        print(f"  Cities with census analysis enabled: {len(cities_with_census)}")
        print(f"{'='*70}\n")
    
    # Build census configuration
    census_cities_config = {}
    for city_name, config in cities_with_census.items():
        # Extract census-specific configuration
        census_config = {
            'state_fips': config.get('census_state_fips'),
            'county_fips': config.get('census_county_fips'),
            'block_shapefile': config.get('census_block_shapefile'),
            'boundary_geojson': config.get('boundary_path'),  # Reuse boundary from parkximity
            'parcels_path': config.get('census_parcels_path'),  # Path to parkximity output
        }
        
        # Validate required fields
        required_fields = ['state_fips', 'county_fips', 'block_shapefile', 'parcels_path']
        missing_fields = [f for f in required_fields if not census_config.get(f)]
        
        if missing_fields:
            if verbose:
                print(f"  ⚠ Skipping {city_name}: missing census config fields: {missing_fields}")
            continue
        
        census_cities_config[city_name] = census_config
    
    if not census_cities_config:
        if verbose:
            print("  No cities with complete census configuration. Skipping census analysis.\n")
        return {}
    
    # Build global census configuration
    census_global_config = {
        'census_api_key': global_settings.get('census_api_key', 'YOUR_API_KEY_HERE'),
        'acs_year': global_settings.get('census_acs_year', 2022),
        'n_jobs': global_settings.get('census_n_jobs', global_settings.get('n_jobs', max(1, cpu_count() - 1))),
        'output_dir': global_settings.get('census_output_dir', 'Data/Processed/Demographics')
    }
    
    # Run census analysis
    try:
        census_results = multi_city_census_analysis(census_cities_config, census_global_config)
        return census_results
    except Exception as e:
        if verbose:
            print(f"\n❌ Census analysis failed: {e}")
            import traceback
            traceback.print_exc()
        return {}


# ======================================================================
# Parallel Processing Helper Functions
# ======================================================================

def _process_street_network_chunk(chunk_info):
    """
    Helper function for parallel street network edge extraction.
    
    Parameters
    ----------
    chunk_info : tuple
        (chunk_gdf, lts_column_name)
    
    Returns
    -------
    list
        List of edge tuples (start, end, attributes)
    """
    from shapely.geometry import LineString
    
    chunk_gdf, lts_col = chunk_info
    edges = []
    
    for geom, lts in zip(chunk_gdf.geometry, chunk_gdf[lts_col]):
        coords = list(geom.coords)
        for j in range(len(coords) - 1):
            start, end = coords[j], coords[j + 1]
            length = LineString([start, end]).length
            if length > 0:
                edges.append((
                    start, end,
                    {"weight": length * (1 + lts), "length": length, "lts": lts},
                ))
    
    return edges

def _validate_geometry_chunk(chunk_gdf):
    """
    Helper function for parallel geometry validation with optimized vectorization.
    
    Parameters
    ----------
    chunk_gdf : GeoDataFrame
        Chunk of geometries to validate
    
    Returns
    -------
    GeoDataFrame
        Filtered chunk containing only valid, non-empty geometries
    """
    # Fully vectorized validation - single pass through data
    # Use bitwise AND to combine conditions efficiently
    valid_mask = (
        chunk_gdf.geometry.notna() &  # Not null
        ~chunk_gdf.geometry.is_empty &  # Not empty
        chunk_gdf.geometry.is_valid  # Valid geometry
    )
    
    return chunk_gdf.loc[valid_mask].copy() if valid_mask.any() else gpd.GeoDataFrame(geometry=[], crs=chunk_gdf.crs)

def _clip_geometries_chunk(chunk_gdf, boundary_geom):
    """
    Helper function for parallel geometry clipping.
    
    Parameters
    ----------
    chunk_gdf : GeoDataFrame
        Chunk of geometries to check
    boundary_geom : shapely.geometry
        Boundary geometry to clip against
    
    Returns
    -------
    GeoDataFrame
        Filtered chunk containing only geometries that intersect boundary
    """
    return chunk_gdf[chunk_gdf.geometry.intersects(boundary_geom)].copy()


def _process_parcel_json_chunk(chunk_info):
    """
    Helper function for parallel JSON export of parcels.
    
    Parameters
    ----------
    chunk_info : tuple
        (chunk_gdf, nodes_array, paths_dict)
    
    Returns
    -------
    list
        List of parcel dictionaries for JSON export
    """
    chunk_gdf, nodes_array, paths_dict = chunk_info
    
    # Build KD-tree for this chunk
    tree = cKDTree(nodes_array)
    
    parcels_data = []
    for parcel_idx, parcel in chunk_gdf.iterrows():
        geom = parcel.geometry
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            cx, cy = geom.centroid.x, geom.centroid.y
        else:
            cx, cy = geom.x, geom.y

        p2n = parcel.get("parcel_to_node_distance", float("inf"))
        n2p = parcel.get("node_to_park_distance", float("inf"))
        total = parcel.get("park_distance", float("inf"))
        npid = parcel.get("nearest_park_id", -1)

        # Get path for this parcel
        _, nn_idx = tree.query([cx, cy])
        nearest_node = tuple(nodes_array[nn_idx])
        path = paths_dict.get(nearest_node, [])
        route = [[float(c) for c in coord] for coord in path]

        parcels_data.append({
            "id": int(parcel_idx) if isinstance(parcel_idx, (int, np.integer)) else str(parcel_idx),
            "centroid": [float(cx), float(cy)],
            "parkximity": float(total) if total != float("inf") else None,
            "parcel_to_node_distance": float(p2n) if p2n != float("inf") else None,
            "node_to_park_distance": float(n2p) if n2p != float("inf") else None,
            "nearest_park_id": int(npid) if isinstance(npid, (int, np.integer)) and npid != -1 else None,
            "route": route,
        })
    
    return parcels_data


def _validate_park_geometry(park_tuple):
    """Helper function for parallel park geometry validation."""
    idx, row = park_tuple
    geom = row.geometry
    
    if geom is None or geom.is_empty:
        return None
        
    if not geom.is_valid:
        try:
            fixed = geom.buffer(0)
            if fixed.is_valid and not fixed.is_empty:
                row = row.copy()
                row["geometry"] = fixed
                row['_repaired'] = True
                return row
        except Exception:
            pass
        return None
    
    return row


def _process_park_batch(batch_info, streets_gdf, streets_tree, buffer_distance, entrance_tolerance):
    """Helper function for parallel park entrance batch processing."""
    batch_parks, batch_start, batch_end, total = batch_info
    
    batch_coords = []
    batch_data = []
    no_access = []

    for park_idx, park in batch_parks.iterrows():
        park_geom = park.geometry
        park_name = (
            park.get("name")
            or park.get("NAME")
            or park.get("park_name")
            or park.get("PARK_NAME")
            or f"Park_{park_idx}"
        )

        buffered = park_geom.buffer(buffer_distance)
        candidates = streets_tree.query(buffered)

        if len(candidates) == 0:
            no_access.append(park_name)
            continue

        intersecting = streets_gdf.iloc[candidates]
        intersecting = intersecting[intersecting.intersects(buffered)]

        if len(intersecting) == 0:
            no_access.append(park_name)
            continue

        park_points = []
        for street_idx, street in intersecting.iterrows():
            street_geom = street.geometry
            
            # Handle MultiLineString by processing each component
            if street_geom.geom_type == 'MultiLineString':
                street_parts = list(street_geom.geoms)
            else:
                street_parts = [street_geom]
            
            # Process each street segment
            for street_part in street_parts:
                try:
                    # Try intersection with exact boundary first
                    intersection = street_part.intersection(park_geom.boundary)
                    
                    # If no exact intersection, try with buffered boundary
                    if intersection.is_empty:
                        # Buffer the park boundary slightly to catch near-misses
                        buffered_boundary = park_geom.boundary.buffer(buffer_distance)
                        intersection = street_part.intersection(buffered_boundary)
                        
                        # If still empty, use nearest point approach
                        if intersection.is_empty:
                            nearest_pt = park_geom.boundary.interpolate(
                                park_geom.boundary.project(street_part)
                            )
                            if nearest_pt and not nearest_pt.is_empty:
                                if nearest_pt.distance(street_part) <= buffer_distance:
                                    park_points.append({
                                        "coord": (nearest_pt.x, nearest_pt.y),
                                        "park_name": park_name,
                                        "park_id": park_idx,
                                        "street_id": street_idx,
                                    })
                            continue
                        
                except Exception:
                    continue

                # Extract all points from the intersection
                for pt in _extract_points_from_intersection(intersection):
                    if pt is not None and not pt.is_empty:
                        park_points.append({
                            "coord": (pt.x, pt.y),
                            "park_name": park_name,
                            "park_id": park_idx,
                            "street_id": street_idx,
                        })

        if park_points:
            # Per-park deduplication
            unique = _deduplicate_points(park_points, entrance_tolerance)
            for p in unique:
                batch_coords.append(p["coord"])
                batch_data.append({
                    "park_name": p["park_name"],
                    "park_id": p["park_id"],
                    "street_id": p["street_id"],
                })
        else:
            # Fallback: nearest point on boundary to closest street
            try:
                nearest_street = intersecting.iloc[0]
                nearest_pt = park_geom.boundary.interpolate(
                    park_geom.boundary.project(nearest_street.geometry.centroid)
                )
                if nearest_pt and not nearest_pt.is_empty:
                    batch_coords.append((nearest_pt.x, nearest_pt.y))
                    batch_data.append({
                        "park_name": park_name,
                        "park_id": park_idx,
                        "street_id": nearest_street.name,
                    })
            except Exception:
                no_access.append(park_name)

    return batch_coords, batch_data, no_access


def _extract_points_from_intersection(intersection):
    """Extract Point objects from various geometry types."""
    points = []
    gtype = intersection.geom_type

    if gtype == "Point":
        points.append(intersection)
    elif gtype == "MultiPoint":
        points.extend(list(intersection.geoms))
    elif gtype == "LineString":
        coords = list(intersection.coords)
        if len(coords) >= 2:
            # Always add endpoints
            points.extend([Point(coords[0]), Point(coords[-1])])
            
            # For long shared boundaries, add intermediate points
            # This catches cases where streets run along park edges
            line_length = intersection.length
            if line_length > 50:  # If shared boundary > 50m, add intermediate points
                # Add points every ~50m along the shared boundary
                num_intermediate = int(line_length / 50)
                for i in range(1, num_intermediate + 1):
                    distance = (i * line_length) / (num_intermediate + 1)
                    pt = intersection.interpolate(distance)
                    if pt and not pt.is_empty:
                        points.append(pt)
    elif gtype == "MultiLineString":
        for line in intersection.geoms:
            coords = list(line.coords)
            if len(coords) >= 2:
                # Always add endpoints
                points.extend([Point(coords[0]), Point(coords[-1])])
                
                # For long shared boundaries, add intermediate points
                line_length = line.length
                if line_length > 50:  # If shared boundary > 50m
                    num_intermediate = int(line_length / 50)
                    for i in range(1, num_intermediate + 1):
                        distance = (i * line_length) / (num_intermediate + 1)
                        pt = line.interpolate(distance)
                        if pt and not pt.is_empty:
                            points.append(pt)
    elif gtype == "GeometryCollection":
        for geom in intersection.geoms:
            points.extend(_extract_points_from_intersection(geom))

    return points


def _deduplicate_points(point_dicts, tolerance):
    """Deduplicate points within a single park using KD-tree."""
    if len(point_dicts) <= 1:
        return point_dicts
    coords = np.array([p["coord"] for p in point_dicts])
    tree = cKDTree(coords)
    remove = set()
    for i, j in tree.query_pairs(r=tolerance):
        if i not in remove:
            remove.add(j)
    return [p for idx, p in enumerate(point_dicts) if idx not in remove]


def _sample_cell_chunk(indices, target_n):
    """
    Sample indices from a cell chunk.
    
    Used for parallel stratified sampling.
    
    Parameters
    ----------
    indices : list
        Parcel indices in this cell
    target_n : int
        Number of samples to draw
    
    Returns
    -------
    list
        Sampled indices
    """
    if target_n >= len(indices):
        return indices
    else:
        return np.random.choice(indices, size=target_n, replace=False).tolist()


def _idw_interpolation_chunk(grid_chunk, coords, values, k):
    """
    Perform IDW interpolation on a chunk of grid points.
    
    Used for parallel processing of large grids.
    
    Parameters
    ----------
    grid_chunk : np.ndarray
        Subset of grid points to interpolate (N, 2)
    coords : np.ndarray
        Parcel coordinates (M, 2)
    values : np.ndarray
        Parcel distance values (M,)
    k : int
        Number of nearest neighbors for IDW
    
    Returns
    -------
    np.ndarray
        Interpolated values for grid_chunk (N,)
    """
    from scipy.spatial import cKDTree
    
    ptree = cKDTree(coords)
    dists, idxs = ptree.query(grid_chunk, k=k)
    weights = 1.0 / (dists + 1e-10)
    return np.sum(weights * values[idxs], axis=1) / np.sum(weights, axis=1)


def _idw_interpolation_chunk_wrapper(args):
    """Wrapper for _idw_interpolation_chunk to work with pool.imap."""
    return _idw_interpolation_chunk(*args)


def _boundary_mask_chunk(points_chunk, prepared_geom):
    """
    Check which points in a chunk are inside the boundary.
    
    Used for parallel processing of large grids when vectorized
    shapely is not available.
    
    Parameters
    ----------
    points_chunk : np.ndarray
        Subset of grid points to check (N, 2)
    prepared_geom : shapely.prepared.PreparedGeometry
        Prepared boundary geometry
    
    Returns
    -------
    np.ndarray
        Boolean mask indicating which points are inside (N,)
    """
    from shapely.geometry import Point
    
    return np.array([
        prepared_geom.contains(Point(x, y))
        for x, y in points_chunk
    ])


def _create_visualization_wrapper(task_name, func, base, analyzer):
    """Wrapper for parallel visualization creation."""
    try:
        func(base)
    except Exception as e:
        print(f"  Warning: Failed to create {task_name}: {e}")


def _process_single_city(city_name, merged_config, target_crs, force_cpu, verbose):
    """Helper function for parallel city processing."""
    try:
        analyzer = create_parkximity_analyzer(
            city_name, merged_config, target_crs, 
            force_cpu=force_cpu, verbose=verbose, _show_status=False
        )
        parcels_gdf = analyzer.full_analysis()
        return city_name, parcels_gdf, None
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR processing {city_name}")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        print(f"{'='*70}\n")
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
        return city_name, None, error_msg


# ======================================================================
# Example Configurations
# ======================================================================

if __name__ == "__main__":
    # Global visualization settings applied to all cities
    global_settings = {
        "park_buffer": 20.0,
        "entrance_tolerance": 2.0,
        "entrance_batch_size": 200,  # Increased from default 50 for better performance
        "heatmap_resolution": 100,
        "max_parcels_for_heatmap": 50000,  # Reduced for better performance
        "heatmap_neighbors": 5,
        "heatmap_smoothing": 3,
        # Font settings for all visualizations
        "font_family": "serif",
        "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font_stretch": "condensed",
        "font_size": 10,
        
        # Output directories
        "exports_output_dir": "Exports",  # Creates Exports/Data/{city}/ and Exports/Visualizations/{city}/
        
        # Census analysis settings (optional - only used if run_census_analysis=True in city config)
        "census_api_key": "YOUR_API_KEY_HERE",  # Get free key at: https://api.census.gov/data/key_signup.html
        "census_acs_year": 2022,
        "census_n_jobs": 4,
    }

    cities_config = {
        'Boston': {
            'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
            'parks_path': 'Data/Raw/Boston/boston_parks.geojson',
            'streets_path': 'Data/Processed/LTS/lts_bos.geojson',
            'boundary_path': 'Data/Raw/Boston/boston_neighborhood_boundaries.geojson.json',
            'lts_column': 'PC2_norm',
            'boundary_county_column': 'name',
            'combine_boundaries': True,
            
            # Census analysis configuration (optional)
            'run_census_analysis': False,  # Set to True to enable census analysis
            'census_state_fips': '25',
            'census_county_fips': ['025'],  # Suffolk County
            'census_block_shapefile': 'Data/Raw/Boston/Massachusets Census Blocks.shp/tl_2025_25_tabblock20.shp',
        },
        'Houston': {
            'parcels_path': 'Data/Raw/Houston/houstonparcelsclipped.geojson',
            'parks_path': 'Data/Raw/Houston/COH_PARKS_(City_of_Houston).geojson',
            'streets_path': 'Data/Processed/LTS/lts_hou.geojson',
            'boundary_path': 'Data/Raw/Houston/houstoncitylimits.geojson',
            'lts_column': 'PC2_norm',
            'combine_boundaries': False,
            
            # Census analysis configuration (optional)
            'run_census_analysis': False,
            'census_state_fips': '48',
            'census_county_fips': ['201'],  # Harris County
            'census_block_shapefile': 'Data/Raw/Houston/Texas Census Blocks.shp/tl_2025_48_tabblock20.shp',
        },
        "NYC": {
            "parcels_path": "Data/Raw/NYC/NYC_clipped_parcels.geojson",
            "parks_path": "Data/Raw/NYC/Parks_Properties_20251120.geojson",
            "streets_path": "Data/Processed/LTS/lts_nyc.geojson",
            "boundary_path": "Data/Raw/NYC/NY_County_FeaturesToJSON.geojson",
            "lts_column": "PC2_norm",
            "combine_boundaries": False,
            
            # Census analysis configuration (optional)
            'run_census_analysis': False,
            'census_state_fips': '36',
            'census_county_fips': ['005', '047', '061', '081', '085'],  # 5 boroughs
            'census_block_shapefile': 'Data/Raw/NYC/New York Census Blocks.shp/tl_2025_36_tabblock20.shp',
        },
        "LA": {
            "parcels_path": "Data/Raw/LA/LA_clipped_parcels.geojson",
            "parks_path": "Data/Raw/LA/la_parks.geojson",
            "streets_path": "Data/Processed/LTS/lts_la.geojson",
            "boundary_path": "Data/Raw/LA/City_Boundary.geojson",
            "lts_column": "PC2_norm",
            "combine_boundaries": False,
            
            # Census analysis configuration (optional)
            'run_census_analysis': False,
            'census_state_fips': '06',
            'census_county_fips': ['037'],  # Los Angeles County
            'census_block_shapefile': 'Data/Raw/LA/California Census Blocks.shp/tl_2025_06_tabblock20.shp',
        },
        "SF": {
            "parcels_path": "Data/Raw/SF/Parcels_–_Active_and_Retired_20251208.geojson",
            "parks_path": "Data/Raw/SF/Recreation_and_Parks_Properties_20251208.geojson",
            "streets_path": "Data/Processed/LTS/lts_sf.geojson",
            "boundary_path": "Data/Raw/SF/SF_Boundary.geojson",
            "lts_column": "PC1",
            "combine_boundaries": False,
        },
    }

    parkximity_results, census_results = multi_city_analysis(cities_config, global_settings=global_settings)

    for city_name, parcels_gdf in parkximity_results.items():
        print(f"\n{city_name} Results:")
        print(f"  Total parcels: {len(parcels_gdf)}")
        valid = parcels_gdf[parcels_gdf["park_distance"] < float("inf")]
        if len(valid) > 0:
            print(f"  Mean parkximity: {valid['park_distance'].mean():.2f} m")
            print(f"  Median parkximity: {valid['park_distance'].median():.2f} m")
