import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, LineString, MultiPolygon
from shapely import wkt
import heapq
from pathlib import Path

# Standard projected CRS for all US analysis (NAD83(2011) / Conus Albers - meters)
TARGET_CRS = "EPSG:6350"


class ParkProximityAnalyzer:
    """
    Analyzes park accessibility for a city using street network analysis
    and creates heatmap visualizations.
    
    All input data is automatically reprojected to EPSG:6350 (NAD83(2011) / Conus Albers)
    which uses meters as the unit of measurement and works for the entire contiguous US.
    """
    
    def __init__(self, city_name, config):
        """
        Initialize analyzer with city-specific configuration.
        
        Parameters:
        -----------
        city_name : str
            Name of the city being analyzed
        config : dict
            Configuration dictionary containing:
                - parcels_path: Path to parcels GeoJSON/CSV
                - parks_path: Path to parks GeoJSON
                - streets_path: Path to streets GeoJSON
                - boundary_path: Path to boundary GeoJSON/CSV (optional)
                - output_dir: Directory for output files
                - combine_boundaries: Boolean, if True combines neighborhood polygons into city boundary
                - lts_column: Column name for LTS values in streets data
                - park_buffer: Buffer distance in meters for park entrance detection (default 50)
                - entrance_tolerance: Tolerance in meters for deduplicating entrances (default 5)
                - heatmap_resolution: Resolution for heatmap grid (default 100)
                - heatmap_neighbors: Number of neighbors for interpolation (default 5)
                - heatmap_smoothing: Gaussian smoothing sigma (default 1)
        """
        self.city_name = city_name
        self.config = config
        self.target_crs = TARGET_CRS
        self.parcels_gdf = None
        self.parks_gdf = None
        self.streets_gdf = None
        self.entrances_gdf = None
        self.boundary_gdf = None
        self.network_graph = None
        self.distances_dict = None
        self.paths_dict = None
        
    def _load_and_reproject(self, filepath, description="data"):
        """
        Load a GeoJSON/shapefile and reproject to target CRS.
        
        Parameters:
        -----------
        filepath : str
            Path to the geospatial file
        description : str
            Description for logging purposes
            
        Returns:
        --------
        GeoDataFrame : Loaded and reprojected data
        """
        gdf = gpd.read_file(filepath)
        source_crs = gdf.crs
        
        if source_crs is None:
            print(f"  Warning: {description} has no CRS defined, assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")
            source_crs = gdf.crs
        
        print(f"  {description}: loaded from {source_crs}, reprojecting to {self.target_crs}")
        
        # Check if source is geographic (will cause issues with meter-based operations)
        if source_crs.is_geographic:
            print(f"    Note: Source CRS is geographic (lat/lon). Reprojection required for accurate distance calculations.")
        
        return gdf.to_crs(self.target_crs)
        
    def load_data(self):
        """Load all required data files and transform to target CRS (EPSG:6350)."""
        print(f"Loading data for {self.city_name}...")
        print(f"Target CRS: {self.target_crs} (NAD83(2011) / Conus Albers - meters)")
        
        # Load boundary first (if available) so we can clip parcels
        if 'boundary_path' in self.config and self.config['boundary_path']:
            self._load_boundary()
        
        # Load and process parcels (will be clipped if boundary is available)
        self._load_parcels()
        
        # Load parks and filter to polygons only
        parks_raw = self._load_and_reproject(self.config['parks_path'], "Parks")
        print(f"Loaded {len(parks_raw)} parks from file")
        
        # Filter to only polygon geometries
        parks_polygons = parks_raw[parks_raw.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        
        removed_count = len(parks_raw) - len(parks_polygons)
        if removed_count > 0:
            print(f"Filtered to {len(parks_polygons)} polygon parks (removed {removed_count} point/line parks)")
        else:
            print(f"All {len(parks_polygons)} parks are polygons")
        
        if len(parks_polygons) == 0:
            print("ERROR: No polygon parks found in dataset!")
            raise ValueError("No polygon parks available for analysis")
        
        # Validate and repair park geometries BEFORE clipping
        print("Validating and repairing park geometries...")
        valid_parks = []
        invalid_count = 0
        repaired_count = 0
        
        for idx, row in parks_polygons.iterrows():
            geom = row.geometry
            
            # Skip empty geometries
            if geom is None or geom.is_empty:
                invalid_count += 1
                continue
            
            # Check if geometry is valid
            if not geom.is_valid:
                # Try to repair using buffer(0) technique
                try:
                    repaired_geom = geom.buffer(0)
                    if repaired_geom.is_valid and not repaired_geom.is_empty:
                        row = row.copy()
                        row.geometry = repaired_geom
                        valid_parks.append(row)
                        repaired_count += 1
                    else:
                        invalid_count += 1
                        if invalid_count <= 3:
                            park_name = row.get('name', row.get('NAME', f'Park_{idx}'))
                            print(f"  Warning: Could not repair park '{park_name}' - skipping")
                except Exception as e:
                    invalid_count += 1
                    if invalid_count <= 3:
                        park_name = row.get('name', row.get('NAME', f'Park_{idx}'))
                        print(f"  Warning: Error repairing park '{park_name}': {str(e)[:50]}")
            else:
                valid_parks.append(row)
        
        if len(valid_parks) == 0:
            print("ERROR: No valid parks after geometry repair!")
            raise ValueError("No valid polygon parks available for analysis")
        
        parks_validated = gpd.GeoDataFrame(valid_parks, crs=self.target_crs)
        
        if repaired_count > 0:
            print(f"Repaired {repaired_count} invalid park geometries")
        if invalid_count > 0:
            print(f"Skipped {invalid_count} parks with unrepairable/empty geometries")
            if invalid_count > 3:
                print(f"  (Only first 3 warnings shown)")
        
        print(f"Valid parks after repair: {len(parks_validated)}")
        
        # NOW clip parks to boundary (after validation)
        if self.boundary_gdf is not None:
            print(f"Clipping parks to {self.city_name} boundary...")
            original_park_count = len(parks_validated)
            
            # Clip parks to boundary - keep only parts within boundary
            try:
                self.parks_gdf = gpd.clip(parks_validated, self.boundary_gdf)
                
                removed_park_count = original_park_count - len(self.parks_gdf)
                print(f"Clipped parks: {len(self.parks_gdf)} parks kept, {removed_park_count} parks removed (outside boundary)")
                
                if len(self.parks_gdf) == 0:
                    print("WARNING: No parks remain after clipping! Check that:")
                    print("  - Boundary polygon correctly represents the city area")
                    print("  - Park coordinates are correct")
            except Exception as e:
                print(f"Warning: Error during park clipping: {str(e)[:100]}")
                print("Continuing with unclipped parks...")
                self.parks_gdf = parks_validated
        else:
            self.parks_gdf = parks_validated
        
        print(f"Final park count: {len(self.parks_gdf)} valid polygon parks")
        
        # Load streets
        self.streets_gdf = self._load_and_reproject(self.config['streets_path'], "Streets")
        print(f"Loaded {len(self.streets_gdf)} street segments")
        
        # Verify CRS consistency
        self._verify_crs_consistency()
        
        # Create diagnostic map before generating entrances
        self._create_diagnostic_map()
        
        # Load park entrances (generate from street-park intersections)
        self._generate_entrances()
    
    def _verify_crs_consistency(self):
        """Verify all loaded data is in the target CRS."""
        datasets = [
            ('Parks', self.parks_gdf),
            ('Streets', self.streets_gdf),
            ('Parcels', self.parcels_gdf),
            ('Boundary', self.boundary_gdf)
        ]
        
        all_consistent = True
        for name, gdf in datasets:
            if gdf is not None and len(gdf) > 0:
                if gdf.crs != self.target_crs:
                    print(f"WARNING: {name} CRS mismatch! Expected {self.target_crs}, got {gdf.crs}")
                    all_consistent = False
        
        if all_consistent:
            print(f"CRS verification passed: all datasets in {self.target_crs}")
    
    def _create_diagnostic_map(self):
        """Create a map showing streets and parks before entrance generation."""
        print("Creating diagnostic map of streets and parks...")
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{self.city_name.lower().replace(' ', '_')}_diagnostic_streets_parks.png"
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot streets colored by LTS value
        lts_column = self.config.get('lts_column', 'final_lts')
        
        if lts_column in self.streets_gdf.columns:
            # Plot streets with color based on LTS
            self.streets_gdf.plot(
                ax=ax,
                column=lts_column,
                cmap='RdYlGn_r',  # Red (high stress) to Green (low stress)
                linewidth=0.8,
                alpha=0.7,
                legend=True,
                legend_kwds={'label': f'LTS Value ({lts_column})', 'orientation': 'horizontal'},
                zorder=1
            )
        else:
            # Plot streets without LTS coloring
            self.streets_gdf.plot(ax=ax, color='gray', linewidth=0.8, alpha=0.7, zorder=1)
            print(f"Warning: LTS column '{lts_column}' not found in streets data")
        
        # Plot parks
        self.parks_gdf.plot(
            ax=ax,
            color='green',
            edgecolor='darkgreen',
            alpha=0.5,
            linewidth=1.5,
            zorder=2
        )
        
        # Plot boundary if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax,
                facecolor='none',
                edgecolor='black',
                linewidth=2.5,
                zorder=3
            )
        
        # Add title and labels
        ax.set_title(
            f'Diagnostic Map: Streets and Parks - {self.city_name}',
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='darkgreen', alpha=0.5, 
                  label=f'Parks ({len(self.parks_gdf)})'),
            Line2D([0], [0], color='gray', linewidth=2, 
                   label=f'Streets ({len(self.streets_gdf)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        # Add info box with data details
        info_text = (
            f"Data Summary:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Parks: {len(self.parks_gdf)}\n"
            f"Streets: {len(self.streets_gdf)}\n"
            f"CRS: {self.target_crs}\n"
            f"\n"
            f"Parks bounds:\n"
            f"  X: [{self.parks_gdf.total_bounds[0]:.2f}, {self.parks_gdf.total_bounds[2]:.2f}]\n"
            f"  Y: [{self.parks_gdf.total_bounds[1]:.2f}, {self.parks_gdf.total_bounds[3]:.2f}]\n"
            f"\n"
            f"Streets bounds:\n"
            f"  X: [{self.streets_gdf.total_bounds[0]:.2f}, {self.streets_gdf.total_bounds[2]:.2f}]\n"
            f"  Y: [{self.streets_gdf.total_bounds[1]:.2f}, {self.streets_gdf.total_bounds[3]:.2f}]"
        )
        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
            verticalalignment='top',
            fontsize=10,
            family='monospace'
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved diagnostic map to {filename}")
        print(f"  → Check this map to verify streets and parks overlap spatially")
    
    def _load_parcels(self):
        """Load and process parcel data with automatic CRS detection and optional clipping."""
        parcels_path = self.config['parcels_path']
        
        # Check if parcels file is GeoJSON or CSV
        if parcels_path.endswith('.geojson') or parcels_path.endswith('.json'):
            # Load as GeoDataFrame directly
            print("Loading parcels from GeoJSON...")
            
            try:
                # Read with 'warn' to convert invalid geometries to None instead of raising errors
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    parcels_gdf_raw = gpd.read_file(parcels_path, on_invalid='warn')
                    
                    # Report any warnings that were raised
                    if w:
                        print(f"  Geometry warnings during load: {len(w)} geometries had issues")
                        # Show first few warnings
                        for i, warning in enumerate(w[:3]):
                            print(f"    Warning {i+1}: {warning.message}")
                        if len(w) > 3:
                            print(f"    ... and {len(w) - 3} more warnings")
                
                # Log and reproject
                source_crs = parcels_gdf_raw.crs
                if source_crs is None:
                    print(f"  Warning: Parcels have no CRS defined, assuming EPSG:4326")
                    parcels_gdf_raw = parcels_gdf_raw.set_crs("EPSG:4326")
                    source_crs = parcels_gdf_raw.crs
                
                print(f"  Parcels: loaded from {source_crs}, reprojecting to {self.target_crs}")
                print(f"  Successfully loaded {len(parcels_gdf_raw)} parcels (some may have None geometry)")
                
            except Exception as e:
                print(f"  Error loading parcels: {e}")
                print(f"  Creating empty parcel dataset - analysis will continue without parcels")
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
                return
            
            # Filter out parcels with invalid or None geometries
            invalid_count = 0
            valid_indices = []
            
            print(f"  Validating parcels...")
            
            for idx, row in parcels_gdf_raw.iterrows():
                geom = row.geometry
                
                # Check if geometry is None (invalid geometries are set to None with on_invalid='warn')
                if geom is None or geom.is_empty:
                    invalid_count += 1
                    continue
                
                # Additional validity checks
                try:
                    if not geom.is_valid:
                        # Try to get more details about the invalid geometry
                        if geom.geom_type == 'Polygon':
                            # Check for rings with < 4 points (invalid linear rings)
                            if len(geom.exterior.coords) < 4:
                                if invalid_count < 5:  # Only print first 5 warnings
                                    print(f"  Warning: Parcel {idx} has invalid exterior ring with only {len(geom.exterior.coords)} points (minimum 4 required)")
                                invalid_count += 1
                                continue
                            
                            # Check interior rings
                            has_invalid_interior = False
                            for interior in geom.interiors:
                                if len(interior.coords) < 4:
                                    if invalid_count < 5:
                                        print(f"  Warning: Parcel {idx} has invalid interior ring with only {len(interior.coords)} points")
                                    has_invalid_interior = True
                                    break
                            
                            if has_invalid_interior:
                                invalid_count += 1
                                continue
                        
                        # Generic invalid geometry warning
                        if invalid_count < 5:
                            print(f"  Warning: Parcel {idx} has invalid geometry (type: {geom.geom_type})")
                        invalid_count += 1
                        continue
                except Exception:
                    # If we can't check validity, skip it
                    invalid_count += 1
                    continue
                
                valid_indices.append(idx)
            
            # Keep only valid parcels and reproject
            if len(valid_indices) > 0:
                self.parcels_gdf = parcels_gdf_raw.loc[valid_indices].copy().to_crs(self.target_crs)
            else:
                print(f"  Warning: No valid parcels found!")
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
            
            if invalid_count > 0:
                print(f"  Skipped {invalid_count} parcels with invalid/None geometries")
                if invalid_count > 5:
                    print(f"  (Only first 5 detailed warnings shown)")
            
            print(f"Loaded {len(self.parcels_gdf)} valid parcels from GeoJSON")
            
        else:
            # Load from CSV with coordinate transformation
            print("Loading parcels from CSV...")
            parcels_df = pd.read_csv(parcels_path, engine='python', on_bad_lines='warn')
            
            # Assume CSV coordinates are in WGS84 (EPSG:4326)
            print(f"  Assuming CSV coordinates are in EPSG:4326 (WGS84 lat/lon)")
            
            # Transform coordinates
            geometries = []
            valid_indices = []
            
            for idx, row in parcels_df.iterrows():
                try:
                    lat = row['centroid_latitude']
                    lon = row['centroid_longitude']
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    # Create point in WGS84
                    geometries.append(Point(lon, lat))
                    valid_indices.append(idx)
                except Exception:
                    continue
            
            valid_parcels_df = parcels_df.iloc[valid_indices].copy()
            
            # Create GeoDataFrame in WGS84 then reproject
            self.parcels_gdf = gpd.GeoDataFrame(
                valid_parcels_df,
                geometry=geometries,
                crs="EPSG:4326"
            ).to_crs(self.target_crs)
            
            print(f"Loaded {len(self.parcels_gdf)} valid parcels from CSV")
        
        # Clip parcels to boundary if boundary is available
        if self.boundary_gdf is not None and len(self.parcels_gdf) > 0:
            self._clip_parcels_to_boundary()
    
    def _clip_parcels_to_boundary(self):
        """Clip parcels to the city boundary polygon."""
        print(f"Clipping parcels to {self.city_name} boundary...")
        
        original_count = len(self.parcels_gdf)
        
        # Use spatial join or clip to keep only parcels within the boundary
        # Using sjoin is more efficient for point data
        clipped_parcels = gpd.sjoin(
            self.parcels_gdf,
            self.boundary_gdf,
            how='inner',
            predicate='within'
        )
        
        # Remove the index column added by sjoin
        if 'index_right' in clipped_parcels.columns:
            clipped_parcels = clipped_parcels.drop(columns=['index_right'])
        
        self.parcels_gdf = clipped_parcels
        
        removed_count = original_count - len(self.parcels_gdf)
        print(f"Clipped parcels: {len(self.parcels_gdf)} parcels kept, {removed_count} parcels removed (outside boundary)")
        
        if len(self.parcels_gdf) == 0:
            print("WARNING: No parcels remain after clipping! Check that:")
            print("  - Boundary polygon correctly represents the city area")
            print("  - Parcel coordinates are correct")
    
    def _generate_entrances(self):
        """
        Assign park entrances as points where the street network intersects 
        with park polygons. Only handles polygon parks.
        """
        print("Generating park entrances from street-park intersections...")
        
        # Verify we have parks and streets to work with
        if len(self.parks_gdf) == 0:
            print("ERROR: No parks available for entrance generation!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
            return
        
        if len(self.streets_gdf) == 0:
            print("ERROR: No streets available for entrance generation!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.target_crs)
            return
        
        # Verify CRS is projected (not geographic) for accurate buffer distances
        if self.parks_gdf.crs.is_geographic:
            print("ERROR: Parks are in a geographic CRS (lat/lon degrees)!")
            print("  Buffer distances would be interpreted as degrees, not meters.")
            print("  This should not happen - please report this bug.")
            raise ValueError("Cannot use geographic CRS for park entrance generation")
        
        # Get buffer distance from config (default 50 meters)
        buffer_distance = self.config.get('park_buffer', 50.0)
        print(f"Using buffer distance: {buffer_distance} meters")
        print(f"Processing {len(self.parks_gdf)} polygon parks")
        
        # Build spatial index for streets (speeds up intersection queries significantly)
        print("  Building spatial index for streets...")
        streets_sindex = self.streets_gdf.sindex
        
        entrance_points = []
        entrance_data = []
        parks_with_entrances = 0
        parks_without_entrances = []
        
        for park_idx, park in self.parks_gdf.iterrows():
            park_geom = park.geometry
            # Better park name handling
            park_name = park.get('name', park.get('NAME', park.get('park_name', park.get('PARK_NAME', f'Unnamed_Park_{park_idx}'))))
            
            # Buffer park to find intersecting streets
            buffered_park = park_geom.buffer(buffer_distance)
            
            # Use spatial index to find candidate streets (much faster than checking all)
            candidate_idx = list(streets_sindex.intersection(buffered_park.bounds))
            if len(candidate_idx) == 0:
                parks_without_entrances.append(park_name)
                continue
            
            # Filter candidates to those that actually intersect
            candidate_streets = self.streets_gdf.iloc[candidate_idx]
            intersecting_streets = candidate_streets[candidate_streets.intersects(buffered_park)]
            
            if len(intersecting_streets) == 0:
                parks_without_entrances.append(park_name)
                continue
            
            parks_with_entrances += 1
            park_entrance_count = 0
            
            for street_idx, street in intersecting_streets.iterrows():
                # Get the intersection with the ORIGINAL park boundary (not buffered)
                intersection = street.geometry.intersection(park_geom)
                
                # If no direct intersection, use the closest point on park boundary
                if intersection.is_empty:
                    try:
                        nearest_point = park_geom.boundary.interpolate(
                            park_geom.boundary.project(street.geometry.centroid)
                        )
                        if nearest_point is not None and not nearest_point.is_empty:
                            entrance_points.append(nearest_point)
                            entrance_data.append({
                                'park_name': park_name,
                                'park_id': park_idx,
                                'street_id': street_idx,
                                'method': 'nearest_point'
                            })
                            park_entrance_count += 1
                    except Exception as e:
                        continue
                    continue
                
                # Handle different geometry types
                if intersection.geom_type == 'Point':
                    entrance_points.append(intersection)
                    entrance_data.append({
                        'park_name': park_name,
                        'park_id': park_idx,
                        'street_id': street_idx,
                        'method': 'direct_intersection'
                    })
                    park_entrance_count += 1
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        entrance_points.append(point)
                        entrance_data.append({
                            'park_name': park_name,
                            'park_id': park_idx,
                            'street_id': street_idx,
                            'method': 'direct_intersection'
                        })
                        park_entrance_count += 1
                elif intersection.geom_type == 'LineString':
                    coords = list(intersection.coords)
                    if len(coords) >= 2:
                        entrance_points.append(Point(coords[0]))
                        entrance_points.append(Point(coords[-1]))
                        entrance_data.extend([
                            {
                                'park_name': park_name,
                                'park_id': park_idx,
                                'street_id': street_idx,
                                'method': 'line_endpoints'
                            },
                            {
                                'park_name': park_name,
                                'park_id': park_idx,
                                'street_id': street_idx,
                                'method': 'line_endpoints'
                            }
                        ])
                        park_entrance_count += 2
                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            entrance_points.append(Point(coords[0]))
                            entrance_points.append(Point(coords[-1]))
                            entrance_data.extend([
                                {
                                    'park_name': park_name,
                                    'park_id': park_idx,
                                    'street_id': street_idx,
                                    'method': 'line_endpoints'
                                },
                                {
                                    'park_name': park_name,
                                    'park_id': park_idx,
                                    'street_id': street_idx,
                                    'method': 'line_endpoints'
                                }
                            ])
                            park_entrance_count += 2
                elif intersection.geom_type in ['Polygon', 'MultiPolygon', 'GeometryCollection']:
                    boundary = intersection.boundary
                    if boundary.geom_type == 'LineString':
                        coords = list(boundary.coords)
                        if len(coords) > 0:
                            entrance_points.append(Point(coords[0]))
                            entrance_data.append({
                                'park_name': park_name,
                                'park_id': park_idx,
                                'street_id': street_idx,
                                'method': 'polygon_boundary'
                            })
                            park_entrance_count += 1
                    elif boundary.geom_type == 'MultiLineString':
                        for line in boundary.geoms:
                            coords = list(line.coords)
                            if len(coords) > 0:
                                entrance_points.append(Point(coords[0]))
                                entrance_data.append({
                                    'park_name': park_name,
                                    'park_id': park_idx,
                                    'street_id': street_idx,
                                    'method': 'polygon_boundary'
                                })
                                park_entrance_count += 1
            
            if park_entrance_count > 0 and parks_with_entrances <= 5:
                print(f"  Park '{park_name}': {park_entrance_count} entrances")
        
        print(f"Parks with entrances: {parks_with_entrances}/{len(self.parks_gdf)}")
        
        if len(parks_without_entrances) > 0:
            print(f"Parks without entrances: {len(parks_without_entrances)}")
            print(f"  Possible reasons: too small, no streets nearby, or disconnected from network")
            if len(parks_without_entrances) <= 5:
                print(f"  Examples: {', '.join(str(p) for p in parks_without_entrances[:5])}")
        
        # Store parks without entrances for diagnostic mapping
        self.parks_without_entrances = parks_without_entrances
        
        if len(entrance_points) == 0:
            print("ERROR: No street-park intersections found!")
            print("This could mean:")
            print("  - Park polygons don't overlap with street network")
            print("  - Parks are too small or streets don't reach them")
            print("  - Data quality issues with park or street geometries")
            print("\nDebugging info:")
            print(f"  Number of parks: {len(self.parks_gdf)}")
            print(f"  Number of streets: {len(self.streets_gdf)}")
            print(f"  CRS: {self.target_crs}")
            print(f"  Parks bounds: {self.parks_gdf.total_bounds}")
            print(f"  Streets bounds: {self.streets_gdf.total_bounds}")
            
            # Sample a few parks and check their geometries
            print("\nSample park analysis:")
            for i, (idx, park) in enumerate(self.parks_gdf.head(3).iterrows()):
                park_name = park.get('name', park.get('NAME', f'Park_{idx}'))
                print(f"  Park: {park_name}")
                print(f"    Type: {park.geometry.geom_type}")
                print(f"    Area: {park.geometry.area:.2f} sq meters")
                print(f"    Bounds: {park.geometry.bounds}")
                
                # Check if any streets are nearby
                buffered = park.geometry.buffer(buffer_distance)
                nearby = self.streets_gdf[self.streets_gdf.intersects(buffered)]
                print(f"    Streets within {buffer_distance}m: {len(nearby)}")
            
            # Create empty GeoDataFrame
            self.entrances_gdf = gpd.GeoDataFrame(
                geometry=[],
                crs=self.target_crs
            )
            return
        
        # Remove duplicate points (within a small tolerance) using KD-tree for efficiency
        tolerance = self.config.get('entrance_tolerance', 5.0)  # 5 meter default
        
        print(f"Removing duplicate entrances within {tolerance}m (processing {len(entrance_points)} points)...")
        
        # Filter out None/empty points first
        valid_points = []
        valid_data = []
        for i, point in enumerate(entrance_points):
            if point is not None and not point.is_empty:
                valid_points.append(point)
                valid_data.append(entrance_data[i])
        
        if len(valid_points) == 0:
            unique_points = []
            unique_data = []
        else:
            # Convert to numpy array for KD-tree
            coords = np.array([[p.x, p.y] for p in valid_points])
            
            # Build KD-tree
            tree = cKDTree(coords)
            
            # Find all pairs within tolerance
            # query_ball_tree returns indices of points within distance
            pairs = tree.query_ball_tree(tree, r=tolerance)
            
            # Keep only the first point in each cluster
            kept_indices = set()
            removed_indices = set()
            
            for i, neighbors in enumerate(pairs):
                if i in removed_indices:
                    continue
                # Keep this point
                kept_indices.add(i)
                # Mark all its neighbors (except itself) as duplicates
                for j in neighbors:
                    if j != i:
                        removed_indices.add(j)
            
            # Extract unique points
            unique_points = [valid_points[i] for i in sorted(kept_indices)]
            unique_data = [valid_data[i] for i in sorted(kept_indices)]
        
        self.entrances_gdf = gpd.GeoDataFrame(
            unique_data,
            geometry=unique_points,
            crs=self.target_crs
        )
        
        print(f"Generated {len(self.entrances_gdf)} unique park entrance points")
        print(f"  (Removed {len(entrance_points) - len(unique_points)} duplicate points within {tolerance}m)")
    
    def _load_boundary(self):
        """Load city/county boundary from GeoJSON or CSV file."""
        boundary_path = self.config['boundary_path']
        
        # Determine file type
        if boundary_path.endswith('.geojson') or boundary_path.endswith('.json'):
            self._load_boundary_from_geojson()
        elif boundary_path.endswith('.csv'):
            self._load_boundary_from_csv()
        else:
            print(f"Warning: Unsupported boundary file format for {boundary_path}")
            print("Supported formats: .geojson, .json, .csv")
    
    def _load_boundary_from_geojson(self):
        """Load city/county boundary from GeoJSON file."""
        print(f"Loading boundary from GeoJSON for {self.city_name}...")
        
        # Load the GeoJSON file
        boundary_gdf = gpd.read_file(self.config['boundary_path'])
        
        # Log source CRS
        source_crs = boundary_gdf.crs
        if source_crs is None:
            print(f"  Warning: Boundary has no CRS defined, assuming EPSG:4326")
            boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
            source_crs = boundary_gdf.crs
        print(f"  Boundary: loaded from {source_crs}, will reproject to {self.target_crs}")
        
        # Check if we need to filter by city/county name
        if 'boundary_name' in self.config and self.config['boundary_name']:
            county_column = self.config.get('boundary_county_column', 'name')
            
            # Filter to the specified city/county
            boundary_gdf = boundary_gdf[boundary_gdf[county_column] == self.config['boundary_name']]
            
            if len(boundary_gdf) == 0:
                print(f"Warning: Could not find boundary for {self.config['boundary_name']}")
                return
        
        # Check if we need to combine neighborhood boundaries
        combine_boundaries = self.config.get('combine_boundaries', False)
        
        if combine_boundaries:
            print(f"Combining {len(boundary_gdf)} neighborhood boundaries...")
            
            # Extract all geometries
            neighborhood_geoms = []
            for idx, row in boundary_gdf.iterrows():
                geom = row.geometry
                
                # Extract individual polygons from MultiPolygons
                if isinstance(geom, MultiPolygon):
                    neighborhood_geoms.extend(list(geom.geoms))
                else:
                    neighborhood_geoms.append(geom)
            
            # Combine all neighborhood polygons into one
            combined_boundary = self._combine_neighborhood_polygons(neighborhood_geoms)
            
            if combined_boundary is None:
                print(f"Warning: Failed to combine neighborhoods for {self.city_name}")
                return
            
            print(f"Successfully combined neighborhoods into city boundary")
            
        else:
            # Use the boundary as-is (single or first feature)
            if len(boundary_gdf) == 0:
                print(f"Warning: No boundary features found in file")
                return
            
            # If multiple features, combine them or use the first/largest
            if len(boundary_gdf) > 1:
                print(f"Warning: Found {len(boundary_gdf)} features. Combining all into one boundary.")
                neighborhood_geoms = []
                for idx, row in boundary_gdf.iterrows():
                    geom = row.geometry
                    if isinstance(geom, MultiPolygon):
                        neighborhood_geoms.extend(list(geom.geoms))
                    else:
                        neighborhood_geoms.append(geom)
                combined_boundary = self._combine_neighborhood_polygons(neighborhood_geoms)
            else:
                combined_boundary = boundary_gdf.iloc[0].geometry
                
                # Extract largest polygon from MultiPolygon
                if isinstance(combined_boundary, MultiPolygon):
                    combined_boundary = max(combined_boundary.geoms, key=lambda a: a.area)
        
        # Create GeoDataFrame with the boundary and reproject
        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[combined_boundary],
            crs=boundary_gdf.crs
        ).to_crs(self.target_crs)
        print(f"Loaded boundary for {self.city_name}")
    
    def _load_boundary_from_csv(self):
        """Load city/county boundary from CSV file with WKT geometry."""
        print(f"Loading boundary from CSV for {self.city_name}...")
        
        county_df = pd.read_csv(self.config['boundary_path'])
        wkt_column = self.config.get('boundary_wkt_column', 'the_geom')
        county_column = self.config.get('boundary_county_column', 'COUNTY')
        
        # Assume CSV WKT is in WGS84 unless specified
        csv_crs = self.config.get('boundary_csv_crs', 'EPSG:4326')
        print(f"  Assuming CSV coordinates are in {csv_crs}")
        
        # Check if we need to combine neighborhood boundaries
        combine_boundaries = self.config.get('combine_boundaries', False)
        
        if combine_boundaries:
            # Load multiple neighborhood polygons and combine them
            print(f"Combining neighborhood boundaries for {self.city_name}...")
            
            # Filter to get all rows for this city
            city_rows = county_df[county_df[county_column] == self.config.get('boundary_name', self.city_name)]
            
            if len(city_rows) == 0:
                print(f"Warning: Could not find any neighborhoods for {self.city_name}")
                return
            
            # Extract all neighborhood geometries
            neighborhood_geoms = []
            for idx, row in city_rows.iterrows():
                boundary_wkt = row[wkt_column]
                geom = wkt.loads(boundary_wkt)
                
                # Extract individual polygons from MultiPolygons
                if isinstance(geom, MultiPolygon):
                    neighborhood_geoms.extend(list(geom.geoms))
                else:
                    neighborhood_geoms.append(geom)
            
            print(f"Found {len(neighborhood_geoms)} neighborhood polygons")
            
            # Combine all neighborhood polygons into one
            combined_boundary = self._combine_neighborhood_polygons(neighborhood_geoms)
            
            if combined_boundary is None:
                print(f"Warning: Failed to combine neighborhoods for {self.city_name}")
                return
            
            print(f"Successfully combined neighborhoods into city boundary")
            
        else:
            # Load single city/county boundary
            county_row = county_df[county_df[county_column] == self.config.get('boundary_name', self.city_name)]
            
            if len(county_row) == 0:
                print(f"Warning: Could not find boundary for {self.city_name}")
                return
            
            boundary_wkt = county_row[wkt_column].iloc[0]
            combined_boundary = wkt.loads(boundary_wkt)
            
            # Extract largest polygon from MultiPolygon
            if isinstance(combined_boundary, MultiPolygon):
                combined_boundary = max(combined_boundary.geoms, key=lambda a: a.area)
        
        # Create GeoDataFrame and reproject
        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[combined_boundary],
            crs=csv_crs
        ).to_crs(self.target_crs)
        print(f"Loaded boundary for {self.city_name}")
    
    def _combine_neighborhood_polygons(self, polygons):
        """
        Combine multiple neighborhood polygons into a single city boundary polygon.
        
        Parameters:
        -----------
        polygons : list
            List of Polygon/MultiPolygon geometries representing neighborhoods
            
        Returns:
        --------
        Polygon : Combined boundary as a single Polygon (exterior only)
        """
        from shapely.ops import unary_union
        
        if not polygons:
            return None
        
        # Use unary_union to merge all polygons
        # This handles overlapping and adjacent polygons properly
        merged = unary_union(polygons)
        
        # If result is a MultiPolygon, extract the largest polygon
        # (This handles cases where neighborhoods aren't perfectly connected)
        if isinstance(merged, MultiPolygon):
            print(f"Warning: Combined boundary resulted in {len(merged.geoms)} disconnected areas")
            print(f"Using largest area as city boundary")
            merged = max(merged.geoms, key=lambda p: p.area)
        
        # Return only the exterior boundary (removes any interior holes)
        # This creates a clean perimeter for the city
        if hasattr(merged, 'exterior'):
            from shapely.geometry import Polygon
            city_boundary = Polygon(merged.exterior)
            return city_boundary
        else:
            return merged
    
    def build_network(self):
        """Build network graph from street segments."""
        print("Building street network...")
        self.network_graph = nx.Graph()
        all_edges = []
        
        for i, row in self.streets_gdf.iterrows():
            linestring = row.geometry
            weight = row[self.config.get('lts_column', 'final_lts')]
            coords = list(linestring.coords)
            
            for j in range(len(coords) - 1):
                start, end = coords[j], coords[j + 1]
                line_geom = LineString([start, end])
                length = line_geom.length
                
                # Skip invalid/zero-length segments
                if length <= 0 or np.isnan(weight):
                    continue
                
                # Weight by LTS value (higher LTS = less desirable = higher cost)
                weighted_length = length * (1 + weight)
                
                all_edges.append((start, end, {
                    'weight': weighted_length,
                    'length': length,
                    'lts': weight
                }))
        
        self.network_graph.add_edges_from(all_edges)
        print(f"Graph created with {self.network_graph.number_of_nodes()} nodes and {self.network_graph.number_of_edges()} edges")
    
    def calculate_distances(self):
        """Calculate network distances from all nodes to nearest park entrance."""
        print("Calculating network distances to parks...")
        
        if len(self.entrances_gdf) == 0:
            print("Warning: No park entrances available. Distances will be infinite.")
            self.distances_dict = {}
            return
        
        # Get entrance coordinates as source nodes using KD-tree for efficiency
        print(f"Finding nearest network nodes for {len(self.entrances_gdf)} park entrances...")
        
        # Build KD-tree of network nodes
        nodes_list = list(self.network_graph.nodes())
        nodes_array = np.array(nodes_list)
        tree = cKDTree(nodes_array)
        
        # Get entrance coordinates
        entrance_coords = np.array([[p.x, p.y] for p in self.entrances_gdf.geometry])
        
        # Find nearest network node for each entrance (vectorized, very fast)
        _, indices = tree.query(entrance_coords)
        
        # Get unique source nodes
        source_nodes = set(tuple(nodes_array[i]) for i in indices)
        
        print(f"Found {len(source_nodes)} unique source nodes from park entrances")
        
        # Run multi-source Dijkstra from all park entrances
        print("Running Dijkstra's algorithm (this may take a few minutes for large networks)...")
        
        # Initialize distances
        self.distances_dict = {node: float('inf') for node in self.network_graph.nodes()}
        
        # Priority queue: (distance, node)
        pq = []
        
        # Initialize all source nodes with distance 0
        for source in source_nodes:
            self.distances_dict[source] = 0
            heapq.heappush(pq, (0, source))
        
        visited = set()
        total_nodes = len(self.network_graph.nodes())
        processed = 0
        last_percent = 0
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in visited:
                continue
            
            visited.add(node)
            processed += 1
            
            # Progress update
            percent = (processed * 100) // total_nodes
            if percent >= last_percent + 10:
                print(f"  Progress: {percent}% ({processed:,}/{total_nodes:,} nodes)")
                last_percent = percent
            
            # Explore neighbors
            for neighbor in self.network_graph.neighbors(node):
                if neighbor in visited:
                    continue
                
                edge_data = self.network_graph.get_edge_data(node, neighbor)
                new_dist = dist + edge_data['weight']
                
                if new_dist < self.distances_dict[neighbor]:
                    self.distances_dict[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        reachable = sum(1 for d in self.distances_dict.values() if d < float('inf'))
        print(f"Calculated distances to {reachable} nodes ({reachable * 100 // total_nodes}%)")
    
    def calculate_parcel_distances(self):
        """Assign park distances to parcels based on nearest network node."""
        print("Calculating parcel distances to parks...")
        
        if len(self.parcels_gdf) == 0:
            print("Warning: No parcels to process.")
            return
        
        # Build KD-tree of network nodes for fast nearest-neighbor lookup
        nodes_list = list(self.network_graph.nodes())
        nodes_array = np.array(nodes_list)
        tree = cKDTree(nodes_array)
        
        # Extract all parcel centroids at once (vectorized)
        print(f"  Processing {len(self.parcels_gdf)} parcels...")
        parcel_coords = np.array([
            (geom.centroid.x, geom.centroid.y) if geom.geom_type in ['Polygon', 'MultiPolygon']
            else (geom.x, geom.y)
            for geom in self.parcels_gdf.geometry
        ])
        
        # Find nearest network node for ALL parcels at once (vectorized KD-tree query)
        _, nearest_indices = tree.query(parcel_coords)
        
        # Look up distances for all parcels (vectorized)
        parcel_distances = np.array([
            self.distances_dict.get(tuple(nodes_array[idx]), float('inf'))
            for idx in nearest_indices
        ])
        
        self.parcels_gdf['park_distance'] = parcel_distances
        
        valid_distances = parcel_distances[parcel_distances < float('inf')]
        if len(valid_distances) > 0:
            print(f"Valid distances: min={valid_distances.min():.2f}, max={valid_distances.max():.2f}, mean={valid_distances.mean():.2f}")
        print(f"Parcels with valid distances: {len(valid_distances)}/{len(parcel_distances)}")
    
    def export_parcels_geojson(self):
        """
        Export parcel data as GeoJSON with only geometry and parkximity calculations.
        
        Creates a simplified GeoJSON file containing:
        - geometry: Original parcel geometry
        - park_distance: LTS-weighted network distance to nearest park (meters)
        - park_distance_km: Same distance in kilometers
        
        Returns:
        --------
        str : Path to the exported GeoJSON file
        """
        print(f"Exporting parcel parkximity data for {self.city_name}...")
        
        if len(self.parcels_gdf) == 0:
            print("Warning: No parcels to export.")
            return None
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add distance in km if not already present
        if 'park_distance_km' not in self.parcels_gdf.columns and 'park_distance' in self.parcels_gdf.columns:
            self.parcels_gdf['park_distance_km'] = self.parcels_gdf['park_distance'] / 1000.0
        
        # Create a new GeoDataFrame with only geometry and parkximity columns
        export_columns = ['geometry']
        if 'park_distance' in self.parcels_gdf.columns:
            export_columns.append('park_distance')
        if 'park_distance_km' in self.parcels_gdf.columns:
            export_columns.append('park_distance_km')
        
        export_gdf = self.parcels_gdf[export_columns].copy()
        
        # Reproject to WGS84 (EPSG:4326) for GeoJSON compatibility
        export_gdf = export_gdf.to_crs("EPSG:4326")
        
        # Generate filename
        output_filename = output_dir / f'{self.city_name.lower().replace(" ", "_")}_parcels_parkximity.geojson'
        
        # Export to GeoJSON
        export_gdf.to_file(output_filename, driver='GeoJSON')
        
        print(f"Exported {len(export_gdf)} parcels to {output_filename}")
        
        # Print summary statistics
        if 'park_distance_km' in export_gdf.columns:
            valid_distances = export_gdf[export_gdf['park_distance_km'] < float('inf')]['park_distance_km']
            if len(valid_distances) > 0:
                print(f"  Distance statistics (km):")
                print(f"    Min: {valid_distances.min():.3f}")
                print(f"    Max: {valid_distances.max():.3f}")
                print(f"    Mean: {valid_distances.mean():.3f}")
                print(f"    Median: {valid_distances.median():.3f}")
                print(f"  Parcels with valid distances: {len(valid_distances)}/{len(export_gdf)}")
        
        return str(output_filename)
    
    def create_visualizations(self):
        """Create all visualization outputs."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = output_dir / f'{self.city_name.lower().replace(" ", "_")}_parkximity'
        
        # Create entrance map
        self._create_entrance_map(base_filename)
        
        # Create parks-without-entrances diagnostic map
        self._create_parks_without_entrances_map(base_filename)
        
        # Create heatmap
        self._create_heatmap(base_filename)
        
        # Create histogram
        self._create_histogram(base_filename)
    
    def _create_entrance_map(self, base_filename):
        """Create a map showing park entrances."""
        print("Creating park entrance map...")
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot parks
        self.parks_gdf.plot(
            ax=ax,
            color='lightgreen',
            edgecolor='darkgreen',
            alpha=0.5,
            linewidth=1
        )
        
        # Plot entrances
        if len(self.entrances_gdf) > 0:
            self.entrances_gdf.plot(
                ax=ax,
                color='red',
                markersize=10,
                alpha=0.7,
                zorder=5
            )
        
        # Plot boundary
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax,
                facecolor='none',
                edgecolor='black',
                linewidth=2
            )
        
        ax.set_title(f'Park Entrances - {self.city_name}', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        entrance_filename = f'{base_filename}_entrances.png'
        plt.savefig(entrance_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved entrance map to {entrance_filename}")
    
    def _create_parks_without_entrances_map(self, base_filename):
        """Create diagnostic map highlighting parks without entrances."""
        print("Creating diagnostic map for parks without entrances...")
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot all parks in green
        self.parks_gdf.plot(
            ax=ax,
            color='lightgreen',
            edgecolor='darkgreen',
            alpha=0.5,
            linewidth=1,
            zorder=1
        )
        
        # Highlight parks without entrances in red
        if hasattr(self, 'parks_without_entrances') and len(self.parks_without_entrances) > 0:
            # Find parks by name that don't have entrances
            parks_no_entrance = self.parks_gdf[
                self.parks_gdf.apply(
                    lambda row: row.get('name', row.get('NAME', row.get('park_name', row.get('PARK_NAME', f'Unnamed_Park_{row.name}')))) 
                    in self.parks_without_entrances,
                    axis=1
                )
            ]
            
            if len(parks_no_entrance) > 0:
                parks_no_entrance.plot(
                    ax=ax,
                    color='red',
                    edgecolor='darkred',
                    alpha=0.7,
                    linewidth=2,
                    zorder=2
                )
        
        # Plot boundary
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(
                ax=ax,
                facecolor='none',
                edgecolor='black',
                linewidth=2,
                zorder=3
            )
        
        ax.set_title(f'Parks Without Entrances (Red) - {self.city_name}', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        diagnostic_filename = f'{base_filename}_parks_without_entrances.png'
        plt.savefig(diagnostic_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        parks_without = len(self.parks_without_entrances) if hasattr(self, 'parks_without_entrances') else 0
        print(f"Saved parks-without-entrances diagnostic map to {diagnostic_filename}")
        print(f"  → {parks_without} parks highlighted in red")
    
    
    
    def _create_heatmap(self, base_filename):
        """Create smoothed heatmap visualization."""
        print("Creating heatmap visualization...")
        
        valid_parcels = self.parcels_gdf[self.parcels_gdf['park_distance'] < float('inf')].copy()
        
        if len(valid_parcels) == 0:
            print("Warning: No valid distances for heatmap.")
            return
        
        print(f"Processing {len(valid_parcels)} parcels for heatmap...")
        
        # Sample parcels if dataset is very large (for performance)
        max_parcels_for_heatmap = self.config.get('max_parcels_for_heatmap', 20000)
        if len(valid_parcels) > max_parcels_for_heatmap:
            print(f"  Large dataset detected - sampling {max_parcels_for_heatmap} parcels for heatmap...")
            valid_parcels = valid_parcels.sample(n=max_parcels_for_heatmap, random_state=42)
        
        valid_parcels['park_distance_km'] = valid_parcels['park_distance'] / 1000.0
        vmin_km = valid_parcels['park_distance_km'].min()
        vmax_km = np.percentile(valid_parcels['park_distance_km'], 95)
        
        # Create grid
        xmin, ymin, xmax, ymax = self.streets_gdf.total_bounds
        res = self.config.get('heatmap_resolution', 100)
        
        x_range = np.arange(xmin, xmax, res)
        y_range = np.arange(ymin, ymax, res)
        print(f"  Grid size: {len(x_range)} x {len(y_range)} = {len(x_range) * len(y_range):,} points")
        
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        
        # Interpolate values
        print("  Extracting parcel coordinates...")
        # Handle both Point and Polygon geometries
        valid_coords = []
        for geom in valid_parcels.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                centroid = geom.centroid
                valid_coords.append([centroid.x, centroid.y])
            else:  # Point
                valid_coords.append([geom.x, geom.y])
        
        valid_points = np.array(valid_coords)
        valid_values = valid_parcels['park_distance'].values
        
        print("  Building spatial index...")
        tree = cKDTree(valid_points)
        
        print("  Interpolating values to grid...")
        k = self.config.get('heatmap_neighbors', 5)
        distances, indices = tree.query(grid_points, k=k)
        
        print("  Computing weighted interpolation...")
        weights = 1.0 / (distances + 1e-10)
        weights_sum = np.sum(weights, axis=1)
        neighbor_values = valid_values[indices]
        grid_values = np.sum(weights * neighbor_values, axis=1) / weights_sum
        grid_values_km = grid_values / 1000.0
        grid_values_km = grid_values_km.reshape(len(y_range), len(x_range))
        
        # Smooth
        print("  Applying Gaussian smoothing...")
        sigma = self.config.get('heatmap_smoothing', 1)
        grid_smooth = gaussian_filter(grid_values_km, sigma=sigma)
        
        # Apply boundary mask - OPTIMIZED VECTORIZED VERSION
        print("  Applying boundary mask...")
        if self.boundary_gdf is not None:
            try:
                from shapely.vectorized import contains
                boundary_geom = self.boundary_gdf.iloc[0].geometry
                
                # Vectorized approach - MUCH faster than nested loop
                grid_x_flat = grid_x.ravel()
                grid_y_flat = grid_y.ravel()
                
                # Check if points are inside boundary using vectorized operation
                inside = contains(boundary_geom, grid_x_flat, grid_y_flat)
                inside_reshaped = inside.reshape(grid_smooth.shape)
                
                # Mask out points outside boundary
                grid_smooth[~inside_reshaped] = np.nan
                print(f"    Masked {(~inside_reshaped).sum():,} grid points outside boundary")
                
            except ImportError:
                # Fallback to prepared geometry approach if shapely.vectorized not available
                print("    Using prepared geometry approach (slower)...")
                from shapely.prepared import prep
                boundary_geom = self.boundary_gdf.iloc[0].geometry
                prepared_boundary = prep(boundary_geom)
                
                # Still faster than nested loop - process in chunks
                chunk_size = 10000
                mask = np.ones(len(grid_points), dtype=bool)
                
                for i in range(0, len(grid_points), chunk_size):
                    end_i = min(i + chunk_size, len(grid_points))
                    chunk_points = [Point(x, y) for x, y in grid_points[i:end_i]]
                    mask[i:end_i] = [prepared_boundary.contains(pt) for pt in chunk_points]
                    
                    if i % 50000 == 0:
                        print(f"    Progress: {i:,}/{len(grid_points):,} points checked")
                
                mask_reshaped = mask.reshape(grid_smooth.shape)
                grid_smooth[~mask_reshaped] = np.nan
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 15))
        
        im = ax.imshow(
            grid_smooth,
            extent=[xmin, xmax, ymin, ymax],
            origin='lower',
            cmap='viridis_r',
            norm=clrs.Normalize(vmin=vmin_km, vmax=vmax_km),
            alpha=0.7,
            zorder=1
        )
        
        # Plot streets in background
        self.streets_gdf.plot(ax=ax, color='black', linewidth=0.4, alpha=0.4, zorder=2)
        
        # Plot park polygons (not entrance points)
        self.parks_gdf.plot(ax=ax, color='green', edgecolor='darkgreen', alpha=0.6, linewidth=1.5, zorder=3)
        
        # Plot boundary if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
        
        cbar = fig.colorbar(im, ax=ax, extend='max', shrink=0.7)
        cbar.set_label('Distance to high-quality parks (kilometers, weighted by LTS)')
        
        ax.set_title(f'Accessibility to High-Quality Parks in {self.city_name} (Heat Map)', fontsize=16)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{base_filename}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {base_filename}_heatmap.png")
    
    def _create_histogram(self, base_filename):
        """Create a histogram of park distances."""
        print("Creating distance histogram...")
        
        valid_parcels = self.parcels_gdf[self.parcels_gdf['park_distance'] < float('inf')].copy()
        
        if len(valid_parcels) == 0:
            print("Warning: No valid distances for histogram.")
            return
        
        # Convert distances to kilometers
        valid_parcels['park_distance_km'] = valid_parcels['park_distance'] / 1000.0
        
        # Calculate statistics
        mean_dist = valid_parcels['park_distance_km'].mean()
        median_dist = valid_parcels['park_distance_km'].median()
        percentile_95 = valid_parcels['park_distance_km'].quantile(0.95)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create histogram
        n, bins, patches = ax.hist(
            valid_parcels['park_distance_km'],
            bins=50,
            color='steelblue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add vertical lines for statistics
        ax.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f} km')
        ax.axvline(median_dist, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f} km')
        ax.axvline(percentile_95, color='purple', linestyle='--', linewidth=2, label=f'95th %ile: {percentile_95:.2f} km')
        
        # Labels and title
        ax.set_xlabel('Weighted Distance to Nearest Park (km)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Parcels', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Distribution of Park Accessibility in {self.city_name}\n(LTS-Weighted Network Distance)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add statistics box
        stats_text = (
            f"Statistics:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Total parcels: {len(valid_parcels):,}\n"
            f"Mean: {mean_dist:.2f} km\n"
            f"Median: {median_dist:.2f} km\n"
            f"Min: {valid_parcels['park_distance_km'].min():.2f} km\n"
            f"Max: {valid_parcels['park_distance_km'].max():.2f} km\n"
            f"Std Dev: {valid_parcels['park_distance_km'].std():.2f} km\n"
            f"95th %ile: {percentile_95:.2f} km"
        )
        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=9,
            family='monospace'
        )
        
        plt.tight_layout()
        
        # Save figure
        histogram_filename = f'{base_filename}_histogram.png'
        plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved histogram to {histogram_filename}")
        print(f"  Mean distance: {mean_dist:.2f} km")
        print(f"  Median distance: {median_dist:.2f} km")
        print(f"  95th percentile: {percentile_95:.2f} km")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'='*60}")
        print(f"Starting analysis for {self.city_name}")
        print(f"{'='*60}\n")
        
        self.load_data()
        self.build_network()
        self.calculate_distances()
        self.calculate_parcel_distances()
        self.create_visualizations()
        self.export_parcels_geojson()

        print(f"\nAnalysis complete for {self.city_name}!")
        return self.parcels_gdf


def run_multi_city_analysis(cities_config):
    """
    Run park proximity analysis for multiple cities.
    
    Parameters:
    -----------
    cities_config : dict
        Dictionary mapping city names to their configuration dictionaries
        
    Returns:
    --------
    dict : Dictionary mapping city names to their results DataFrames
    """
    results = {}
    
    for city_name, config in cities_config.items():
        analyzer = ParkProximityAnalyzer(city_name, config)
        results[city_name] = analyzer.run_full_analysis()
    
    return results


# Example usage
if __name__ == "__main__":
    # Multi-city configuration dictionary
    # Note: source_crs and target_crs are no longer needed!
    # All data is automatically reprojected to EPSG:6350 (NAD83(2011) / Conus Albers)
    cities_config = {
        # 'Boston': {
        #     'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
        #     'parks_path': 'Data/Raw/Boston/boston_parks.geojson',
        #     'streets_path': 'Data/Processed/LTS/lts_bos.geojson',
        #     'boundary_path': 'Data/Raw/Boston/boston_neighborhood_boundaries.geojson.json',
        #     'output_dir': 'Visualizations/Boston',
        #     'lts_column': 'PC2_norm',
        #     'boundary_county_column': 'name',
        #     'combine_boundaries': True,
        #     'park_buffer': 50.0,           # meters
        #     'entrance_tolerance': 5.0,      # meters
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        # 'Houston': {
        #     'parcels_path': 'Data/Raw/Houston/houstonparcelsclipped.geojson',
        #     'parks_path': 'Data/Raw/Houston/COH_PARKS_(City_of_Houston).geojson',
        #     'streets_path': 'Data/Processed/LTS/lts_hou.geojson',
        #     'boundary_path': 'Data/Raw/Houston/houstoncitylimits.geojson',
        #     'output_dir': 'Visualizations/Houston',
        #     'lts_column': 'PC2_norm',
        #     'combine_boundaries': False,
        #     'park_buffer': 50.0,           # meters
        #     'entrance_tolerance': 5.0,      # meters
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        'NYC': {
            'parcels_path': 'Data/Raw/NYC/NYC_clipped_parcels.geojson',
            'parks_path': 'Data/Raw/NYC/Parks_Properties_20251120.geojson',
            'streets_path': 'Data/Processed/LTS/lts_nyc.geojson',
            'boundary_path': 'Data/Raw/NYC/NY_County_FeaturesToJSON.geojson',
            'output_dir': 'Visualizations/NYC',
            'lts_column': 'PC2_norm',
            'combine_boundaries': False,
            'park_buffer': 50.0,           # meters
            'entrance_tolerance': 5.0,      # meters
            'heatmap_resolution': 100,
            'heatmap_neighbors': 5,
            'heatmap_smoothing': 1
        },
        'LA': {
            'parcels_path': 'Data/Raw/LA/LA_clipped_parcels.geojson',
            'parks_path': 'Data/Raw/LA/la_parks.geojson',
            'streets_path': 'Data/Processed/LTS/lts_la.geojson',
            'boundary_path': 'Data/Raw/LA/City_Boundary.geojson',
            'output_dir': 'Visualizations/LA',
            'lts_column': 'PC2_norm',
            'combine_boundaries': False,
            'park_buffer': 50.0,           # meters
            'entrance_tolerance': 5.0,      # meters
            'heatmap_resolution': 100,
            'heatmap_neighbors': 5,
            'heatmap_smoothing': 1
        },
        'SF': {
            'parcels_path': 'Data/Raw/SF/Parcels_–_Active_and_Retired_20251208.geojson',
            'parks_path': 'Data/Raw/SF/Recreation_and_Parks_Properties_20251208.geojson',
            'streets_path': 'Data/Processed/LTS/lts_sf.geojson',
            'boundary_path': 'Data/Raw/SF/SF_Boundary.geojson',
            'output_dir': 'Visualizations/SF',
            'lts_column': 'PC1',
            'combine_boundaries': False,
            'park_buffer': 50.0,           # meters
            'entrance_tolerance': 5.0,      # meters
            'heatmap_resolution': 100,
            'heatmap_neighbors': 5,
            'heatmap_smoothing': 1
        }
    }
    
    # Run analysis for all cities
    results = run_multi_city_analysis(cities_config)
    
    # Access results for each city
    for city_name, parcels_gdf in results.items():
        print(f"\n{city_name} Results:")
        print(f"  Total parcels analyzed: {len(parcels_gdf)}")
        valid = parcels_gdf[parcels_gdf['park_distance'] < float('inf')]
        if len(valid) > 0:
            print(f"  Mean distance to parks: {valid['park_distance'].mean():.2f} meters")
            print(f"  Median distance to parks: {valid['park_distance'].median():.2f} meters")