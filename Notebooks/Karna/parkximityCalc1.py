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
from pyproj import Transformer
import heapq
from pathlib import Path


class ParkProximityAnalyzer:
    """
    Analyzes park accessibility for a city using street network analysis
    and creates heatmap visualizations.
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
                - parcels_path: Path to parcels CSV
                - parks_path: Path to parks GeoJSON
                - streets_path: Path to streets GeoJSON
                - boundary_path: Path to county boundary CSV (optional)
                - source_crs: Source coordinate reference system (e.g., "EPSG:4326")
                - target_crs: Target CRS for analysis (e.g., "EPSG:2227")
                - output_dir: Directory for output files
                - combine_boundaries: Boolean, if True combines neighborhood polygons into city boundary
        """
        self.city_name = city_name
        self.config = config
        self.parcels_gdf = None
        self.parks_gdf = None
        self.streets_gdf = None
        self.entrances_gdf = None
        self.boundary_gdf = None
        self.network_graph = None
        self.distances_dict = None
        self.paths_dict = None
        
    def load_data(self):
        """Load all required data files and transform to target CRS."""
        print(f"Loading data for {self.city_name}...")
        
        # Load boundary first (if available) so we can clip parcels
        if 'boundary_path' in self.config and self.config['boundary_path']:
            self._load_boundary()
        
        # Load and process parcels (will be clipped if boundary is available)
        self._load_parcels()
        
        # Load parks
        self.parks_gdf = gpd.read_file(self.config['parks_path']).to_crs(self.config['target_crs'])
        print(f"Loaded {len(self.parks_gdf)} parks")
        
        # Load streets
        self.streets_gdf = gpd.read_file(self.config['streets_path']).to_crs(self.config['target_crs'])
        print(f"Loaded {len(self.streets_gdf)} street segments")
        
        # Create diagnostic map before generating entrances
        self._create_diagnostic_map()
        
        # Load park entrances (generate from street-park intersections)
        self._generate_entrances()
    
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
            f"CRS: {self.parks_gdf.crs}\n"
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
        """Load and process parcel data with coordinate transformation and optional clipping."""
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
                
                print(f"  Successfully loaded {len(parcels_gdf_raw)} parcels (some may have None geometry)")
                
            except Exception as e:
                print(f"  Error loading parcels: {e}")
                print(f"  Creating empty parcel dataset - analysis will continue without parcels")
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
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
            
            # Keep only valid parcels
            if len(valid_indices) > 0:
                self.parcels_gdf = parcels_gdf_raw.iloc[valid_indices].copy().to_crs(self.config['target_crs'])
            else:
                print(f"  Warning: No valid parcels found!")
                self.parcels_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
            
            if invalid_count > 0:
                print(f"  Skipped {invalid_count} parcels with invalid/None geometries")
                if invalid_count > 5:
                    print(f"  (Only first 5 detailed warnings shown)")
            
            print(f"Loaded {len(self.parcels_gdf)} valid parcels from GeoJSON")
            
        else:
            # Load from CSV with coordinate transformation
            print("Loading parcels from CSV...")
            parcels_df = pd.read_csv(parcels_path, engine='python', on_bad_lines='warn')
            
            # Create transformer
            transformer = Transformer.from_crs(
                self.config.get('source_crs', 'EPSG:4326'),
                self.config['target_crs'],
                always_xy=False
            )
            
            # Transform coordinates
            geometries = []
            valid_indices = []
            
            for idx, row in parcels_df.iterrows():
                try:
                    lat = row['centroid_latitude']
                    lon = row['centroid_longitude']
                    
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    
                    x, y = transformer.transform(lat, lon)
                    geometries.append(Point(x, y))
                    valid_indices.append(idx)
                except Exception:
                    continue
            
            valid_parcels_df = parcels_df.iloc[valid_indices].copy()
            self.parcels_gdf = gpd.GeoDataFrame(
                valid_parcels_df,
                geometry=geometries,
                crs=self.config['target_crs']
            )
            print(f"Loaded {len(self.parcels_gdf)} valid parcels from CSV")
        
        # Clip parcels to boundary if boundary is available
        if self.boundary_gdf is not None and len(self.parcels_gdf) > 0:
            self._clip_parcels_to_boundary()
    
    def _clip_parcels_to_boundary(self):
        """Clip parcels to the city boundary polygon."""
        print(f"Clipping parcels to {self.city_name} boundary...")
        
        original_count = len(self.parcels_gdf)
        
        # Ensure both are in the same CRS
        if self.parcels_gdf.crs != self.boundary_gdf.crs:
            self.boundary_gdf = self.boundary_gdf.to_crs(self.parcels_gdf.crs)
        
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
            print("  - Boundary and parcels are in compatible coordinate systems")
            print("  - Boundary polygon correctly represents the city area")
            print("  - Parcel coordinates are correct")
    
    def _generate_entrances(self):
        """
        Assign park entrances as points where the street network intersects 
        with park polygons. For point parks, uses the park point itself as entrance.
        """
        print("Generating park entrances from street-park intersections...")
        
        # Verify we have parks and streets to work with
        if len(self.parks_gdf) == 0:
            print("ERROR: No parks available for entrance generation!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
            return
        
        if len(self.streets_gdf) == 0:
            print("ERROR: No streets available for entrance generation!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
            return
        
        # Verify CRS match
        if self.parks_gdf.crs != self.streets_gdf.crs:
            print(f"WARNING: CRS mismatch detected!")
            print(f"  Parks CRS: {self.parks_gdf.crs}")
            print(f"  Streets CRS: {self.streets_gdf.crs}")
            print(f"  Converting parks to streets CRS...")
            self.parks_gdf = self.parks_gdf.to_crs(self.streets_gdf.crs)
        
        # Check if parks are points or polygons
        point_parks = 0
        polygon_parks = 0
        for geom in self.parks_gdf.geometry:
            if geom.geom_type in ['Point', 'MultiPoint']:
                point_parks += 1
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                polygon_parks += 1
        
        print(f"Park types: {point_parks} points, {polygon_parks} polygons")
        
        # Get buffer distance from config (default 50 meters)
        buffer_distance = self.config.get('park_buffer', 50.0)
        print(f"Using buffer distance: {buffer_distance} meters")
        
        entrance_points = []
        entrance_data = []
        parks_with_entrances = 0
        
        for park_idx, park in self.parks_gdf.iterrows():
            park_geom = park.geometry
            park_name = park.get('name', park.get('NAME', f'Park_{park_idx}'))
            
            # Handle point parks differently
            if park_geom.geom_type == 'Point':
                # For point parks, use the point itself as the entrance
                entrance_points.append(park_geom)
                entrance_data.append({
                    'park_name': park_name,
                    'park_id': park_idx,
                    'street_id': None,
                    'method': 'point_park'
                })
                parks_with_entrances += 1
                continue
            elif park_geom.geom_type == 'MultiPoint':
                # For multipoint, use all points
                for point in park_geom.geoms:
                    entrance_points.append(point)
                    entrance_data.append({
                        'park_name': park_name,
                        'park_id': park_idx,
                        'street_id': None,
                        'method': 'multipoint_park'
                    })
                parks_with_entrances += 1
                continue
            
            # For polygon parks, buffer and find intersections
            buffered_park = park_geom.buffer(buffer_distance)
            
            # Find all streets that intersect with the buffered park
            intersecting_streets = self.streets_gdf[self.streets_gdf.intersects(buffered_park)]
            
            if len(intersecting_streets) == 0:
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
        
        if len(entrance_points) == 0:
            print("ERROR: No street-park intersections found!")
            print("This could mean:")
            print("  - Parks and streets are in different coordinate systems")
            print("  - Park polygons don't overlap with street network")
            print("  - Parks are too small or streets don't reach them")
            print("  - Data quality issues with park or street geometries")
            print("\nDebugging info:")
            print(f"  Number of parks: {len(self.parks_gdf)}")
            print(f"  Number of streets: {len(self.streets_gdf)}")
            print(f"  Parks CRS: {self.parks_gdf.crs}")
            print(f"  Streets CRS: {self.streets_gdf.crs}")
            print(f"  Parks bounds: {self.parks_gdf.total_bounds}")
            print(f"  Streets bounds: {self.streets_gdf.total_bounds}")
            
            # Sample a few parks and check their geometries
            print("\nSample park analysis:")
            for i, (idx, park) in enumerate(self.parks_gdf.head(3).iterrows()):
                park_name = park.get('name', park.get('NAME', f'Park_{idx}'))
                print(f"  Park: {park_name}")
                print(f"    Type: {park.geometry.geom_type}")
                print(f"    Area: {park.geometry.area:.2f}")
                print(f"    Bounds: {park.geometry.bounds}")
                
                # Check if any streets are nearby
                buffered = park.geometry.buffer(50)  # 50 meter buffer
                nearby = self.streets_gdf[self.streets_gdf.intersects(buffered)]
                print(f"    Streets within 50m: {len(nearby)}")
            
            # Create empty GeoDataFrame
            self.entrances_gdf = gpd.GeoDataFrame(
                geometry=[],
                crs=self.config['target_crs']
            )
            return
        
        # Remove duplicate points (within a small tolerance)
        unique_points = []
        unique_data = []
        tolerance = self.config.get('entrance_tolerance', 1.0)  # 1 meter default
        
        for i, point in enumerate(entrance_points):
            # Skip None points
            if point is None or point.is_empty:
                continue
                
            is_duplicate = False
            for existing_point in unique_points:
                if existing_point is not None and point.distance(existing_point) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
                unique_data.append(entrance_data[i])
        
        self.entrances_gdf = gpd.GeoDataFrame(
            unique_data,
            geometry=unique_points,
            crs=self.config['target_crs']
        )
        
        print(f"Generated {len(self.entrances_gdf)} unique park entrance points from intersections")
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
        
        # Create GeoDataFrame with the boundary
        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[combined_boundary],
            crs=boundary_gdf.crs
        ).to_crs(self.config['target_crs'])
        print(f"Loaded boundary for {self.city_name}")
    
    def _load_boundary_from_csv(self):
        """Load city/county boundary from CSV file with WKT geometry."""
        print(f"Loading boundary from CSV for {self.city_name}...")
        
        county_df = pd.read_csv(self.config['boundary_path'])
        wkt_column = self.config.get('boundary_wkt_column', 'the_geom')
        county_column = self.config.get('boundary_county_column', 'COUNTY')
        
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
        
        self.boundary_gdf = gpd.GeoDataFrame(
            geometry=[combined_boundary],
            crs=self.config.get('source_crs', 'EPSG:4326')
        ).to_crs(self.config['target_crs'])
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
                
                if length > 0:
                    weight_val = length * weight
                    all_edges.append((
                        start, end,
                        {'weight': weight_val, 'lts_value': weight, 'length': length}
                    ))
        
        self.network_graph.add_edges_from(all_edges)
        print(f"Graph created with {self.network_graph.number_of_nodes()} nodes "
              f"and {self.network_graph.number_of_edges()} edges")
    
    def calculate_distances(self):
        """Calculate distances from all network nodes to nearest park entrance."""
        print("Calculating network distances to parks...")
        
        # Check if we have any park entrances
        if self.entrances_gdf is None or len(self.entrances_gdf) == 0:
            print("ERROR: No park entrances found!")
            print("This means the street network does not intersect with any park polygons.")
            print("Possible causes:")
            print("  1. Parks and streets are in different coordinate systems (CRS mismatch)")
            print("  2. Parks are very small and don't touch the street network")
            print("  3. Street or park data quality issues")
            print("  4. Parks were clipped out by the boundary")
            print("\nTroubleshooting steps:")
            print("  - Verify parks_gdf and streets_gdf have the same CRS")
            print("  - Check if parks and streets visually overlap in QGIS")
            print("  - Try increasing buffer around parks for intersection detection")
            raise ValueError("No park entrances generated - cannot calculate distances")
        
        # Get source nodes (nearest to park entrances)
        network_nodes_array = np.array(list(self.network_graph.nodes()))
        point_coords = np.array([(geom.x, geom.y) for geom in self.entrances_gdf.geometry])
        
        if len(point_coords) == 0:
            raise ValueError("Park entrances GeoDataFrame is empty - cannot calculate distances")
        
        tree = cKDTree(network_nodes_array)
        distances, indices = tree.query(point_coords)
        source_nodes = [tuple(network_nodes_array[i]) for i in indices]
        source_nodes = list(set(source_nodes))
        print(f"Found {len(source_nodes)} unique source nodes from park entrances")
        
        # Run Dijkstra's algorithm
        distances = {}
        paths = {}
        lts_sums = {}
        
        for source in source_nodes:
            distances[source] = 0
            paths[source] = [source]
            lts_sums[source] = 0
        
        queue = [(0, 0, source) for source in source_nodes]
        visited = set()
        
        while queue:
            dist, lts_sum, current = heapq.heappop(queue)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.network_graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                edge_data = self.network_graph[current][neighbor]
                edge_length = edge_data['length']
                edge_lts = edge_data['lts_value']
                
                new_dist = distances[current] + edge_length
                new_lts_sum = lts_sums[current] + edge_lts
                
                if neighbor not in distances or new_dist < distances[neighbor] or \
                   (new_dist == distances[neighbor] and new_lts_sum < lts_sums[neighbor]):
                    
                    distances[neighbor] = new_dist
                    lts_sums[neighbor] = new_lts_sum
                    paths[neighbor] = paths[current] + [neighbor]
                    heapq.heappush(queue, (new_dist, new_lts_sum, neighbor))
        
        self.distances_dict = distances
        self.paths_dict = paths
        print(f"Calculated distances to {len(self.distances_dict)} nodes")
    
    def calculate_parcel_distances(self):
        """Calculate park distances for all parcels."""
        print("Calculating parcel distances to parks...")
        
        network_nodes_array = np.array(list(self.network_graph.nodes()))
        tree = cKDTree(network_nodes_array)
        
        # Handle both Point and Polygon geometries by using centroids
        parcel_coords = np.array([
            (geom.centroid.x, geom.centroid.y) if geom.geom_type in ['Polygon', 'MultiPolygon'] 
            else (geom.x, geom.y) 
            for geom in self.parcels_gdf.geometry
        ])
        nn_distances, nn_indices = tree.query(parcel_coords)
        
        for i, (parcel_idx, parcel) in enumerate(self.parcels_gdf.iterrows()):
            parcel_to_node_distance = nn_distances[i]
            nearest_node = tuple(network_nodes_array[nn_indices[i]])
            
            if nearest_node in self.distances_dict:
                node_to_park_distance = self.distances_dict[nearest_node]
                total_distance = parcel_to_node_distance + node_to_park_distance
            else:
                node_to_park_distance = float('inf')
                total_distance = float('inf')
            
            self.parcels_gdf.loc[parcel_idx, 'parcel_to_node_distance'] = parcel_to_node_distance
            self.parcels_gdf.loc[parcel_idx, 'node_to_park_distance'] = node_to_park_distance
            self.parcels_gdf.loc[parcel_idx, 'park_distance'] = total_distance
        
        valid_distances = self.parcels_gdf[self.parcels_gdf['park_distance'] < float('inf')]['park_distance']
        print(f"Valid distances: min={valid_distances.min():.2f}, "
              f"max={valid_distances.max():.2f}, mean={valid_distances.mean():.2f}")
        print(f"Parcels with valid distances: {len(valid_distances)}/{len(self.parcels_gdf)}")
    
    def create_visualizations(self):
        """Create both point map and smoothed heatmap visualizations."""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = output_dir / f"{self.city_name.lower().replace(' ', '_')}_parkximity"
        
        # Create entrance map first
        self._create_entrance_map(output_dir)
        
        # Then create analysis maps
        self._create_point_map(base_filename)
        self._create_heatmap(base_filename)
    
    def _create_entrance_map(self, output_dir):
        """Create a map showing parks and their generated entrance points."""
        print("Creating park entrance map...")
        
        if self.entrances_gdf is None or len(self.entrances_gdf) == 0:
            print("Warning: No entrances to visualize")
            return
        
        filename = output_dir / f"{self.city_name.lower().replace(' ', '_')}_park_entrances.png"
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Plot streets as background
        if self.streets_gdf is not None and len(self.streets_gdf) > 0:
            self.streets_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)
        
        # Plot parks
        self.parks_gdf.plot(ax=ax, color='green', edgecolor='darkgreen', alpha=0.5, linewidth=1, zorder=2)
        
        # Plot entrance points
        self.entrances_gdf.plot(ax=ax, color='red', markersize=30, alpha=0.8, zorder=3, label='Park Entrances')
        
        # Plot boundary if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2, zorder=4)
        
        # Add title and labels
        ax.set_title(f'Park Entrances Generated for {self.city_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label=f'Parks ({len(self.parks_gdf)})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label=f'Entrances ({len(self.entrances_gdf)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Add info box
        info_text = (
            f"Parks: {len(self.parks_gdf)}\n"
            f"Entrances: {len(self.entrances_gdf)}\n"
            f"Avg entrances per park: {len(self.entrances_gdf) / len(self.parks_gdf):.1f}"
        )
        ax.text(
            0.02, 0.02,
            info_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'),
            verticalalignment='bottom',
            fontsize=10
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved entrance map to {filename}")
    
    def _create_point_map(self, base_filename):
        """Create point-based visualization."""
        print("Creating point map visualization...")
        
        valid_parcels = self.parcels_gdf[self.parcels_gdf['park_distance'] < float('inf')].copy()
        
        if len(valid_parcels) == 0:
            print("Warning: No valid distances for visualization.")
            return
        
        valid_parcels['park_distance_km'] = valid_parcels['park_distance'] / 1000.0
        vmin_km = valid_parcels['park_distance_km'].min()
        vmax_km = np.percentile(valid_parcels['park_distance_km'], 95)
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Streets
        self.streets_gdf.plot(ax=ax, color='gray', linewidth=0.4, alpha=0.5)
        
        # Get coordinates - handle both Point and Polygon geometries
        x_coords = []
        y_coords = []
        for geom in valid_parcels.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                centroid = geom.centroid
                x_coords.append(centroid.x)
                y_coords.append(centroid.y)
            else:  # Point
                x_coords.append(geom.x)
                y_coords.append(geom.y)
        
        # Parcel points
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=valid_parcels['park_distance_km'],
            cmap='viridis_r',
            norm=clrs.Normalize(vmin=vmin_km, vmax=vmax_km),
            alpha=0.6,
            s=20,
            edgecolor=None,
            zorder=2
        )
        
        # Parks
        self.parks_gdf.plot(ax=ax, color='blue', edgecolor='darkblue', alpha=0.7, zorder=3)
        
        # Boundary
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4)
        
        cbar = fig.colorbar(scatter, ax=ax, extend='max', shrink=0.7)
        cbar.set_label('Distance to high-quality parks (kilometers, weighted by LTS)')
        
        ax.set_title(f'Accessibility to High-Quality Parks in {self.city_name}', fontsize=16)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        ax.text(
            0.02, 0.02,
            f"Analysis of {len(valid_parcels)} parcels\nto {len(self.parks_gdf)} high-quality parks",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        plt.tight_layout()
        plt.savefig(f'{base_filename}_pointmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved point map to {base_filename}_pointmap.png")
    
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
        
        self.streets_gdf.plot(ax=ax, color='black', linewidth=0.4, alpha=0.4, zorder=2)
        self.parks_gdf.plot(ax=ax, color='blue', edgecolor='darkblue', alpha=0.7, zorder=3)
        
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
    cities_config = {
        'Boston': {
            'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
            'parks_path': 'Data/Raw/Boston/Boston_park_features.geojson.json',
            'streets_path': 'Data/Processed/lts_bos.geojson',
            'boundary_path': 'Data/Raw/Boston/boston_neighborhood_boundaries.geojson.json',
            'source_crs': 'EPSG:4326',
            'target_crs': 'EPSG:2227',
            'output_dir': 'Visualizations',
            'lts_column': 'PC2_norm',
            # 'boundary_name': 'Boston',  # Optional: filter by name column
            'boundary_county_column': 'name',  # Column to filter on (for GeoJSON)
            'combine_boundaries': True,
            'park_buffer': 50.0,  # Buffer parks by 50 meters (parks appear small in Boston data)
            'entrance_tolerance': 5.0,  # Merge entrances within 5 meters
            'heatmap_resolution': 100,
            'heatmap_neighbors': 5,
            'heatmap_smoothing': 1
        }
        # 'San Francisco': {
        #     'parcels_path': '/workspaces/final-proj-pretty-map-squadron/data/raw/sf_parcels.csv',
        #     'parks_path': '/workspaces/final-proj-pretty-map-squadron/data/processed/sf_top25_parks.geojson',
        #     'streets_path': '/workspaces/final-proj-pretty-map-squadron/data/processed/sf_lts_final.geojson',
        #     'boundary_path': '/workspaces/final-proj-pretty-map-squadron/data/raw/sf_boundary.geojson',
        #     'source_crs': 'EPSG:4326',
        #     'target_crs': 'EPSG:2227',
        #     'output_dir': '/workspaces/final-proj-pretty-map-squadron/vis/',
        #     'lts_column': 'final_lts',
        #     'boundary_name': 'San Francisco',  # Optional: filter by name column
        #     'boundary_county_column': 'name',  # Column to filter on (for GeoJSON)
        #     'combine_boundaries': False,  # Single city polygon
        #     'entrance_tolerance': 1.0,
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        # 'Oakland': {
        #     'parcels_path': '/path/to/oakland_parcels.csv',
        #     'parks_path': '/path/to/oakland_parks.geojson',
        #     'streets_path': '/path/to/oakland_streets.geojson',
        #     'boundary_path': '/path/to/oakland_neighborhoods.geojson',  # GeoJSON with multiple neighborhoods
        #     'source_crs': 'EPSG:4326',
        #     'target_crs': 'EPSG:2227',
        #     'output_dir': '/path/to/oakland_output/',
        #     'lts_column': 'final_lts',
        #     'boundary_name': 'Oakland',  # Optional: filter by name
        #     'boundary_county_column': 'city',  # Column containing city name
        #     'combine_boundaries': True,  # Combine multiple neighborhood polygons
        #     'entrance_tolerance': 1.0,
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        # 'Berkeley': {
        #     'parcels_path': '/path/to/berkeley_parcels.csv',
        #     'parks_path': '/path/to/berkeley_parks.geojson',
        #     'streets_path': '/path/to/berkeley_streets.geojson',
        #     'boundary_path': '/path/to/Bay_Area_County_Polygons.csv',  # CSV with WKT geometry
        #     'source_crs': 'EPSG:4326',
        #     'target_crs': 'EPSG:2227',
        #     'output_dir': '/path/to/berkeley_output/',
        #     'lts_column': 'final_lts',
        #     'boundary_name': 'Berkeley',
        #     'boundary_wkt_column': 'the_geom',  # WKT column (for CSV files)
        #     'boundary_county_column': 'COUNTY',  # Column to filter on (for CSV)
        #     'combine_boundaries': False,
        #     'entrance_tolerance': 1.0,
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        # Add more cities as needed
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