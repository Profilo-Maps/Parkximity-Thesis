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
from shapely.strtree import STRtree
from pyproj import Transformer
import heapq
from pathlib import Path
import pickle


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
        
        # Load parks and filter to polygons only
        parks_raw = gpd.read_file(self.config['parks_path']).to_crs(self.config['target_crs'])
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
        
        parks_validated = gpd.GeoDataFrame(valid_parks, crs=self.config['target_crs'])
        
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
            
            # Ensure both are in the same CRS
            if parks_validated.crs != self.boundary_gdf.crs:
                self.boundary_gdf = self.boundary_gdf.to_crs(parks_validated.crs)
            
            # Clip parks to boundary - keep only parts within boundary
            try:
                self.parks_gdf = gpd.clip(parks_validated, self.boundary_gdf)
                
                removed_park_count = original_park_count - len(self.parks_gdf)
                print(f"Clipped parks: {len(self.parks_gdf)} parks kept, {removed_park_count} parks removed (outside boundary)")
                
                if len(self.parks_gdf) == 0:
                    print("WARNING: No parks remain after clipping! Check that:")
                    print("  - Boundary and parks are in compatible coordinate systems")
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
        self.streets_gdf = gpd.read_file(self.config['streets_path']).to_crs(self.config['target_crs'])
        print(f"Loaded {len(self.streets_gdf)} street segments")
        
        # Create diagnostic map before generating entrances
        self._create_diagnostic_map()
        
        # Load park entrances (generate from street-park intersections)
        self._generate_entrances_optimized()
    
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
    
    def _generate_entrances_optimized(self):
        """
        OPTIMIZED: Generate park entrances with batching, vectorization, and checkpointing.
        Memory-efficient version that processes parks in batches and saves progress to disk.
        """
        print("Generating park entrances (OPTIMIZED VERSION)...")
        
        # Verify we have parks and streets
        if len(self.parks_gdf) == 0 or len(self.streets_gdf) == 0:
            print("ERROR: No parks or streets available!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
            return
        
        # Verify CRS match
        if self.parks_gdf.crs != self.streets_gdf.crs:
            print(f"Converting parks to streets CRS...")
            self.parks_gdf = self.parks_gdf.to_crs(self.streets_gdf.crs)
        
        # Configuration
        buffer_distance = self.config.get('park_buffer', 50.0)
        entrance_tolerance = self.config.get('entrance_tolerance', 5.0)
        batch_size = self.config.get('entrance_batch_size', 50)  # Process 50 parks at a time
        
        print(f"Config: buffer={buffer_distance}m, tolerance={entrance_tolerance}m, batch_size={batch_size}")
        
        # Setup checkpoint directory
        checkpoint_dir = Path(self.config.get('output_dir', 'Visualizations')) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{self.city_name.lower().replace(' ', '_')}_entrances_checkpoint.pkl"
        
        # Check for existing checkpoint
        if checkpoint_file.exists():
            print(f"Found checkpoint file, loading previous progress...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_entrance_coords = checkpoint_data['coords']
                all_entrance_data = checkpoint_data['data']
                start_idx = checkpoint_data['last_processed'] + 1
            print(f"Resuming from park {start_idx}/{len(self.parks_gdf)}")
        else:
            all_entrance_coords = []
            all_entrance_data = []
            start_idx = 0
        
        # Create spatial index for streets (once)
        print("Building spatial index for streets...")
        streets_tree = STRtree(self.streets_gdf.geometry)
        
        # Process parks in batches
        total_parks = len(self.parks_gdf)
        parks_without_entrances = []
        
        for batch_start in range(start_idx, total_parks, batch_size):
            batch_end = min(batch_start + batch_size, total_parks)
            batch_parks = self.parks_gdf.iloc[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start}-{batch_end}/{total_parks} "
                f"({100*batch_end//total_parks}%) - {len(all_entrance_coords)} entrances so far")
            
            # Process this batch
            batch_coords, batch_data, batch_no_access = self._process_park_batch(
                batch_parks, streets_tree, buffer_distance, entrance_tolerance
            )
            
            # Append to running totals
            all_entrance_coords.extend(batch_coords)
            all_entrance_data.extend(batch_data)
            parks_without_entrances.extend(batch_no_access)
            
            # Save checkpoint every batch
            checkpoint_data = {
                'coords': all_entrance_coords,
                'data': all_entrance_data,
                'last_processed': batch_end - 1,
                'parks_without_entrances': parks_without_entrances
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        print(f"\n✓ Processed all {total_parks} parks")
        print(f"  Total entrance candidates: {len(all_entrance_coords)}")
        
        # VECTORIZED GLOBAL DEDUPLICATION
        print("\nPerforming global deduplication (vectorized)...")
        unique_coords, unique_data = self._vectorized_deduplication(
            all_entrance_coords, all_entrance_data, entrance_tolerance
        )
        
        # Create GeoDataFrame
        if len(unique_coords) == 0:
            print("ERROR: No entrances generated!")
            self.entrances_gdf = gpd.GeoDataFrame(geometry=[], crs=self.config['target_crs'])
        else:
            unique_points = [Point(x, y) for x, y in unique_coords]
            self.entrances_gdf = gpd.GeoDataFrame(
                unique_data,
                geometry=unique_points,
                crs=self.config['target_crs']
            )
            
            removed = len(all_entrance_coords) - len(unique_coords)
            print(f"✓ Generated {len(self.entrances_gdf)} unique entrances")
            print(f"  Removed {removed} duplicates ({100*removed/len(all_entrance_coords):.1f}%)")
        
        # Store parks without entrances
        self.parks_without_entrances = parks_without_entrances
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"✓ Cleaned up checkpoint file")


    def _process_park_batch(self, batch_parks, streets_tree, buffer_distance, entrance_tolerance):
        """
        Process a batch of parks to generate entrance points.
        Uses vectorized operations within each park for speed.
        
        Returns:
        --------
        tuple: (coordinates, data, parks_without_access)
        """
        batch_coords = []
        batch_data = []
        parks_no_access = []
        
        for park_idx, park in batch_parks.iterrows():
            park_geom = park.geometry
            park_name = park.get('name', park.get('NAME', park.get('park_name', 
                                park.get('PARK_NAME', f'Unnamed_Park_{park_idx}'))))
            
            # Buffer and find intersecting streets
            buffered_park = park_geom.buffer(buffer_distance)
            potential_indices = streets_tree.query(buffered_park)
            
            if len(potential_indices) == 0:
                parks_no_access.append(park_name)
                continue
            
            intersecting_streets = self.streets_gdf.iloc[potential_indices]
            intersecting_streets = intersecting_streets[intersecting_streets.intersects(buffered_park)]
            
            if len(intersecting_streets) == 0:
                parks_no_access.append(park_name)
                continue
            
            # Collect all intersection points for this park
            park_points = []
            
            for street_idx, street in intersecting_streets.iterrows():
                try:
                    intersection = street.geometry.intersection(park_geom.boundary)
                except Exception:
                    continue
                
                if intersection.is_empty:
                    continue
                
                # Extract points based on geometry type
                points = self._extract_points_from_intersection(intersection)
                
                for pt in points:
                    if pt is not None and not pt.is_empty:
                        park_points.append({
                            'coord': (pt.x, pt.y),
                            'park_name': park_name,
                            'park_id': park_idx,
                            'street_id': street_idx,
                            'method': 'boundary_intersection'
                        })
            
            # Vectorized deduplication within this park
            if len(park_points) > 0:
                unique_park_points = self._deduplicate_park_points_vectorized(
                    park_points, entrance_tolerance
                )
                
                for pt_data in unique_park_points:
                    batch_coords.append(pt_data['coord'])
                    batch_data.append({
                        'park_name': pt_data['park_name'],
                        'park_id': pt_data['park_id'],
                        'street_id': pt_data['street_id'],
                        'method': pt_data['method']
                    })
            else:
                # Fallback: nearest point on boundary
                try:
                    nearest_street = intersecting_streets.iloc[0]
                    nearest_point = park_geom.boundary.interpolate(
                        park_geom.boundary.project(nearest_street.geometry.centroid)
                    )
                    if nearest_point is not None and not nearest_point.is_empty:
                        batch_coords.append((nearest_point.x, nearest_point.y))
                        batch_data.append({
                            'park_name': park_name,
                            'park_id': park_idx,
                            'street_id': nearest_street.name,
                            'method': 'nearest_point_fallback'
                        })
                except Exception:
                    parks_no_access.append(park_name)
        
        return batch_coords, batch_data, parks_no_access


    def _extract_points_from_intersection(self, intersection):
        """Extract point coordinates from various geometry types."""
        points = []
        
        geom_type = intersection.geom_type
        
        if geom_type == 'Point':
            points.append(intersection)
        elif geom_type == 'MultiPoint':
            points.extend(list(intersection.geoms))
        elif geom_type == 'LineString':
            coords = list(intersection.coords)
            if len(coords) >= 2:
                points.extend([Point(coords[0]), Point(coords[-1])])
        elif geom_type == 'MultiLineString':
            for line in intersection.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    points.extend([Point(coords[0]), Point(coords[-1])])
        elif geom_type == 'GeometryCollection':
            for geom in intersection.geoms:
                if geom.geom_type == 'Point':
                    points.append(geom)
                elif geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        points.extend([Point(coords[0]), Point(coords[-1])])
        
        return points


    def _deduplicate_park_points_vectorized(self, park_points, tolerance):
        """
        Vectorized deduplication of points within a single park.
        Much faster than nested loops for large point sets.
        """
        if len(park_points) <= 1:
            return park_points
        
        # Convert to numpy array for vectorized operations
        coords_array = np.array([pt['coord'] for pt in park_points])
        
        # Build KDTree for efficient spatial queries
        tree = cKDTree(coords_array)
        
        # Find all pairs within tolerance
        pairs = tree.query_pairs(r=tolerance)
        
        # Create set of indices to remove (keep first occurrence)
        indices_to_remove = set()
        for i, j in pairs:
            if i not in indices_to_remove:
                indices_to_remove.add(j)
        
        # Keep only unique points
        unique_points = [pt for i, pt in enumerate(park_points) if i not in indices_to_remove]
        
        return unique_points


    def _vectorized_deduplication(self, coords_list, data_list, tolerance):
        """
        Vectorized global deduplication across all entrances.
        Uses spatial indexing for O(n log n) performance instead of O(n²).
        """
        if len(coords_list) == 0:
            return [], []
        
        print(f"  Building spatial index for {len(coords_list)} points...")
        coords_array = np.array(coords_list)
        tree = cKDTree(coords_array)
        
        print(f"  Finding duplicates within {tolerance}m...")
        pairs = tree.query_pairs(r=tolerance)
        
        print(f"  Found {len(pairs)} duplicate pairs, filtering...")
        # Use set for O(1) lookup
        indices_to_remove = set()
        for i, j in pairs:
            if i not in indices_to_remove:
                indices_to_remove.add(j)
        
        # Filter in one pass
        unique_coords = [coords_list[i] for i in range(len(coords_list)) if i not in indices_to_remove]
        unique_data = [data_list[i] for i in range(len(data_list)) if i not in indices_to_remove]
        
        return unique_coords, unique_data
    
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
        
        # Run Dijkstra's algorithm with progress tracking
        print("Running Dijkstra's algorithm (this may take a few minutes for large networks)...")
        distances = {}
        paths = {}
        lts_sums = {}
        
        for source in source_nodes:
            distances[source] = 0
            paths[source] = [source]
            lts_sums[source] = 0
        
        queue = [(0, 0, source) for source in source_nodes]
        visited = set()
        
        # Progress tracking
        total_nodes = self.network_graph.number_of_nodes()
        last_progress = 0
        
        while queue:
            dist, lts_sum, current = heapq.heappop(queue)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Print progress every 10%
            progress = int((len(visited) / total_nodes) * 100)
            if progress >= last_progress + 10:
                print(f"  Progress: {progress}% ({len(visited):,}/{total_nodes:,} nodes)")
                last_progress = progress
            
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
        print(f"Calculated distances to {len(self.distances_dict)} nodes (100%)")
    
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
        
        # Create diagnostic map for parks without entrances
        if hasattr(self, 'parks_without_entrances'):
            self._create_parks_without_entrances_map(output_dir, self.parks_without_entrances)
        
        # Then create analysis maps
        self._create_point_map(base_filename)
        self._create_heatmap(base_filename)
        self._create_distance_histogram(base_filename)
    
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
    
    def _create_parks_without_entrances_map(self, output_dir, parks_without_entrances_list):
        """Create a diagnostic map showing parks without entrances."""
        print("Creating diagnostic map for parks without entrances...")
        
        if not parks_without_entrances_list or len(parks_without_entrances_list) == 0:
            print("  All parks have entrances - skipping diagnostic map")
            return
        
        filename = output_dir / f"{self.city_name.lower().replace(' ', '_')}_parks_without_entrances.png"
        
        # Create a set of park names without entrances for quick lookup
        parks_without_set = set(parks_without_entrances_list)
        
        # Split parks into two groups
        parks_with_entrances = []
        parks_without_entrances_geoms = []
        
        for idx, park in self.parks_gdf.iterrows():
            park_name = park.get('name', park.get('NAME', park.get('park_name', park.get('PARK_NAME', f'Unnamed_Park_{idx}'))))
            if park_name in parks_without_set:
                parks_without_entrances_geoms.append(park)
            else:
                parks_with_entrances.append(park)
        
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Plot streets as background
        if self.streets_gdf is not None and len(self.streets_gdf) > 0:
            self.streets_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)
        
        # Plot parks WITH entrances in green
        if len(parks_with_entrances) > 0:
            parks_with_gdf = gpd.GeoDataFrame(parks_with_entrances, crs=self.parks_gdf.crs)
            parks_with_gdf.plot(ax=ax, color='green', edgecolor='darkgreen', alpha=0.4, linewidth=0.8, zorder=2)
        
        # Plot parks WITHOUT entrances in red (highlighted)
        if len(parks_without_entrances_geoms) > 0:
            parks_without_gdf = gpd.GeoDataFrame(parks_without_entrances_geoms, crs=self.parks_gdf.crs)
            parks_without_gdf.plot(ax=ax, color='red', edgecolor='darkred', alpha=0.7, linewidth=1.5, zorder=3)
        
        # Plot boundary if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2.5, zorder=4)
        
        # Add title and labels
        ax.set_title(
            f'Diagnostic: Parks Without Entrances - {self.city_name}',
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
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='darkgreen', alpha=0.4, 
                  label=f'Parks WITH entrances ({len(parks_with_entrances)})'),
            Patch(facecolor='red', edgecolor='darkred', alpha=0.7, 
                  label=f'Parks WITHOUT entrances ({len(parks_without_entrances_geoms)})'),
            Patch(facecolor='lightgray', alpha=0.6, 
                  label=f'Street Network')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        # Add info box with details
        info_text = (
            f"Parks Analysis:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Total parks: {len(self.parks_gdf)}\n"
            f"With entrances: {len(parks_with_entrances)}\n"
            f"Without entrances: {len(parks_without_entrances_geoms)}\n"
            f"% without access: {100*len(parks_without_entrances_geoms)/len(self.parks_gdf):.1f}%\n"
            f"\n"
            f"Possible reasons for no entrances:\n"
            f"  • Park too small\n"
            f"  • No streets within buffer ({self.config.get('park_buffer', 50.0)}m)\n"
            f"  • Disconnected from street network\n"
            f"  • Interior courtyard/private park"
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
        
        # Sample a few parks without entrances and add their names to map
        if len(parks_without_entrances_geoms) > 0 and len(parks_without_entrances_geoms) <= 20:
            print(f"  Labeling {len(parks_without_entrances_geoms)} parks without entrances...")
            for park in parks_without_entrances_geoms[:20]:  # Limit to 20 labels
                park_name = park.get('name', park.get('NAME', park.get('park_name', park.get('PARK_NAME', 'Unnamed'))))
                centroid = park.geometry.centroid
                ax.annotate(
                    park_name,
                    xy=(centroid.x, centroid.y),
                    fontsize=6,
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved parks-without-entrances diagnostic map to {filename}")
        print(f"  → {len(parks_without_entrances_geoms)} parks highlighted in red")
    
    
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
    
    def _create_distance_histogram(self, base_filename):
        """Create histogram of weighted distances from parcels to nearest park."""
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
        # 'Boston': {
        #     'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
        #     'parks_path': 'Data/Raw/Boston/boston_parks.geojson',  # Corrected path for polygon parks
        #     'streets_path': 'Data/Processed/lts_bos.geojson',
        #     'boundary_path': 'Data/Raw/Boston/boston_neighborhood_boundaries.geojson.json',
        #     'source_crs': 'EPSG:4326',
        #     'target_crs': 'EPSG:2227',
        #     'output_dir': 'Visualizations',
        #     'lts_column': 'PC2_norm',
        #     'boundary_county_column': 'name',
        #     'combine_boundaries': True,
        #     'park_buffer': 50.0,
        #     'entrance_tolerance': 5.0,
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        # },
        # 'LA': {
        #     'parcels_path': 'Data/Raw/LA/LA_clipped_parcels.geojson',
        #     'parks_path': 'Data/Raw/LA/la_parks.geojson',  # Corrected path for polygon parks
        #     'streets_path': 'Data/Processed/lts_la.geojson',
        #     'boundary_path': 'Data/Raw/LA/City_Boundary.geojson',
        #     'source_crs': 'EPSG:2229',
        #     'target_crs': 'EPSG:4326',
        #     'output_dir': 'Visualizations',
        #     'lts_column': 'PC2_norm',
        #     # 'boundary_county_column': 'name',
        #     'combine_boundaries': False,
        #     'park_buffer': 50.0,
        #     'entrance_tolerance': 5.0,
        #     'heatmap_resolution': 100,
        #     'heatmap_neighbors': 5,
        #     'heatmap_smoothing': 1
        #     }
        'Houston': {
            'parcels_path': 'Data/Raw/Houston/houstonparcelsclipped.geojson',
            'parks_path': 'Data/Raw/Houston/COH_PARKS_(City_of_Houston).geojson',  # Corrected path for polygon parks
            'streets_path': 'Data/Processed/lts_hou.geojson',
            'boundary_path': 'Data/Raw/Houston/houstoncitylimits.geojson',
            'source_crs': 'EPSG:2278',
            'target_crs': 'EPSG:4326',
            'output_dir': 'Visualizations',
            'lts_column': 'PC2_norm',
            # 'boundary_county_column': 'name',
            'combine_boundaries': False,
            'park_buffer': 50.0,
            'entrance_tolerance': 5.0,
            'heatmap_resolution': 100,
            'heatmap_neighbors': 5,
            'heatmap_smoothing': 1
            }
    }
    
    # Run analysis for all cities
    results = run_multi_city_analysis(cities_config)
    
    # Print results
    for city_name, parcels_gdf in results.items():
        print(f"\n{city_name} Results:")
        print(f"  Total parcels: {len(parcels_gdf)}")
        valid = parcels_gdf[parcels_gdf['park_distance'] < float('inf')]
        if len(valid) > 0:
            print(f"  Mean distance: {valid['park_distance'].mean():.2f} meters")
            print(f"  Median distance: {valid['park_distance'].median():.2f} meters")