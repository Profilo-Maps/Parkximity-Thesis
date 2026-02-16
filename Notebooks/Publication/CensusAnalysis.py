"""
Census Data Analysis Pipeline

This module provides a complete pipeline for downloading census demographic data
and spatially joining it to parcels with parkximity calculations. It combines
the functionality of census data download and spatial aggregation into a single
streamlined workflow.

Key Features:
- Downloads ACS demographic data from Census API
- Clips census blocks to city boundaries
- Spatially joins census data to parcels
- Calculates derived demographic metrics
- Supports parallel processing for large datasets

Usage:
    from CensusAnalysis import CensusAnalyzer, multi_city_census_analysis
    
    # Single city
    analyzer = CensusAnalyzer('Boston', config)
    analyzer.full_analysis()
    
    # Multiple cities
    results = multi_city_census_analysis(cities_config)
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import requests
import time
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ======================================================================
# Census Variable Definitions
# ======================================================================

# Raw Census API variable names -> human-readable names
RAW_CENSUS_VARIABLES = {
    # Commuting data
    "B08301_001E": "total_commuters",
    "B08301_001M": "total_commuters_moe",
    "B08301_010E": "public_transit",
    "B08301_010M": "public_transit_moe",
    "B08301_018E": "bicycle",
    "B08301_018M": "bicycle_moe",
    "B08301_019E": "walked",
    "B08301_019M": "walked_moe",
    # Income
    "B19013_001E": "median_hh_income",
    "B19013_001M": "median_hh_income_moe",
    # Race/Ethnicity
    "B03002_001E": "total_population",
    "B03002_001M": "total_population_moe",
    "B03002_004E": "black_alone",
    "B03002_004M": "black_alone_moe",
    "B03002_012E": "hispanic_latino",
    "B03002_012M": "hispanic_latino_moe",
}

# Pre-computed variable names that may already be in the data
PRECOMPUTED_CENSUS_VARIABLES = [
    "pct_non_car_commute",
    "pct_non_car_commute_moe",
    "median_hh_income",
    "median_hh_income_moe",
    "pct_black_latino",
    "pct_black_latino_moe",
    "pct_black",
    "pct_black_moe",
    "pct_hispanic",
    "pct_hispanic_moe",
    "pct_latino",
    "pct_latino_moe",
    "total_population",
    "total_population_moe",
    "pct_transit",
    "pct_transit_moe",
    "pct_bicycle",
    "pct_bicycle_moe",
    "pct_walked",
    "pct_walked_moe",
]

# Columns to exclude from spatial join (metadata, not demographic data)
EXCLUDE_COLUMNS = [
    'STATEFP20', 'COUNTYFP20', 'TRACTCE20', 'BLOCKCE20',
    'GEOID20', 'GEOIDFQ20', 'NAME20', 'MTFCC20', 'UR20',
    'UACE20', 'FUNCSTAT20', 'ALAND20', 'AWATER20',
    'INTPTLAT20', 'INTPTLON20', 'HOUSING20', 'POP20',
    'block_group_geoid', 'GEOID', 'geometry',
    'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLOCKCE',
    'GEOID10', 'NAME10', 'NAME',
]


# ======================================================================
# Parallel Processing Helper Functions
# ======================================================================

def _validate_geometry_chunk(chunk):
    """Validate geometries in a chunk (for parallel processing)."""
    valid_mask = (
        chunk.geometry.notna() &
        ~chunk.geometry.is_empty &
        chunk.geometry.is_valid
    )
    return chunk.loc[valid_mask].copy()


def _clip_geometries_chunk(chunk, boundary_geom):
    """Clip geometries to boundary in a chunk (for parallel processing)."""
    return chunk[chunk.geometry.intersects(boundary_geom)].copy()


# ======================================================================
# Main Census Analyzer Class
# ======================================================================

class CensusAnalyzer:
    """
    Analyzes census demographics by downloading ACS data and spatially
    joining it to parcels with parkximity calculations.
    
    This class handles the complete pipeline from data download to
    final output generation.
    """
    
    def __init__(self, city_name, config, target_crs="EPSG:6350"):
        """
        Parameters
        ----------
        city_name : str
            Name of the city being analyzed.
        config : dict
            Configuration dictionary containing:
                - census_api_key: Census API key
                - state_fips: State FIPS code
                - county_fips: List of county FIPS codes
                - block_shapefile: Path to census block shapefile
                - boundary_geojson: Path to city boundary GeoJSON
                - parcels_path: Path to parcels with parkximity data
                - exports_output_dir: Base exports directory (census data goes to Exports/Data/{city}/)
                - acs_year: ACS year (default 2022)
                - n_jobs: Number of parallel workers (default: cpu_count() - 1)
        target_crs : str
            Target coordinate reference system (default EPSG:6350).
        """
        self.city_name = city_name
        self.config = config
        self.target_crs = target_crs
        
        self.blocks_gdf = None
        self.boundary_gdf = None
        self.parcels_gdf = None
        self.demographics_df = None
        self.result_gdf = None
        
        # Parallel processing configuration
        self.n_jobs = config.get('n_jobs', max(1, cpu_count() - 1))
        self.acs_year = config.get('acs_year', 2022)
    
    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    
    def load_census_blocks(self):
        """Load census block shapefile and clip to city boundary."""
        print(f"Loading census blocks for {self.city_name}...")
        
        # Load block shapefile
        block_path = Path(self.config['block_shapefile'])
        if not block_path.exists():
            raise FileNotFoundError(f"Block shapefile not found: {block_path}")
        
        self.blocks_gdf = gpd.read_file(block_path)
        print(f"  Loaded {len(self.blocks_gdf)} blocks from shapefile")
        
        # Load city boundary
        boundary_path = Path(self.config['boundary_geojson'])
        if not boundary_path.exists():
            raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
        
        self.boundary_gdf = gpd.read_file(boundary_path)
        print(f"  Loaded city boundary")
        
        # Clip blocks to city boundary
        self._clip_blocks_to_boundary()
        
        # Reproject to target CRS
        if self.blocks_gdf.crs != self.target_crs:
            print(f"  Reprojecting blocks: {self.blocks_gdf.crs} -> {self.target_crs}")
            self.blocks_gdf = self.blocks_gdf.to_crs(self.target_crs)
    
    def _clip_blocks_to_boundary(self):
        """Clip census blocks to city boundary using parallel processing."""
        print(f"  Clipping blocks to city boundary...")
        
        # Ensure same CRS
        if self.blocks_gdf.crs != self.boundary_gdf.crs:
            self.boundary_gdf = self.boundary_gdf.to_crs(self.blocks_gdf.crs)
        
        # Get boundary geometry
        boundary_geom = self.boundary_gdf.unary_union
        
        # Bounding box filter first
        bounds = self.boundary_gdf.total_bounds
        blocks_subset = self.blocks_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        print(f"  Reduced to {len(blocks_subset)} candidate blocks within bounding box")
        
        # Parallel clipping for large datasets
        if len(blocks_subset) > 10000:
            chunk_size = max(1000, len(blocks_subset) // (self.n_jobs * 2))
            chunks = [blocks_subset.iloc[i:i+chunk_size] 
                     for i in range(0, len(blocks_subset), chunk_size)]
            
            print(f"  Clipping using {self.n_jobs} workers ({len(chunks)} chunks)...")
            
            with Pool(processes=self.n_jobs) as pool:
                results = list(tqdm(
                    pool.starmap(
                        _clip_geometries_chunk,
                        [(chunk, boundary_geom) for chunk in chunks]
                    ),
                    total=len(chunks),
                    desc="  Clipping blocks",
                    unit="chunk"
                ))
            
            self.blocks_gdf = pd.concat(results, ignore_index=True)
        else:
            # Small dataset - use gpd.clip
            self.blocks_gdf = gpd.clip(blocks_subset, self.boundary_gdf)
        
        print(f"  ✓ {len(self.blocks_gdf)} blocks within city boundary")
    
    def load_parcels(self):
        """Load parcels with parkximity data."""
        print(f"Loading parcels for {self.city_name}...")
        
        parcels_path = Path(self.config['parcels_path'])
        if not parcels_path.exists():
            raise FileNotFoundError(f"Parcels file not found: {parcels_path}")
        
        self.parcels_gdf = gpd.read_file(parcels_path)
        print(f"  Loaded {len(self.parcels_gdf)} parcels")
        
        # Reproject to target CRS if needed
        if self.parcels_gdf.crs != self.target_crs:
            print(f"  Reprojecting parcels: {self.parcels_gdf.crs} -> {self.target_crs}")
            self.parcels_gdf = self.parcels_gdf.to_crs(self.target_crs)
    
    # ------------------------------------------------------------------
    # Census API Data Fetching
    # ------------------------------------------------------------------
    
    def fetch_acs_demographics(self):
        """Fetch ACS demographic data from Census API."""
        print(f"Fetching ACS {self.acs_year} demographic data...")
        
        api_key = self.config['census_api_key']
        state_fips = self.config['state_fips']
        county_fips_list = self.config['county_fips']
        
        if api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Please set your Census API key in the config!")
        
        all_data = []
        headers = None
        
        # Fetch data for each county
        for i, county_fips in enumerate(county_fips_list, 1):
            print(f"  Fetching county {i}/{len(county_fips_list)} (FIPS: {county_fips})...")
            
            url = f"https://api.census.gov/data/{self.acs_year}/acs/acs5"
            var_string = ",".join(RAW_CENSUS_VARIABLES.keys())
            
            params = {
                "get": f"NAME,{var_string}",
                "for": "block group:*",
                "in": f"state:{state_fips} county:{county_fips}",
                "key": api_key
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    print(f"  Warning: API request failed for county {county_fips}")
                    continue
                
                data = response.json()
                if len(data) > 1:  # Has data beyond header
                    if headers is None:
                        headers = data[0]
                    all_data.extend(data[1:])
                
                time.sleep(0.5)  # Be nice to the API
                
            except Exception as e:
                print(f"  Warning: Could not fetch data for county {county_fips}: {e}")
                continue
        
        if not all_data:
            raise Exception("No ACS data retrieved")
        
        # Create dataframe
        self.demographics_df = pd.DataFrame(all_data, columns=headers)
        self.demographics_df['GEOID'] = (
            self.demographics_df['state'] + 
            self.demographics_df['county'] + 
            self.demographics_df['tract'] + 
            self.demographics_df['block group']
        )
        
        # Convert to numeric
        for var in RAW_CENSUS_VARIABLES.keys():
            self.demographics_df[var] = pd.to_numeric(
                self.demographics_df[var], errors='coerce'
            )
        
        print(f"  Retrieved data for {len(self.demographics_df)} block groups")
    
    def calculate_demographics(self):
        """Calculate demographic percentages and derived metrics."""
        print("Calculating demographic metrics...")
        
        df = self.demographics_df
        
        # Non-car commuters percentage
        total_commuters = df['B08301_001E'].replace(0, pd.NA)
        non_car = df['B08301_010E'] + df['B08301_018E'] + df['B08301_019E']
        df['pct_non_car_commute'] = (non_car / total_commuters * 100).round(2)
        
        # MOE for non-car commute (simplified - sum of squared MOEs)
        moe_squared = (
            df['B08301_010M']**2 + 
            df['B08301_018M']**2 + 
            df['B08301_019M']**2
        )
        df['pct_non_car_commute_moe'] = (
            (moe_squared**0.5) / total_commuters * 100
        ).round(2)
        
        # Black and Latino percentage
        total_pop = df['B03002_001E'].replace(0, pd.NA)
        black_latino = df['B03002_004E'] + df['B03002_012E']
        df['pct_black_latino'] = (black_latino / total_pop * 100).round(2)
        
        # MOE for black and Latino
        moe_squared = df['B03002_004M']**2 + df['B03002_012M']**2
        df['pct_black_latino_moe'] = (
            (moe_squared**0.5) / total_pop * 100
        ).round(2)
        
        # Median household income (already in correct format)
        df['median_hh_income'] = df['B19013_001E']
        df['median_hh_income_moe'] = df['B19013_001M']
        
        print("  ✓ Calculated demographic metrics")
    
    # ------------------------------------------------------------------
    # Spatial Join
    # ------------------------------------------------------------------
    
    def extract_block_group_geoid(self):
        """Extract block group GEOID from block GEOID."""
        # Block GEOID format: SSCCCTTTTTTBBBB (15 digits)
        # Block Group GEOID: SSCCCTTTTTTB (12 digits)
        
        geoid_field = None
        for field in ['GEOID20', 'GEOID10', 'GEOID', 'BLOCKID']:
            if field in self.blocks_gdf.columns:
                geoid_field = field
                break
        
        if not geoid_field:
            raise ValueError("No GEOID field found in blocks shapefile")
        
        self.blocks_gdf['block_group_geoid'] = (
            self.blocks_gdf[geoid_field].astype(str).str[:12]
        )
    
    def merge_blocks_and_demographics(self):
        """Merge block geometries with block group demographic data."""
        print("Merging demographic data with census blocks...")
        
        # Extract block group GEOID
        self.extract_block_group_geoid()
        
        # Select demographic columns to merge
        demo_cols = [
            'GEOID', 
            'pct_non_car_commute', 'pct_non_car_commute_moe',
            'median_hh_income', 'median_hh_income_moe',
            'pct_black_latino', 'pct_black_latino_moe'
        ]
        
        # Merge
        self.blocks_gdf = self.blocks_gdf.merge(
            self.demographics_df[demo_cols],
            left_on='block_group_geoid',
            right_on='GEOID',
            how='left'
        )
        
        # Drop redundant GEOID column
        if 'GEOID' in self.blocks_gdf.columns:
            self.blocks_gdf = self.blocks_gdf.drop(columns=['GEOID'])
        
        print(f"  ✓ Merged demographics for {len(self.blocks_gdf)} blocks")
    
    def spatial_join_parcels_to_blocks(self):
        """Spatially join census block data to parcels."""
        print("Performing spatial join of parcels to census blocks...")
        
        # Ensure same CRS
        if self.parcels_gdf.crs != self.blocks_gdf.crs:
            self.blocks_gdf = self.blocks_gdf.to_crs(self.parcels_gdf.crs)
        
        # Get census columns to keep
        census_cols = [
            'pct_non_car_commute', 'pct_non_car_commute_moe',
            'median_hh_income', 'median_hh_income_moe',
            'pct_black_latino', 'pct_black_latino_moe'
        ]
        
        # Keep only necessary columns from blocks
        blocks_subset = self.blocks_gdf[census_cols + ['geometry']].copy()
        
        # Detect parcel geometry type
        geom_types = self.parcels_gdf.geometry.geom_type.unique()
        is_point_data = all(gt in ['Point', 'MultiPoint'] for gt in geom_types)
        
        if is_point_data:
            print("  Using 'within' predicate for point parcels...")
            joined = gpd.sjoin(
                self.parcels_gdf,
                blocks_subset,
                how='left',
                predicate='within'
            )
        else:
            print("  Using centroid method for polygon parcels...")
            parcels_work = self.parcels_gdf.copy()
            parcels_work['_original_geometry'] = parcels_work.geometry
            parcels_work.geometry = parcels_work.geometry.centroid
            
            joined = gpd.sjoin(
                parcels_work,
                blocks_subset,
                how='left',
                predicate='within'
            )
            
            joined.geometry = joined['_original_geometry']
            joined = joined.drop(columns=['_original_geometry'])
        
        # Clean up join artifacts
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
        
        # Handle duplicates
        original_len = len(joined)
        joined = joined[~joined.index.duplicated(keep='first')]
        if len(joined) < original_len:
            print(f"  Removed {original_len - len(joined)} duplicate matches")
        
        self.result_gdf = joined.reset_index(drop=True)
        
        # Report statistics
        total = len(self.result_gdf)
        matched = self.result_gdf['pct_non_car_commute'].notna().sum()
        unmatched = total - matched
        
        print(f"  ✓ Joined {matched:,} parcels ({matched/total*100:.1f}%)")
        if unmatched > 0:
            print(f"  Warning: {unmatched:,} parcels ({unmatched/total*100:.1f}%) "
                  f"did not match any census block")
    
    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    
    def save_output(self):
        """Save the final result to both GeoJSON and JSON formats."""
        # Use Exports/Data/{city}/ structure
        exports_dir = Path(self.config.get('exports_output_dir', 'Exports'))
        data_dir = exports_dir / "Data" / self.city_name.replace(' ', '_').replace(',', '')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as GeoJSON
        geojson_filename = f"{self.city_name.lower().replace(' ', '_').replace(',', '')}_parcels_with_demographics.geojson"
        geojson_path = city_folder / geojson_filename
        
        print(f"Saving output to {city_folder}...")
        self.result_gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"  ✓ Saved GeoJSON: {geojson_filename}")
        
        # Also save as regular JSON (without geometry, for data analysis)
        json_filename = f"{self.city_name.lower().replace(' ', '_').replace(',', '')}_demographics_data.json"
        json_path = city_folder / json_filename
        
        # Drop geometry column for JSON export
        data_df = pd.DataFrame(self.result_gdf.drop(columns='geometry'))
        data_df.to_json(json_path, orient='records', indent=2)
        print(f"  ✓ Saved JSON: {json_filename}")
        
        print(f"  Total: {len(self.result_gdf):,} parcels with {len(self.result_gdf.columns)} columns")
    
    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    
    def full_analysis(self):
        """Run the complete census analysis pipeline."""
        print(f"\n{'='*60}")
        print(f"Census Analysis: {self.city_name}")
        print(f"{'='*60}\n")
        
        try:
            # Load data
            self.load_census_blocks()
            self.load_parcels()
            
            # Fetch and process census data
            self.fetch_acs_demographics()
            self.calculate_demographics()
            
            # Merge and join
            self.merge_blocks_and_demographics()
            self.spatial_join_parcels_to_blocks()
            
            # Save output
            self.save_output()
            
            print(f"\n✓ Census analysis complete for {self.city_name}")
            return self.result_gdf
            
        except Exception as e:
            print(f"\n✗ Error processing {self.city_name}: {e}")
            import traceback
            traceback.print_exc()
            return None


# ======================================================================
# Multi-City Analysis
# ======================================================================

def multi_city_census_analysis(cities_config, global_config=None):
    """
    Run census analysis for multiple cities with consolidated global settings.
    
    Parameters
    ----------
    cities_config : dict
        Dictionary with city-specific configurations:
        {
            'CityName': {
                'state_fips': '25',
                'county_fips': ['025'],
                'block_shapefile': 'path/to/blocks.shp',
                'boundary_geojson': 'path/to/boundary.geojson',
                'parcels_path': 'path/to/parcels.geojson',
                # Optional overrides for global settings:
                'census_api_key': 'city_specific_key',
                'acs_year': 2020,
                'n_jobs': 2
            },
            ...
        }
    global_config : dict, optional
        Global configuration applied to all cities:
        {
            'census_api_key': 'your_key',
            'acs_year': 2022,
            'n_jobs': 4,
            'output_dir': 'Data/Processed/Demographics'
        }
    
    Returns
    -------
    dict : Results for each city
    """
    if global_config is None:
        global_config = {}
    
    # Extract global settings with defaults
    global_api_key = global_config.get('census_api_key', 'YOUR_API_KEY_HERE')
    global_acs_year = global_config.get('acs_year', 2022)
    global_n_jobs = global_config.get('n_jobs', max(1, cpu_count() - 1))
    global_exports_dir = global_config.get('exports_output_dir', 'Exports')
    
    results = {}
    
    for city_name, city_config in cities_config.items():
        # Merge global and city-specific configs (city-specific takes precedence)
        merged_config = {
            'census_api_key': city_config.get('census_api_key', global_api_key),
            'acs_year': city_config.get('acs_year', global_acs_year),
            'n_jobs': city_config.get('n_jobs', global_n_jobs),
            'exports_output_dir': global_exports_dir,
            **city_config  # Include all city-specific settings
        }
        
        analyzer = CensusAnalyzer(city_name, merged_config)
        results[city_name] = analyzer.full_analysis()
    
    # Print summary
    print(f"\n{'='*60}")
    print("CENSUS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for city_name, result in results.items():
        if result is not None:
            print(f"  ✓ {city_name}: {len(result):,} parcels")
        else:
            print(f"  ✗ {city_name}: FAILED")
    
    return results


# ======================================================================
# Example Configuration
# ======================================================================

if __name__ == "__main__":
    
    # Global configuration (applied to all cities)
    global_config = {
        'census_api_key': 'YOUR_API_KEY_HERE',  # Get free key at: https://api.census.gov/data/key_signup.html
        'acs_year': 2022,
        'n_jobs': 4,
        'exports_output_dir': 'Exports'
    }
    
    # City-specific configuration
    cities_config = {
        'Boston': {
            'state_fips': '25',
            'county_fips': ['025'],  # Suffolk County
            'block_shapefile': 'Data/Raw/Boston/Massachusets Census Blocks.shp/tl_2025_25_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/Boston/bostoncitylimits.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/boston_parcels_parkximity.geojson',
        },
        'NYC': {
            'state_fips': '36',
            'county_fips': ['005', '047', '061', '081', '085'],  # 5 boroughs
            'block_shapefile': 'Data/Raw/NYC/New York Census Blocks.shp/tl_2025_36_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/NYC/NY_County_FeaturesToJSON.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/nyc_parcels_parkximity.geojson',
        },
        'LA': {
            'state_fips': '06',
            'county_fips': ['037'],  # Los Angeles County
            'block_shapefile': 'Data/Raw/LA/California Census Blocks.shp/tl_2025_06_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/LA/City_Boundary.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/la_parcels_parkximity.geojson',
        },
        'San Francisco': {
            'state_fips': '06',
            'county_fips': ['075'],  # San Francisco County
            'block_shapefile': 'Data/Raw/SF/California Census Blocks.shp/tl_2025_06_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/SF/SF_Boundary.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/sf_parcels_parkximity.geojson',
        },
        'Indianapolis': {
            'state_fips': '18',
            'county_fips': ['097'],  # Marion County
            'block_shapefile': 'Data/Raw/Indianapolis/Indiana Census Blocks.shp/tl_2025_18_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/Indianapolis/Township_Boundaries.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/indianapolis_parcels_parkximity.geojson',
        },
        'Houston': {
            'state_fips': '48',
            'county_fips': ['201'],  # Harris County
            'block_shapefile': 'Data/Raw/Houston/Texas Census Blocks.shp/tl_2025_48_tabblock20.shp',
            'boundary_geojson': 'Data/Raw/Houston/houstoncitylimits.geojson',
            'parcels_path': 'Data/Processed/ParkXimity/houston_parcels_parkximity.geojson',
        },
    }
    
    # Run analysis for all cities
    results = multi_city_census_analysis(cities_config, global_config)
