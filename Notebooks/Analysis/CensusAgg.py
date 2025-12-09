"""
Census Block to Parcel Spatial Join Script

This script performs a spatial join to add census block demographic data 
to parcels that have parkximity calculations. Each parcel is assigned the 
demographic characteristics of the census block it falls within.

Input:
- Parcel GeoJSONs with parkximity values (park_distance, park_distance_km)
- Census block GeoJSONs with demographic variables

Output:
- Combined GeoJSONs with both parkximity and demographic data per parcel
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Census variables - supports both raw API names and pre-computed names
# The script will auto-detect which columns are present

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
    # Also check for individual race/ethnicity if present
    "pct_black",
    "pct_black_moe",
    "pct_hispanic",
    "pct_hispanic_moe",
    "pct_latino",
    "pct_latino_moe",
    # Population
    "total_population",
    "total_population_moe",
    # Commute modes
    "pct_transit",
    "pct_transit_moe",
    "pct_bicycle",
    "pct_bicycle_moe",
    "pct_walked",
    "pct_walked_moe",
]

# Columns to exclude from the join (metadata, not demographic data)
EXCLUDE_COLUMNS = [
    'STATEFP20', 'COUNTYFP20', 'TRACTCE20', 'BLOCKCE20', 
    'GEOID20', 'GEOIDFQ20', 'NAME20', 'MTFCC20', 'UR20', 
    'UACE20', 'FUNCSTAT20', 'ALAND20', 'AWATER20',
    'INTPTLAT20', 'INTPTLON20', 'HOUSING20', 'POP20',
    'block_group_geoid', 'GEOID', 'geometry',
    # Also exclude state/county FIPS variations
    'STATEFP', 'COUNTYFP', 'TRACTCE', 'BLOCKCE',
    'GEOID10', 'NAME10', 'NAME',
]


def load_and_validate_geojson(filepath, name):
    """Load a GeoJSON file and validate it has geometry."""
    print(f"  Loading {name}...")
    gdf = gpd.read_file(filepath)
    
    if gdf.empty:
        raise ValueError(f"{name} is empty!")
    
    if gdf.geometry.isna().all():
        raise ValueError(f"{name} has no valid geometries!")
    
    print(f"    Loaded {len(gdf)} features")
    return gdf


def standardize_crs(parcels_gdf, blocks_gdf):
    """Ensure both GeoDataFrames are in the same CRS."""
    # Use the parcels CRS as the target, or default to WGS84
    if parcels_gdf.crs is None:
        parcels_gdf = parcels_gdf.set_crs("EPSG:4326")
    
    if blocks_gdf.crs is None:
        blocks_gdf = blocks_gdf.set_crs("EPSG:4326")
    
    # Reproject blocks to match parcels if needed
    if parcels_gdf.crs != blocks_gdf.crs:
        print(f"    Reprojecting census blocks from {blocks_gdf.crs} to {parcels_gdf.crs}")
        blocks_gdf = blocks_gdf.to_crs(parcels_gdf.crs)
    
    return parcels_gdf, blocks_gdf


def rename_census_columns(blocks_gdf):
    """Rename raw census variable columns to human-readable names if present."""
    rename_map = {}
    for old_name, new_name in RAW_CENSUS_VARIABLES.items():
        if old_name in blocks_gdf.columns:
            rename_map[old_name] = new_name
    
    if rename_map:
        blocks_gdf = blocks_gdf.rename(columns=rename_map)
        print(f"    Renamed {len(rename_map)} raw census columns to human-readable names")
    
    return blocks_gdf


def get_census_columns(blocks_gdf):
    """
    Get list of census demographic columns to join.
    Auto-detects both raw API names (renamed) and pre-computed names.
    """
    census_cols = []
    
    # Check for renamed raw census columns
    for new_name in RAW_CENSUS_VARIABLES.values():
        if new_name in blocks_gdf.columns:
            census_cols.append(new_name)
    
    # Check for pre-computed columns
    for col_name in PRECOMPUTED_CENSUS_VARIABLES:
        if col_name in blocks_gdf.columns and col_name not in census_cols:
            census_cols.append(col_name)
    
    # If no specific columns found, grab all numeric columns except excluded ones
    if not census_cols:
        print("    No predefined census columns found, scanning for numeric columns...")
        for col in blocks_gdf.columns:
            if col not in EXCLUDE_COLUMNS and col != 'geometry':
                # Check if it's a numeric column
                if blocks_gdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    census_cols.append(col)
    
    return census_cols


def spatial_join_parcels_to_blocks(parcels_gdf, blocks_gdf):
    """
    Perform spatial join to assign census block data to parcels.
    
    Works with both point and polygon parcels.
    - Point parcels: Uses 'within' predicate directly
    - Polygon parcels: Uses centroid for the join
    
    Parameters:
    -----------
    parcels_gdf : GeoDataFrame
        Parcels with parkximity values (points or polygons)
    blocks_gdf : GeoDataFrame  
        Census blocks with demographic variables (polygons)
    
    Returns:
    --------
    GeoDataFrame with parcels containing both parkximity and demographic data
    """
    print(f"  Performing spatial join...")
    
    # Standardize CRS
    parcels_gdf, blocks_gdf = standardize_crs(parcels_gdf, blocks_gdf)
    
    # Detect parcel geometry type
    geom_types = parcels_gdf.geometry.geom_type.unique()
    print(f"    Parcel geometry types: {list(geom_types)}")
    
    # Print all columns in census blocks for debugging
    print(f"    Census block columns available: {list(blocks_gdf.columns)}")
    
    # Rename raw census columns if present
    blocks_gdf = rename_census_columns(blocks_gdf)
    
    # Get the census columns to keep
    census_cols = get_census_columns(blocks_gdf)
    
    if not census_cols:
        print("    WARNING: No census variables found in blocks data!")
        print("    Expected variables include:")
        for var in list(RAW_CENSUS_VARIABLES.keys())[:5]:
            print(f"      - {var}")
        for var in PRECOMPUTED_CENSUS_VARIABLES[:5]:
            print(f"      - {var}")
        return parcels_gdf
    
    print(f"    Found {len(census_cols)} census variables to join:")
    for col in census_cols:
        print(f"      - {col}")
    
    # Keep only necessary columns from blocks (census vars + geometry)
    blocks_subset = blocks_gdf[census_cols + ['geometry']].copy()
    
    # Check if parcels are points or polygons
    is_point_data = all(gt in ['Point', 'MultiPoint'] for gt in geom_types)
    
    if is_point_data:
        # Points can use 'within' directly
        print("    Using 'within' predicate for point parcels...")
        joined = gpd.sjoin(
            parcels_gdf,
            blocks_subset,
            how='left',
            predicate='within'
        )
    else:
        # For polygons, use centroid
        print("    Using centroid method for polygon parcels...")
        parcels_work = parcels_gdf.copy()
        parcels_work['_original_geometry'] = parcels_work.geometry
        parcels_work.geometry = parcels_work.geometry.centroid
        
        joined = gpd.sjoin(
            parcels_work,
            blocks_subset,
            how='left',
            predicate='within'
        )
        
        # Restore original geometry
        joined.geometry = joined['_original_geometry']
        joined = joined.drop(columns=['_original_geometry'])
    
    # Clean up join artifacts
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    # Handle duplicate rows from join (keep first match)
    original_len = len(joined)
    joined = joined[~joined.index.duplicated(keep='first')]
    if len(joined) < original_len:
        print(f"    Removed {original_len - len(joined)} duplicate matches")
    
    joined = joined.reset_index(drop=True)
    
    # Report join statistics
    total_parcels = len(joined)
    matched_parcels = joined[census_cols[0]].notna().sum()
    unmatched = total_parcels - matched_parcels
    
    print(f"    Joined {matched_parcels:,} parcels ({matched_parcels/total_parcels*100:.1f}%)")
    if unmatched > 0:
        print(f"    WARNING: {unmatched:,} parcels ({unmatched/total_parcels*100:.1f}%) did not match any census block")
    
    return joined


def calculate_derived_metrics(gdf):
    """
    Calculate useful derived demographic metrics.
    Skips calculations if pre-computed versions already exist.
    """
    print("  Calculating derived metrics (if not already present)...")
    
    # Only calculate commute mode shares if raw data exists and percentages don't
    if 'total_commuters' in gdf.columns and gdf['total_commuters'].notna().any():
        total = gdf['total_commuters'].replace(0, pd.NA)
        
        if 'public_transit' in gdf.columns and 'pct_transit' not in gdf.columns:
            gdf['pct_transit'] = (gdf['public_transit'] / total * 100).round(2)
        
        if 'bicycle' in gdf.columns and 'pct_bicycle' not in gdf.columns:
            gdf['pct_bicycle'] = (gdf['bicycle'] / total * 100).round(2)
        
        if 'walked' in gdf.columns and 'pct_walked' not in gdf.columns:
            gdf['pct_walked'] = (gdf['walked'] / total * 100).round(2)
        
        # Combined active/sustainable transport
        if all(col in gdf.columns for col in ['public_transit', 'bicycle', 'walked']):
            if 'pct_sustainable_commute' not in gdf.columns and 'pct_non_car_commute' not in gdf.columns:
                gdf['pct_sustainable_commute'] = (
                    (gdf['public_transit'] + gdf['bicycle'] + gdf['walked']) / total * 100
                ).round(2)
    
    # Only calculate race/ethnicity percentages if raw data exists and percentages don't
    if 'total_population' in gdf.columns and gdf['total_population'].notna().any():
        total_pop = gdf['total_population'].replace(0, pd.NA)
        
        if 'black_alone' in gdf.columns and 'pct_black' not in gdf.columns:
            gdf['pct_black'] = (gdf['black_alone'] / total_pop * 100).round(2)
        
        if 'hispanic_latino' in gdf.columns and 'pct_hispanic' not in gdf.columns:
            gdf['pct_hispanic'] = (gdf['hispanic_latino'] / total_pop * 100).round(2)
    
    return gdf


def process_city(city_name, parcels_path, blocks_path, output_dir):
    """
    Process a single city - join census data to parcels.
    
    Parameters:
    -----------
    city_name : str
        Name of the city
    parcels_path : str
        Path to parcel GeoJSON with parkximity values
    blocks_path : str
        Path to census blocks GeoJSON with demographic variables
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    GeoDataFrame with joined data
    """
    print(f"\n{'='*60}")
    print(f"Processing {city_name}")
    print(f"{'='*60}")
    
    # Load data
    parcels_gdf = load_and_validate_geojson(parcels_path, "Parcels")
    blocks_gdf = load_and_validate_geojson(blocks_path, "Census Blocks")
    
    # Perform spatial join
    joined_gdf = spatial_join_parcels_to_blocks(parcels_gdf, blocks_gdf)
    
    # Calculate derived metrics (only if not already present)
    joined_gdf = calculate_derived_metrics(joined_gdf)
    
    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{city_name.lower()}_parcels_with_demographics.geojson"
    
    print(f"  Saving to {output_path}...")
    joined_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"  ✓ Saved {len(joined_gdf):,} parcels with {len(joined_gdf.columns)} columns")
    
    # Print final column list
    print(f"  Final columns: {list(joined_gdf.columns)}")
    
    return joined_gdf
    
    # Perform spatial join
    joined_gdf = spatial_join_parcels_to_blocks(parcels_gdf, blocks_gdf, method=join_method)
    
    # Calculate derived metrics
    joined_gdf = calculate_derived_metrics(joined_gdf)
    
    # Save output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{city_name.lower()}_parcels_with_demographics.geojson"
    
    print(f"  Saving to {output_path}...")
    joined_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"  ✓ Saved {len(joined_gdf):,} parcels with {len(joined_gdf.columns)} columns")
    
    return joined_gdf


def process_all_cities(cities_config):
    """
    Process multiple cities.
    
    Parameters:
    -----------
    cities_config : dict
        Dictionary with city configurations:
        {
            'CityName': {
                'parcels_path': 'path/to/parcels.geojson',
                'blocks_path': 'path/to/census_blocks.geojson',
                'output_dir': 'path/to/output'
            },
            ...
        }
    
    Returns:
    --------
    dict : Results for each city
    """
    results = {}
    
    for city_name, config in cities_config.items():
        try:
            results[city_name] = process_city(
                city_name=city_name,
                parcels_path=config['parcels_path'],
                blocks_path=config['blocks_path'],
                output_dir=config['output_dir']
            )
        except Exception as e:
            print(f"  ERROR processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            results[city_name] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for city_name, result in results.items():
        if result is not None:
            print(f"  ✓ {city_name}: {len(result):,} parcels")
        else:
            print(f"  ✗ {city_name}: FAILED")
    
    return results


# =============================================================================
# CONFIGURATION - Update these paths for your data
# =============================================================================

if __name__ == "__main__":
    
    # Configure your cities here
    # Update paths to match your directory structure
    cities_config = {
        'Boston': {
            'parcels_path': 'Data/Processed/ParkXimity/boston_parcels_parkximity.geojson',
            'blocks_path': 'Data/Processed/Census/boston_massachusetts.geojson',
            'output_dir': 'Data/Processed/Demographics'
        },
        # 'Houston': {
        #     'parcels_path': 'Visualizations/Houston/houston_parcels_parkximity.geojson',
        #     'blocks_path': 'Data/Census/houston_census_blocks.geojson',
        #     'output_dir': 'Data/Processed/Demographics'
        # },
        'NYC': {
            'parcels_path': 'Data/Processed/ParkXimity/nyc_parcels_parkximity.geojson',
            'blocks_path': 'Data/Processed/Census/new_york_city_new_york.geojson',
            'output_dir': 'Data/Processed/Demographics'
        },
        'LA': {
            'parcels_path': 'Data/Processed/ParkXimity/la_parcels_parkximity.geojson',
            'blocks_path': 'Data/Processed/Census/los_angeles_california.geojson',
            'output_dir': 'Data/Processed/Demographics'
        },
        # 'SF': {
        #     'parcels_path': 'Visualizations/SF/sf_parcels_parkximity.geojson',
        #     'blocks_path': 'Data/Census/sf_census_blocks.geojson',
        #     'output_dir': 'Data/Processed/Demographics'
        # },
    }
    
     # Process all cities
    results = process_all_cities(cities_config)