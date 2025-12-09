import geopandas as gpd
import pandas as pd
from pathlib import Path
import requests
import time

# Configuration
CONFIG = {
    "census_api_key": "8d34d0028813e9b71b4c5110845270e08505639b",
    "output_dir": "Data/Processed/Census",
    "acs_year": 2022,
    "cities": [
        {
            "name": "San Francisco city, California",
            "state_fips": "06",
            "county_fips": ["075"],  # San Francisco County
            "block_shapefile": "Data/Raw/California Census Blocks.shp/tl_2025_06_tabblock20.shp",
            "boundary_geojson": "Data/Raw/SF/SF_Boundary.geojson"
        },
        {
            "name": "Los Angeles, California",
            "state_fips": "06",
            "county_fips": ["037"],  # Los Angeles County
            "block_shapefile": "Data/Raw/California Census Blocks.shp/tl_2025_06_tabblock20.shp",
            "boundary_geojson": "Data/Raw/LA/City_Boundary.geojson"
        },
        {
            "name": "New York City, New York",
            "state_fips": "36",
            "county_fips": ["005", "047", "061", "081", "085"],  # Bronx, Kings (Brooklyn), New York (Manhattan), Queens, Richmond (Staten Island)
            "block_shapefile": "Data/Raw/New York Census Blocks.shp/tl_2025_36_tabblock20.shp",
            "boundary_geojson": "Data/Raw/NYC/NY_County_FeaturesToJSON.geojson"
        },
        {
            "name": "Boston, Massachusetts",
            "state_fips": "25",
            "county_fips": ["025"],  # Suffolk County
            "block_shapefile": "Data/Raw/Massachusets Census Blocks.shp/tl_2025_25_tabblock20.shp",
            "boundary_geojson": "Data/Raw/Boston/bostoncitylimits.geojson"
        },
        {
            "name": "Indianapolis, Indiana",
            "state_fips": "18",
            "county_fips": ["097"],  # Marion County
            "block_shapefile": "Data/Raw/Indiana Census Blocks.shp/tl_2025_18_tabblock20.shp",
            "boundary_geojson": "Data/Raw/Indianapolis/Township_Boundaries.geojson"
        },
        {
            "name": "Houston, Texas",
            "state_fips": "48",
            "county_fips": ["201"],  # Harris County
            "block_shapefile": "Data/Raw/Texas Census Blocks.shp/tl_2025_48_tabblock20.shp",
            "boundary_geojson": "Data/Raw/Houston/houstoncitylimits.geojson"
        }
    ]
}

# ACS variables to fetch
VARIABLES = [
    "B08301_001E", "B08301_001M",  # Total commuters
    "B08301_010E", "B08301_010M",  # Public transportation
    "B08301_018E", "B08301_018M",  # Bicycle
    "B08301_019E", "B08301_019M",  # Walked
    "B19013_001E", "B19013_001M",  # Median household income
    "B03002_001E", "B03002_001M",  # Total population
    "B03002_004E", "B03002_004M",  # Black or African American alone
    "B03002_012E", "B03002_012M",  # Hispanic or Latino
]


def load_state_blocks(block_shapefile_path):
    """Load census block shapefile"""
    print(f"  Loading block shapefile from {block_shapefile_path}...")
    
    shapefile_path = Path(block_shapefile_path)
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Block shapefile not found: {shapefile_path}")
    
    gdf = gpd.read_file(shapefile_path)
    print(f"  Loaded {len(gdf)} blocks from {shapefile_path.name}")
    
    return gdf


def load_city_boundary(boundary_geojson_path):
    """Load city boundary GeoJSON"""
    print(f"  Loading city boundary from {boundary_geojson_path}...")
    
    boundary_path = Path(boundary_geojson_path)
    
    if not boundary_path.exists():
        raise FileNotFoundError(f"City boundary file not found: {boundary_path}")
    
    gdf = gpd.read_file(boundary_path)
    print(f"  Loaded city boundary from {boundary_path.name}")
    
    return gdf


def clip_blocks_to_city(blocks_gdf, city_boundary_gdf):
    """Clip census blocks to city boundary"""
    print(f"  Clipping {len(blocks_gdf)} blocks to city boundary...")
    
    # Ensure both are in the same CRS
    print(f"  Checking CRS... blocks: {blocks_gdf.crs}, boundary: {city_boundary_gdf.crs}")
    if blocks_gdf.crs != city_boundary_gdf.crs:
        print(f"  Reprojecting boundary to match blocks CRS...")
        city_boundary_gdf = city_boundary_gdf.to_crs(blocks_gdf.crs)
    
    # First, do a quick bounding box filter to reduce candidates
    print(f"  Filtering blocks by bounding box...")
    bounds = city_boundary_gdf.total_bounds
    blocks_subset = blocks_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    print(f"  Reduced to {len(blocks_subset)} candidate blocks within bounding box")
    
    # Now do the actual clip on the reduced dataset
    print(f"  Performing spatial clip (this may take a while)...")
    clipped = gpd.clip(blocks_subset, city_boundary_gdf)
    
    print(f"  ✓ {len(clipped)} blocks within city boundary")
    
    return clipped


def make_census_request(url, params):
    """Make a Census API request with error handling"""
    response = requests.get(url, params=params)
    if response.status_code != 200:
        error_msg = f"API request failed: {response.status_code}"
        try:
            error_detail = response.json()
            error_msg += f"\nAPI Error: {error_detail}"
        except:
            error_msg += f"\nResponse: {response.text[:500]}"
        raise Exception(error_msg)
    return response.json()


def fetch_acs_data(state_fips, county_fips_list, api_key, year):
    """Fetch ACS demographic data for block groups in specified counties"""
    print(f"  Fetching demographic data from ACS {year}...")
    print(f"  Counties: {', '.join(county_fips_list)}")
    
    all_data = []
    
    # Fetch data for each specified county
    for i, county_fips in enumerate(county_fips_list, 1):
        print(f"  Fetching county {i}/{len(county_fips_list)} (FIPS: {county_fips})...")
        
        url = f"https://api.census.gov/data/{year}/acs/acs5"
        var_string = ",".join(VARIABLES)
        
        params = {
            "get": f"NAME,{var_string}",
            "for": "block group:*",
            "in": f"state:{state_fips} county:{county_fips}",
            "key": api_key
        }
        
        try:
            data = make_census_request(url, params)
            if len(data) > 1:  # Has data beyond header
                # First iteration, save headers
                if not all_data:
                    headers = data[0]
                all_data.extend(data[1:])
        except Exception as e:
            print(f"  Warning: Could not fetch data for county {county_fips}: {e}")
            continue
    
    # Create dataframe
    if not all_data:
        raise Exception("No ACS data retrieved")
    
    df = pd.DataFrame(all_data, columns=headers)
    df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
    
    # Convert to numeric
    for var in VARIABLES:
        df[var] = pd.to_numeric(df[var], errors='coerce')
    
    print(f"  Retrieved data for {len(df)} block groups")
    
    return df


def calculate_percentage_with_moe(numerator_cols, numerator_moe_cols, denominator, denominator_moe, df):
    """Calculate percentage and MOE for sum of numerators over denominator"""
    numerator_sum = sum(df[col] for col in numerator_cols)
    numerator_moe = (sum(df[col]**2 for col in numerator_moe_cols)**0.5)
    
    percentage = (numerator_sum / df[denominator]) * 100
    percentage_moe = (numerator_moe / df[denominator]) * 100
    
    return percentage, percentage_moe


def calculate_demographics(df):
    """Calculate demographic percentages and MOEs"""
    # Non-car commuters percentage
    df['pct_non_car_commute'], df['pct_non_car_commute_moe'] = calculate_percentage_with_moe(
        ['B08301_010E', 'B08301_018E', 'B08301_019E'],
        ['B08301_010M', 'B08301_018M', 'B08301_019M'],
        'B08301_001E',
        'B08301_001M',
        df
    )
    
    # Black and Latino percentage
    df['pct_black_latino'], df['pct_black_latino_moe'] = calculate_percentage_with_moe(
        ['B03002_004E', 'B03002_012E'],
        ['B03002_004M', 'B03002_012M'],
        'B03002_001E',
        'B03002_001M',
        df
    )
    
    # Median household income
    df['median_hh_income'] = df['B19013_001E']
    df['median_hh_income_moe'] = df['B19013_001M']
    
    return df


def extract_block_group_geoid(gdf):
    """Extract block group GEOID from block GEOID"""
    # Block GEOID format: SSCCCTTTTTTBBBB (15 digits)
    # Block Group GEOID: SSCCCTTTTTTB (12 digits)
    
    # Try common field names for GEOID
    geoid_field = None
    for field in ['GEOID20', 'GEOID10', 'GEOID', 'BLOCKID']:
        if field in gdf.columns:
            geoid_field = field
            break
    
    if not geoid_field:
        raise ValueError("No GEOID field found in blocks shapefile")
    
    gdf['block_group_geoid'] = gdf[geoid_field].astype(str).str[:12]
    
    return gdf


def merge_blocks_and_demographics(blocks_gdf, demographics_df):
    """Merge block geometries with block group demographic data"""
    print(f"  Merging demographic data with blocks...")
    
    # Extract block group GEOID from blocks
    blocks_gdf = extract_block_group_geoid(blocks_gdf)
    
    # Merge
    result = blocks_gdf.merge(
        demographics_df[['GEOID', 'pct_non_car_commute', 'pct_non_car_commute_moe',
                        'median_hh_income', 'median_hh_income_moe',
                        'pct_black_latino', 'pct_black_latino_moe']],
        left_on='block_group_geoid',
        right_on='GEOID',
        how='left',
        suffixes=('', '_acs')
    )
    
    # Drop redundant GEOID column from merge
    if 'GEOID_acs' in result.columns:
        result = result.drop(columns=['GEOID_acs'])
    
    print(f"  Merged data for {len(result)} blocks")
    
    return result


def process_city(city_config, api_key, year, output_dir):
    """Process a single city and generate GeoJSON"""
    print(f"\nProcessing {city_config['name']}...")
    
    try:
        # Load state block shapefile
        blocks_gdf = load_state_blocks(city_config['block_shapefile'])
        
        # Load city boundary
        city_boundary_gdf = load_city_boundary(city_config['boundary_geojson'])
        
        # Clip blocks to city boundary
        clipped_blocks = clip_blocks_to_city(blocks_gdf, city_boundary_gdf)
        
        if len(clipped_blocks) == 0:
            print(f"  No blocks found within city boundary")
            return False
        
        # Fetch ACS demographic data for specified counties
        demographics_df = fetch_acs_data(
            city_config['state_fips'], 
            city_config['county_fips'], 
            api_key, 
            year
        )
        
        # Calculate demographics
        demographics_df = calculate_demographics(demographics_df)
        
        # Merge blocks with demographics
        result_gdf = merge_blocks_and_demographics(clipped_blocks, demographics_df)
        
        # Generate output filename from city name
        output_filename = city_config['name'].lower().replace(" ", "_").replace(",", "").replace(".", "") + ".geojson"
        output_path = Path(output_dir) / output_filename
        
        result_gdf.to_file(output_path, driver="GeoJSON")
        print(f"  ✓ Saved {len(result_gdf)} blocks to {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error processing {city_config['name']}: {e}")
        return False


def main():
    """Main execution function"""
    # Extract config values
    api_key = CONFIG["census_api_key"]
    output_dir = Path(CONFIG["output_dir"])
    year = CONFIG["acs_year"]
    cities = CONFIG["cities"]
    
    output_dir.mkdir(exist_ok=True)
    
    if api_key == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Census API key in the CONFIG dictionary!")
        print("Get a free key at: https://api.census.gov/data/key_signup.html")
        return
    
    for city_config in cities:
        process_city(city_config, api_key, year, output_dir)
        time.sleep(1)  # Be nice to the API
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()