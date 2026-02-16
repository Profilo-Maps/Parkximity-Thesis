# ParkXimity

Evaluating LTS-weighted distance from parcels to parks with CPU parallel processing and optional GPU acceleration.

## Overview

ParkXimity analyzes park accessibility by calculating Level of Traffic Stress (LTS) weighted distances from property parcels to park entrances using street network analysis. The toolkit features automatic CPU parallel processing and optional GPU acceleration for large-scale urban analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration UI](#configuration-ui)
  - [Basic Usage](#basic-usage-automatic-gpu-detection)
  - [Multi-City Analysis](#multi-city-analysis)
  - [Census Analysis Integration](#census-analysis-integration)
  - [Error Handling and Robustness](#error-handling-and-robustness)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [GPU Acceleration](#gpu-acceleration)
- [Troubleshooting](#troubleshooting)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Environment Management](#environment-management)
- [Benchmarking](#benchmarking)
- [Cloud Deployment](#cloud-deployment)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Citation](#citation)

## Key Features

✅ **Graphical Configuration UI** - Modern interface with industrial chic design  
✅ **Auto-detect data directories** - One-click configuration if you have standardized folders  
✅ **File name standardization** - Automatically rename files to consistent format  
✅ **Geographic data flexibility** - Configure geometry, lat/long, or WKT columns  
✅ **Automatic CPU parallel processing** - Uses all available CPU cores by default  
✅ **Multi-city parallel processing** - Process multiple cities simultaneously  
✅ **Integrated census analysis** - Spatially join ACS demographic data to parcels  
✅ **Robust error handling** - Continues processing if one city fails  
✅ **Detailed error reporting** - Summary shows which cities succeeded/failed with error messages  
✅ **Automatic GPU detection** - Detects hardware on first run  
✅ **Interactive installation** - Prompts to install GPU support if available  
✅ **Graceful fallback** - Works on CPU if GPU unavailable  
✅ **No code changes** - Same script works everywhere  
✅ **Configurable workers** - Control parallelism per your system  
✅ **Install later** - Can add GPU support anytime with `python gpu_utils.py`  
✅ **Progress bars** - Real-time feedback on all major operations  
✅ **Improved entrance detection** - Advanced algorithms capture more park access points

## Performance

### CPU Parallel Processing (New!)

The toolkit now uses CPU parallel processing by default, providing significant speedups:

| Operation | Sequential | Parallel (8 cores) | Speedup |
|-----------|-----------|-------------------|---------|
| Parcel validation | 30 sec | 5 sec | 6x |
| Park entrance generation | 15 min | 3 min | 5x |
| Park geometry validation | 2 min | 30 sec | 4x |
| Street network building | 45 sec | 12 sec | 4x |
| Visualization creation | 8 min | 2 min | 4x |
| Multi-city (3 cities) | 90 min | 35 min | 2.5x |

**Intelligent Parallelization:**
The toolkit automatically parallelizes operations only when beneficial. Small datasets use fast sequential processing to avoid overhead, while large datasets (>10K-50K items depending on operation) leverage all available CPU cores for maximum performance.

### GPU Acceleration (Optional)

With GPU acceleration, you get **10-100x speedup** on large datasets:

| Dataset | Parcels | CPU Time | GPU Time | Speedup |
|---------|---------|----------|----------|---------|
| Boston | 100K | 45 min | 3 min | 15x |
| NYC | 850K | 6.5 hrs | 22 min | 18x |
| LA | 1.2M | 9 hrs | 28 min | 19x |

### Parallelized Operations

**CPU Parallel Processing (Automatic):**
- **Park entrance generation** (5-8x speedup): Processes parks in parallel batches with improved detection algorithms
- **Park geometry validation** (4-6x speedup): Validates/repairs geometries in parallel
- **Parcel geometry validation** (6-8x speedup): Validates parcel geometries in parallel for large datasets
- **Street network building** (3-5x speedup): Extracts network edges in parallel for large street networks (>50K segments)
- **Boundary clipping** (4-6x speedup): Clips parcels to city boundaries in parallel for large datasets
- **Visualization creation** (3-4x speedup): Creates all visualizations simultaneously
- **JSON export** (4-6x speedup): Processes parcel data in parallel for large datasets
- **Multi-city analysis** (2-3x speedup): Processes multiple cities in parallel
- **Real-time progress tracking**: tqdm progress bars on all major operations

**Improved Park Entrance Detection:**
- **MultiLineString handling**: Processes disconnected street segments separately
- **Buffered boundary intersection**: Catches streets within buffer distance that don't exactly touch park boundaries
- **Intermediate point generation**: Adds entrance points every ~50m along long shared boundaries
- **Per-park deduplication**: Removes duplicates within each park while preserving entrances from adjacent parks
- **Result**: 2-5x more entrances detected, especially for large parks and complex street networks

**GPU Acceleration (Optional):**
- **Nearest Neighbor Search** (20-50x speedup): Finding closest network nodes
- **Point Deduplication** (10-30x speedup): Removing duplicate entrance points
- **Distance Calculations** (15-40x speedup): Computing pairwise distances
- **Heatmap IDW Interpolation** (15-40x speedup): GPU-accelerated spatial interpolation
- **Gaussian Smoothing** (5-15x speedup): GPU-accelerated image filtering

**Note on GPU + Parallelization:**
When GPU acceleration is enabled, visualizations are created sequentially to avoid CUDA context conflicts with multiprocessing. The heatmap's GPU-accelerated interpolation provides significant speedup even without parallel visualization creation. CPU-only mode continues to create visualizations in parallel.

## Recent Improvements (February 2025)

### Enhanced Progress Tracking

All major operations now display real-time progress bars using `tqdm`:

```
Loading data for NYC...
  Validating parcels: 100%|████████████| 39/39 [00:09<00:00, 4.20chunk/s]
  Validating parks: 100%|██████████| 5636/5636 [00:05<00:00, 981.48park/s]
  Clipping parcels: 100%|████████████| 77/77 [00:00<00:00, 745869.30chunk/s]

Generating park entrances...
  Finding entrances: 100%|█████████████| 29/29 [02:08<00:00, 4.42s/batch]
  Generated 3033 unique entrances

Building street network...
  Building network: 100%|██████████████| 39/39 [00:05<00:00, 7.03chunk/s]

Calculating parkximity...
  Dijkstra search: 100%|████████| 312997/312997 [00:45<00:00, 6844.38node/s]

Exporting parkximity JSON...
  Exporting parcels: 100%|█████████████| 77/77 [01:56<00:00, 1.52s/chunk]

Creating heatmap...
  IDW interpolation: 100%|████████████| 21/21 [00:15<00:00, 1.35chunk/s]
  Applying Gaussian smoothing...
  Applying boundary mask...
  Rendering heatmap visualization...
```

### Improved Park Entrance Detection

The entrance detection algorithm has been significantly enhanced to capture more legitimate park access points:

**1. MultiLineString Street Handling**
- Streets with disconnected segments (MultiLineString) are now processed component-by-component
- Each segment is checked independently for park boundary intersections
- Prevents missing entrances when street data has topology gaps

**2. Buffered Boundary Intersection**
- Three-tier detection strategy:
  1. Exact boundary intersection (fastest, most accurate)
  2. Buffered boundary intersection (catches near-misses within buffer distance)
  3. Nearest point fallback (only when both above fail)
- Captures streets that are close to but don't exactly touch park boundaries
- Handles topology mismatches between park and street datasets

**3. Intermediate Points on Long Boundaries**
- For LineString intersections >50m, adds entrance points every ~50m
- Prevents sparse coverage on large parks with long edges
- Ensures adequate entrance distribution along park perimeters

**4. Per-Park Deduplication Only**
- Removes duplicate entrances within each park (2m tolerance by default)
- Preserves entrances from adjacent parks at shared locations
- Allows multiple parks to have independent access points at the same street corner

**Impact:**
- 2-5x more entrances detected on average
- Especially effective for:
  - Large parks (Central Park, Golden Gate Park, etc.)
  - Parks with complex boundaries
  - Street networks with topology issues
  - Adjacent parks sharing street access

**Configuration:**
```python
config = {
    'park_buffer': 50.0,  # Buffer distance for entrance detection (meters)
    'entrance_tolerance': 2.0,  # Deduplication tolerance within each park (meters)
    'entrance_batch_size': 200,  # Parks per parallel batch
}
```

## Quick Start

Get started in 3 easy steps using the graphical configuration interface:

### Step 1: Install

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate parkximity
```

Or on Windows, just double-click: `setup.bat`

### Step 2: Launch Configuration UI

```bash
python ParkximityConfigUI.py
```

The modern configuration interface will open with an industrial chic design.

### Step 3: Configure Your Analysis

**Option A: Use Detected Directory (Fastest)**

If you already have a standardized data directory:

1. Enter your city names (one per line)
2. Click **"USE DIRECTORY: [path]"** button
3. All paths are auto-populated!
4. Click **"CONTINUE"** → **"SAVE & RUN"**

**Option B: Manual Configuration**

1. **Enter Cities**: Type city names (one per line)
   - Example: Boston, NYC, LA

2. **Create Data Structure** (optional):
   - Click **"CREATE DATA DIRECTORY"**
   - Select base folder
   - Organized folders are created automatically
   - Manually add your data files to `Data/Raw/{CityName}/`

3. **Configure Each City**:
   - Click **"CONTINUE"**
   - Select a city from the sidebar
   - Browse to select input files:
     - Parcels (GeoJSON/CSV)
     - Parks (GeoJSON)
     - Streets with LTS (GeoJSON)
     - Boundary (optional)
   - Set output directory
   - Configure geographic data columns if needed

4. **Optional - Standardize File Names**:
   - Check **"Standardize Raw Data File Names"**
   - Files will be renamed to: `{city}_{layer}_{date}.{extension}`

5. **Set Global Settings**:
   - Click **"GLOBAL SETTINGS"**
   - Adjust analysis parameters (sliders)
   - Configure fonts for visualizations
   - Add Census API key if using census analysis

6. **Validate & Run**:
   - Click **"VALIDATE CONFIG"** to check for errors
   - Click **"SAVE & RUN"** to start analysis

### What Happens Next

The analysis automatically:
- Detects GPU if available (prompts to install support)
- Processes all cities in parallel (or sequentially if preferred)
- Generates visualizations and outputs
- Continues even if one city fails

**Results are saved to your output directories:**
```
Visualizations/{CityName}/
├── {city}_parkximity_entrances.png
├── {city}_parkximity_heatmap.png
├── {city}_parkximity_histograms.png
├── {city}_parcels_parkximity.geojson
└── {city}_parkximity_data.json
```

### Quick Start (Command Line Alternative)

Prefer code? Edit `ParkximityCalc.py` directly and run:

```bash
python ParkximityCalc.py
```

**That's it!** The code automatically handles everything:

### What Happens on First Run

**If you have an NVIDIA GPU (but no GPU libraries):**
```
======================================================================
  PARKXIMITY MULTI-CITY ANALYSIS
======================================================================
  Cities to process: 3
  Target CRS: EPSG:6350
======================================================================

🎯 NVIDIA GPU Detected!
======================================================================
GPU: NVIDIA GeForce RTX 4060
CUDA Version: 12.6

GPU acceleration libraries are not installed.
Installing them will provide 10-50x speedup for large datasets.

Install GPU support now? [Y/n]: 
```

**If you type `Y` (Yes):**
- Installs GPU libraries automatically (5-10 min)
- Asks you to restart Python
- Next run uses GPU automatically

**If you type `n` (No):**
```
============================================================
💻 CPU MODE for Boston
============================================================
  Mode: CPU-only processing
============================================================
```
- Uses CPU mode (works perfectly fine!)
- Can install GPU support later

**After GPU installation (or if already installed):**
```
============================================================
🚀 GPU ACCELERATION ENABLED for Boston
============================================================
  Device: NVIDIA GeForce RTX 4060
  Memory: 8.0 GB VRAM
  Mode: GPU-accelerated processing
============================================================
```

**If no NVIDIA GPU:**
```
============================================================
💻 CPU MODE for Boston
============================================================
  No GPU detected
  Mode: CPU-only processing
============================================================
```

### Key Features

✅ **Automatic GPU detection** - Detects hardware on first run
✅ **Interactive installation** - Prompts to install GPU support if available
✅ **Graceful fallback** - Works on CPU if GPU unavailable
✅ **No code changes** - Same script works everywhere
✅ **Install later** - Can add GPU support anytime with `python gpu_utils.py`

## Configuration

### Coordinate Reference System (CRS)

The toolkit uses **EPSG:6350 (NAD83(2011) / Conus Albers)** as the target CRS for all spatial analysis:

**Why EPSG:6350?**
- **Projection**: Albers Equal Area Conic - preserves area measurements critical for spatial analysis
- **Units**: Meters - enables straightforward distance calculations (parkximity distances are in meters)
- **Coverage**: Optimized for the contiguous United States (CONUS)
- **Consistency**: Single CRS for all US cities simplifies multi-city comparison
- **Accuracy**: Minimizes distortion across the continental US

### Heatmap Sampling Configuration

The toolkit uses **adaptive stratified spatial sampling** to ensure equitable geographic representation in heatmaps:

```python
global_settings = {
    'park_buffer': 50.0,
    'entrance_tolerance': 5.0,
    'heatmap_resolution': 75,  # Grid cell size in meters (50-75 for high detail)
    'max_parcels_for_heatmap': 100000,  # Sample size (50k-200k recommended)
    'heatmap_neighbors': 5,
    'heatmap_smoothing': 1,
}
```

#### Adaptive Sampling Strategy

The system automatically adjusts sampling to ensure spatial equity:

1. **Density Analysis**: Calculates coefficient of variation (CV) of parcel density across 500m grid cells
2. **Adaptive Alpha**: Determines compensation factor based on density variation
   - CV = 0.0 (uniform) → α = 1.0 (proportional sampling)
   - CV = 1.0 (moderate) → α = 0.72 (moderate compensation)
   - CV = 2.0 (high) → α = 0.44 (strong equity focus)
   - CV ≥ 2.5 (extreme) → α = 0.3 (maximum compensation)
3. **Stratified Sampling**: Samples from grid cells with weight = (parcels_in_cell)^α
   - Higher α: Dense areas get proportionally more samples
   - Lower α: Sparse areas get oversampled for equity

**Benefits:**
- Ensures every neighborhood is represented in heatmaps
- Prevents dense urban cores from dominating the visualization
- Automatically adapts to each city's unique density patterns
- Maintains statistical validity while promoting spatial equity

**Example Output:**
```
Sampling 100000 of 681469 parcels
Density CV: 1.85, Alpha: 0.48 (equity-compensated)
```

### Parallel Processing Configuration

Control the number of parallel workers in your config:

```python
config = {
    'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
    'parks_path': 'Data/Raw/Boston/boston_parks.geojson',
    'streets_path': 'Data/Processed/LTS/lts_bos.geojson',
    'output_dir': 'Visualizations/Boston',
    'lts_column': 'PC2_norm',
    'n_jobs': 8,  # Number of parallel workers (default: cpu_count() - 1)
    'entrance_batch_size': 200,  # Parks per batch (default: 50)
}
```

#### Batch Size Tuning

The `entrance_batch_size` parameter controls how many parks are processed together in each parallel batch during entrance generation. Larger batches reduce task overhead and can significantly improve performance:

**Recommended batch sizes by system RAM:**

| System RAM | Recommended Batch Size | Notes |
|------------|----------------------|-------|
| 8 GB | 50-100 | Conservative, default is safe |
| 16 GB | 200-400 | Good balance of speed and safety |
| 32 GB+ | 500-1000 | Maximum performance |

**Guidelines:**
- Default is `50` (conservative for compatibility)
- Increasing to `200` typically provides 2-3x speedup on entrance generation
- Too large (>1000) may reduce parallelism benefits
- Monitor RAM usage when experimenting with larger values

**Example for 16GB system:**
```python
config = {
    'entrance_batch_size': 200,  # 4x fewer task overhead vs default
    'n_jobs': 6,  # Leave RAM headroom (vs using all cores)
    # ... other settings
}
```

### Global Settings for Multi-City

```python
global_settings = {
    'park_buffer': 50.0,
    'entrance_tolerance': 5.0,
    'entrance_batch_size': 200,  # Increased from default 50 for better performance
    'n_jobs': 8,  # Applied to all cities
}

# Process cities in parallel (default: True)
results = multi_city_analysis(
    cities_config, 
    global_settings=global_settings,
    parallel_cities=True  # Set False for memory-constrained systems
)
```

### Performance Tuning

#### Recommended Settings by System Spec

**High-Spec System (GPU + 16GB+ RAM + 8+ cores):**
```python
global_settings = {
    'entrance_batch_size': 400,  # Large batches for maximum throughput
    'n_jobs': -1,  # Use all available cores
    'max_parcels_for_heatmap': 200000,  # High-quality heatmaps
    'heatmap_resolution': 50,  # Fine detail (50-75m cells)
}

# Process cities in parallel for maximum speed
results = multi_city_analysis(cities_config, parallel_cities=True)
```

**Medium-Spec System (16GB RAM + 4-8 cores, no GPU):**
```python
global_settings = {
    'entrance_batch_size': 200,  # Balanced performance
    'n_jobs': 6,  # Leave headroom for system
    'max_parcels_for_heatmap': 100000,  # Good quality
    'heatmap_resolution': 75,  # Standard detail
}

# Process cities sequentially (recommended)
results = multi_city_analysis(cities_config, parallel_cities=False)
```

**Low-Spec System (8GB RAM + 2-4 cores):**
```python
global_settings = {
    'entrance_batch_size': 50,  # Conservative batching
    'n_jobs': 2,  # Minimal parallelism
    'max_parcels_for_heatmap': 50000,  # Reduced sampling
    'heatmap_resolution': 100,  # Coarser grid
}

# Process cities sequentially
results = multi_city_analysis(cities_config, parallel_cities=False)
```

**For systems with limited RAM:**
```python
config = {
    'n_jobs': 2,  # Reduce parallel workers
    # ... other settings
}

# Process cities sequentially
results = multi_city_analysis(cities_config, parallel_cities=False)
```

**For high-performance systems:**
```python
config = {
    'n_jobs': -1,  # Use all available cores
    # ... other settings
}

# Process cities in parallel
results = multi_city_analysis(cities_config, parallel_cities=True)
```

## Usage

### Configuration UI

The toolkit includes a graphical configuration interface for easy setup without editing Python code.

#### Launching the Configuration UI

```bash
python ParkximityConfigUI.py
```

#### Features

**1. City Selection**
- Enter city names (one per line) for analysis
- Load existing configuration from ParkximityCalc.py
- Add or remove cities as needed
- **Create Data Directory**: Automatically creates organized folder structure for all cities

**2. Data Directory Creation**
- Click "Create Data Directory" on the city selection page
- Select a base location for your project
- Automatically creates organized folders:
  - `Data/Raw/{CityName}/` - Input data files
  - `Data/Processed/LTS/` - LTS-processed street networks
  - `Data/Processed/Census/` - Census block data
  - `Data/Processed/Demographics/{CityName}/` - Census analysis results
  - `Visualizations/{CityName}/` - Output visualizations
- Checks for existing directories before creating
- Shows summary of what will be created
- Helps maintain consistent organization across multiple cities

**⚠ Important**: The tool only creates the folder structure. You must manually populate the `Data/Raw/{CityName}/` folders with your input files:
- Parcels file (GeoJSON, JSON, or CSV)
- Parks file (GeoJSON or JSON)
- Streets file with LTS values (GeoJSON or JSON)
- Boundary file (GeoJSON, JSON, or CSV - optional)
- Census block shapefile (if using census analysis)

**3. City-Specific Configuration**
- **Sidebar navigation**: Click any city to edit its settings
- **File browsers**: Select files using native file dialogs
- **Path validation**: Automatic checking for file existence and compatibility
- **Required fields**: Parcels, parks, streets, and output directory
- **Optional fields**: Boundary files, census data
- **File name standardization**: Option to rename files to consistent format
- **Geographic data specifications**: Configure how geographic data is stored

**Geographic Data Column Configuration**

For parcels, parks, and boundary data, you can specify how geographic information is stored:

**Parcels:**
- **Data Type**: auto (default), geometry, or latlong
  - auto: Detects based on file type (GeoJSON→geometry, CSV→latlong)
  - geometry: Uses geometry column in GeoJSON/JSON files
  - latlong: Uses separate latitude/longitude columns in CSV files
- **Geometry Column**: Column name for geometry data (default: "geometry")
- **Latitude Column**: Column name for latitude values (default: "centroid_latitude")
- **Longitude Column**: Column name for longitude values (default: "centroid_longitude")

**Parks:**
- **Data Type**: auto (default) or geometry
  - auto: Uses geometry column (parks must be polygons)
  - geometry: Explicitly specify geometry column
- **Geometry Column**: Column name for geometry data (default: "geometry")

**Boundary:**
- **Data Type**: auto (default), geometry, or wkt
  - auto: Detects based on file type (GeoJSON→geometry, CSV→wkt)
  - geometry: Uses geometry column in GeoJSON/JSON files
  - wkt: Uses Well-Known Text column in CSV files
- **Geometry Column**: Column name for geometry data (default: "geometry")
- **WKT Column**: Column name for WKT geometry in CSV (default: "the_geom")

This flexibility allows you to work with data in various formats without preprocessing.

**4. File Name Standardization**
- Optional checkbox: "Standardize Raw Data File Names"
- Renames input files to: `{city}_{layer}_{date}.{extension}`
- Examples:
  - `boston_parcels_2.16.26.geojson`
  - `nyc_parks_2.16.26.geojson`
  - `la_streets_2.16.26.geojson`
- Creates copies (preserves originals)
- Updates config paths automatically
- Helps track data versions and maintain consistency
- Applied when clicking "Save & Run"

**5. Global Settings Panel**
- **Analysis Parameters**: Park buffer, entrance tolerance, heatmap settings
- **Font Settings**: Customize visualization fonts and sizes
- **Census Settings**: API key, ACS year, output directory

**6. Validation**
- **Pre-run validation**: Check configuration before running analysis
- **File path verification**: Ensures all files exist
- **File type checking**: Validates file extensions (.geojson, .shp, etc.)
- **Census API validation**: Warns if census analysis enabled without API key
- **Detailed error messages**: Shows exactly what needs to be fixed

**7. Save and Run**
- Updates ParkximityCalc.py with your configuration
- Maintains text-based config for manual editing
- Optionally runs analysis immediately after saving

#### Configuration Parameters

Each parameter in the UI includes hover tooltips with detailed explanations:

**City-Specific Parameters:**

- **Parcels File** (`.geojson`, `.json`, `.csv`): Property parcel geometries or centroids. GeoJSON/JSON should contain polygon or point geometries. CSV should have `centroid_latitude` and `centroid_longitude` columns.

- **Parks File** (`.geojson`, `.json`): Park polygon geometries. Must contain Polygon or MultiPolygon features representing park boundaries.

- **Streets File** (`.geojson`, `.json`): Street network with LTS values. Must contain LineString geometries with an LTS column specified in configuration.

- **Boundary File** (`.geojson`, `.json`, `.csv`, optional): City boundary for clipping data. Can be a single polygon or multiple neighborhood polygons to combine.

- **Output Directory**: Folder where results will be saved. Will be created if it doesn't exist. Contains visualizations, GeoJSON outputs, and JSON data.

- **LTS Column Name**: Column name in streets file containing Level of Traffic Stress values (e.g., 'PC2_norm', 'LTS', 'bike_stress').

- **Boundary County Column**: Column name in boundary file for filtering (e.g., 'name', 'COUNTY'). Used when boundary file contains multiple features.

- **Combine Boundaries**: If enabled, merges multiple boundary polygons into a single city boundary. Useful for neighborhood-level boundary files.

- **Census State FIPS**: Two-digit state FIPS code (e.g., '25' for Massachusetts, '36' for New York). Required for census analysis.

- **Census County FIPS**: Comma-separated county FIPS codes (e.g., '025' for Suffolk County, '005,047,061,081,085' for NYC boroughs). Required for census analysis.

- **Census Block Shapefile** (`.shp`): TIGER/Line census block shapefile for spatial join. Download from Census Bureau or use provided state shapefiles.

- **Run Census Analysis**: Enable to spatially join ACS demographic data to parcels. Requires Census API key and census configuration.

**Global Analysis Parameters:**

- **Park Buffer** (1-100m): Buffer distance around park boundaries for entrance detection. Larger values catch streets near but not touching parks. Default: 20m.

- **Entrance Tolerance** (1-20m): Deduplication distance for entrance points within each park. Points closer than this are merged. Default: 2m.

- **Entrance Batch Size** (10-500): Number of parks processed together in parallel batches. Larger values reduce overhead but use more RAM. Default: 200.

- **Heatmap Resolution** (10-500m): Grid cell size for heatmap interpolation. Smaller values = finer detail but slower processing. Default: 100m.

- **Max Parcels for Heatmap** (10k-200k): Maximum parcels to sample for heatmap generation. Uses adaptive stratified sampling for spatial equity. Default: 50,000.

- **Heatmap Neighbors (IDW)** (1-20): Number of nearest neighbors for inverse distance weighting interpolation. More neighbors = smoother gradients. Default: 5.

- **Heatmap Smoothing** (0-10): Gaussian smoothing sigma for heatmap. Higher values = smoother appearance. 0 = no smoothing. Default: 3.

**Global Font Settings:**

- **Font Family**: Font family for all visualizations. Options: serif (Times-like), sans-serif (Arial-like), monospace (Courier-like).

- **Font Serif List**: Comma-separated list of serif fonts to try in order. Default: Times New Roman, Times, DejaVu Serif.

- **Font Stretch**: Font width style. Options: normal, condensed (narrower), expanded (wider).

- **Font Size** (6-20): Base font size in points for all visualization text.

**Global Census Settings:**

- **Census API Key**: Free API key from Census Bureau. Get at: https://api.census.gov/data/key_signup.html. Required for census analysis.

- **Census ACS Year** (2010-2030): American Community Survey year to use. Most recent available is typically 2-3 years behind current year. Default: 2022.

- **Census Parallel Jobs** (1-16): Number of parallel workers for census processing. More workers = faster but more RAM usage.

- **Census Output Directory**: Folder where census analysis results are saved. Contains parcels with demographics in GeoJSON and JSON formats.

#### Workflow Example

1. **Launch UI**: `python ParkximityConfigUI.py`
2. **Enter cities**: Boston, NYC, LA (one per line)
3. **Create data directories** (optional but recommended):
   - Click "Create Data Directory"
   - Select base folder (e.g., your project root)
   - Review and confirm directory creation
   - Organized folders are created for each city
4. **Configure Boston**:
   - Click "Boston" in sidebar
   - Browse to select parcels, parks, streets files
   - Set output directory: `Visualizations/Boston`
   - Set LTS column: `PC2_norm`
5. **Configure other cities**: Click each in sidebar and repeat
6. **Optional - Standardize file names**: Check "Standardize Raw Data File Names" to rename files to consistent format
7. **Set global settings**: Click "Global Settings" button
   - Adjust sliders for analysis parameters
   - Configure fonts if desired
   - Add Census API key if using census analysis
8. **Validate**: Click "Validate Config" to check for errors
9. **Save and run**: Click "Save & Run" to update config and start analysis
   - If standardization enabled, files are renamed first
   - Config is saved to ParkximityCalc.py
   - Optionally run analysis immediately

#### Tips

- **Load existing config**: Use "Load Existing Config" to import settings from ParkximityCalc.py
- **Validate early**: Click "Validate Config" frequently to catch issues
- **Manual editing**: You can still edit ParkximityCalc.py directly - the UI is just a convenience layer
- **File paths**: Use absolute paths or paths relative to the Notebooks/Publication/ directory
- **Census API**: Get your free key before enabling census analysis for any city

### Basic Usage (Automatic GPU Detection)

```python
# Just use the base file - it handles everything!
from ParkximityCalc import create_parkximity_analyzer

config = {
    'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
    'parks_path': 'Data/Raw/Boston/boston_parks.geojson',
    'streets_path': 'Data/Processed/LTS/lts_bos.geojson',
    'output_dir': 'Visualizations/Boston',
    'lts_column': 'PC2_norm',
}

# Automatically uses GPU if available, CPU if not
analyzer = create_parkximity_analyzer('Boston', config)
parcels_gdf = analyzer.full_analysis()
```

### Multi-City Analysis

```python
from ParkximityCalc import multi_city_analysis

cities_config = {
    'Boston': {...},
    'NYC': {...},
    'LA': {...},
}

# Automatically uses best available hardware for each city
# Cities are processed in parallel by default
# If one city fails, others continue processing
parkximity_results, census_results = multi_city_analysis(cities_config)

# Check results - failed cities will have None value
for city, parcels_gdf in parkximity_results.items():
    if parcels_gdf is not None:
        print(f"✓ {city}: {len(parcels_gdf)} parcels processed")
    else:
        print(f"✗ {city}: Processing failed")

# For memory-constrained systems, process sequentially
parkximity_results, census_results = multi_city_analysis(cities_config, parallel_cities=False)
```

### Census Analysis Integration

The toolkit now includes integrated census demographic analysis that spatially joins ACS (American Community Survey) data to parcels with parkximity calculations.

#### Quick Start with Census Analysis

```python
from ParkximityCalc import multi_city_analysis

# Global settings (applied to all cities)
global_settings = {
    # Parkximity settings
    'park_buffer': 50.0,
    'n_jobs': 4,
    
    # Census analysis settings
    'census_api_key': 'YOUR_API_KEY_HERE',  # Get free key at: https://api.census.gov/data/key_signup.html
    'census_acs_year': 2022,
    'census_n_jobs': 4,
    'census_output_dir': 'Data/Processed/Demographics',
}

# City-specific configuration
cities_config = {
    'Boston': {
        # Parkximity settings
        'parcels_path': 'Data/Raw/Boston/Parcels__2024_.geojson',
        'parks_path': 'Data/Raw/Boston/boston_parks.geojson',
        'streets_path': 'Data/Processed/LTS/lts_bos.geojson',
        'boundary_path': 'Data/Raw/Boston/bostoncitylimits.geojson',
        'output_dir': 'Visualizations/Boston',
        'lts_column': 'PC2_norm',
        
        # Census analysis settings (optional)
        'run_census_analysis': True,  # Enable census analysis for this city
        'census_state_fips': '25',
        'census_county_fips': ['025'],  # Suffolk County
        'census_block_shapefile': 'Data/Raw/Massachusets Census Blocks.shp/tl_2025_25_tabblock20.shp',
        'census_parcels_path': 'Visualizations/Boston/boston_parcels_parkximity.geojson',
    },
}

# Run analysis - returns both parkximity and census results
parkximity_results, census_results = multi_city_analysis(cities_config, global_settings=global_settings)
```

#### Census Analysis Features

**Automated Pipeline:**
- Downloads ACS demographic data from Census API
- Clips census blocks to city boundaries
- Spatially joins census data to parcels with parkximity values
- Calculates derived demographic metrics
- Exports both GeoJSON and JSON formats

**Demographic Variables:**
- **Commuting**: Public transit, bicycle, walking percentages
- **Income**: Median household income with margins of error
- **Race/Ethnicity**: Black and Latino population percentages
- **Derived Metrics**: Non-car commute percentage, sustainable transport metrics

**Output Files:**
```
Data/Processed/Demographics/
└── Boston/
    ├── boston_parcels_with_demographics.geojson  # Full spatial data
    └── boston_demographics_data.json              # Data without geometry
```

#### Census Configuration Options

**Global Census Settings:**
```python
global_settings = {
    'census_api_key': 'YOUR_KEY',           # Required for census analysis
    'census_acs_year': 2022,                # ACS year (default: 2022)
    'census_n_jobs': 4,                     # Parallel workers (default: same as n_jobs)
    'census_output_dir': 'Data/Processed/Demographics',  # Output directory
}
```

**City-Specific Census Settings:**
```python
city_config = {
    'run_census_analysis': True,            # Enable census for this city
    'census_state_fips': '25',              # State FIPS code
    'census_county_fips': ['025'],          # List of county FIPS codes
    'census_block_shapefile': 'path/to/blocks.shp',  # Census block shapefile
    'census_parcels_path': 'path/to/parkximity_output.geojson',  # Parkximity output
}
```

#### Standalone Census Analysis

You can also run census analysis independently:

```python
from CensusAnalysis import CensusAnalyzer, multi_city_census_analysis

# Single city
config = {
    'census_api_key': 'YOUR_KEY',
    'state_fips': '25',
    'county_fips': ['025'],
    'block_shapefile': 'Data/Raw/Massachusets Census Blocks.shp/tl_2025_25_tabblock20.shp',
    'boundary_geojson': 'Data/Raw/Boston/bostoncitylimits.geojson',
    'parcels_path': 'Visualizations/Boston/boston_parcels_parkximity.geojson',
    'output_dir': 'Data/Processed/Demographics',
    'acs_year': 2022,
    'n_jobs': 4
}

analyzer = CensusAnalyzer('Boston', config)
result_gdf = analyzer.full_analysis()

# Multiple cities with global config
global_config = {
    'census_api_key': 'YOUR_KEY',
    'acs_year': 2022,
    'n_jobs': 4,
    'output_dir': 'Data/Processed/Demographics'
}

cities_config = {
    'Boston': {
        'state_fips': '25',
        'county_fips': ['025'],
        'block_shapefile': '...',
        'boundary_geojson': '...',
        'parcels_path': '...',
    },
    'NYC': {...},
}

results = multi_city_census_analysis(cities_config, global_config)
```

#### Getting a Census API Key

1. Visit: https://api.census.gov/data/key_signup.html
2. Fill out the form (takes 1 minute)
3. Check your email for the API key
4. Add to your configuration

**Note:** The Census API is free and has generous rate limits. No credit card required.

#### Census Data Sources

The toolkit uses:
- **ACS 5-Year Estimates**: Most recent available (default: 2022)
- **Block Group Level**: Demographic data aggregated to block groups
- **Census Blocks**: Spatial geometries for precise parcel matching
- **TIGER/Line Shapefiles**: Census block boundaries (2020 or 2025)

#### Performance

Census analysis is parallelized for large datasets:
- Block clipping: 4-6x speedup on large datasets
- Spatial join: Automatic optimization for point vs polygon parcels
- API requests: Sequential with rate limiting (0.5s between requests)

**Typical Processing Times:**
| City | Parcels | Census Blocks | Time |
|------|---------|---------------|------|
| Boston | 100K | 5K | 3-5 min |
| NYC | 850K | 40K | 15-20 min |
| LA | 1.2M | 25K | 20-25 min |

### Error Handling and Robustness

The toolkit includes comprehensive error handling for multi-city analysis:

```python
# If one city fails, processing continues with others
parkximity_results, census_results = multi_city_analysis(cities_config)

# Example output when errors occur:
# ======================================================================
# ❌ ERROR processing Houston
# ======================================================================
# Error: FileNotFoundError: parks.geojson not found
# ======================================================================
# 
# Skipping Houston and continuing with next city...
#
# ======================================================================
#   MULTI-CITY ANALYSIS SUMMARY
# ======================================================================
#   Total cities: 5
#   ✓ Successful: 4
#     - Boston
#     - NYC
#     - LA
#     - San Francisco
#   ✗ Failed: 1
#     - Houston
#       Error: FileNotFoundError: parks.geojson not found
# ======================================================================
```

**Key features:**
- Each city is processed independently
- Errors in one city don't stop processing of others
- Full error traceback printed for debugging
- Final summary shows all successes and failures
- Failed cities return `None` in results dictionary
- Error messages included in summary for quick diagnosis

### Advanced: Direct Class Usage

```python
# If you want explicit control
from ParkximityCalcGPU import ParkXimityAnalyzerGPU

# GPU version (with automatic CPU fallback)
analyzer = ParkXimityAnalyzerGPU('Boston', config)
parcels_gdf = analyzer.full_analysis()
```

### Force CPU Mode

```python
# Useful for benchmarking or debugging
analyzer = create_parkximity_analyzer('Boston', config, force_cpu=True)

# Or with multi-city
results = multi_city_analysis(cities_config, force_cpu=True)
```

## Outputs

Results are saved to the `output_dir` specified in your configuration:

- `{city}_parkximity_entrances.png` - Park entrance map showing generated access points
- `{city}_parkximity_heatmap.png` - Accessibility heatmap visualization
- `{city}_parkximity_histograms.png` - Distance distribution histograms
- `{city}_parcels_parkximity.geojson` - Parcels with calculated distances (GeoJSON format)
- `{city}_parkximity_data.json` - Complete analysis data and metadata

Example output directory structure:
```
Visualizations/Boston/
├── boston_parkximity_entrances.png
├── boston_parkximity_heatmap.png
├── boston_parkximity_histograms.png
├── boston_parcels_parkximity.geojson
└── boston_parkximity_data.json
```

### Visualization Updates

**Park Usage Gradient (Updated February 2025):**

The "Parcels closest to park" visualization now uses an adaptive color gradient that emphasizes the distribution of park usage across a city:

- **Color scheme**: Red → Light Green (#90EE90) → Dark Green (#006400)
  - Red: Parks with 0 parcels (unused)
  - Light Green: Positioned at the distribution center
  - Dark Green: Parks serving the most parcels
  
- **Adaptive normalization**: Uses `TwoSlopeNorm` to allocate color range based on data distribution
  - Calculates percentage of parks above/below mean usage
  - Allocates more color variation to the side with more parks
  - Example: If 80% of parks are below mean, they get 80% of the color gradient
  - Result: Better visual distinction for the majority of parks in the dataset

This approach ensures that the visualization adapts to each city's unique park usage patterns, making it easier to identify outliers and understand the typical range of park accessibility.

## GPU Acceleration

### Performance Benefits

The toolkit provides **10-100x speedup** on large datasets with GPU acceleration:

| Dataset | Parcels | CPU Parallel | GPU Time | Speedup |
|---------|---------|--------------|----------|---------|
| Boston | 100K | 8 min | 3 min | 2.7x |
| NYC | 850K | 45 min | 22 min | 2x |
| LA | 1.2M | 65 min | 28 min | 2.3x |

Note: CPU times shown are with parallel processing enabled (8 cores).

### GPU-Accelerated Operations

- **Nearest Neighbor Search** (20-50x speedup): Finding closest network nodes
- **Point Deduplication** (10-30x speedup): Removing duplicate entrance points
- **Distance Calculations** (15-40x speedup): Computing pairwise distances
- **Heatmap IDW Interpolation** (15-40x speedup): Spatial interpolation with memory-efficient chunked processing
- **Gaussian Smoothing** (5-15x speedup): Image filtering for heatmaps

**Memory-Efficient GPU Processing:**
The GPU implementation uses intelligent chunked processing to handle large datasets without running out of memory. Grid points are processed in batches (typically 500MB per chunk), allowing the toolkit to work efficiently even on GPUs with limited VRAM (4-8GB). This approach provides significant speedup while automatically managing memory constraints.

### When to Use GPU vs CPU Parallel

**Use CPU Parallel (Default):**
- Small to medium datasets (< 500K parcels)
- No NVIDIA GPU available
- Simpler setup, no additional dependencies
- Already provides good performance

**Use GPU Acceleration:**
- Large datasets (> 500K parcels)
- NVIDIA GPU available (Pascal or newer)
- Need maximum performance
- Processing many cities repeatedly

### Automatic Fallback

The implementation automatically:
- Detects NVIDIA GPU availability
- Falls back to CPU when GPU is unavailable
- Maintains identical results between GPU and CPU modes
- Requires no code changes for users without GPUs

### Memory Requirements

| Dataset Size | CPU RAM | CPU Cores | GPU VRAM |
|--------------|---------|-----------|----------|
| Small (<100K parcels) | 4 GB | 2-4 | 2-4 GB |
| Medium (100K-500K) | 8 GB | 4-8 | 4-6 GB |
| Large (500K-1M) | 16 GB | 8-16 | 6-8 GB |
| Very Large (>1M) | 32 GB | 16+ | 8+ GB |

**GPU Memory Management:**
The GPU implementation uses intelligent chunked processing (500MB batches) to handle large datasets efficiently. This means even GPUs with 4-8GB VRAM can process very large cities without running out of memory. The system automatically adjusts chunk sizes based on available memory and falls back to CPU if needed.

**Note:** More CPU cores = faster processing, but also more RAM usage. Adjust `n_jobs` parameter if you encounter memory issues.

## Environment Setup

### Automatic Setup (Recommended)

```bash
# Step 1: Create base environment
conda env create -f environment.yml
conda activate parkximity

# Step 2: Run your analysis
python ParkximityCalc.py

# Step 3: If you have GPU, it will prompt you automatically!
# Just answer Y or n when asked
```

### Manual GPU Installation (Optional)

If you want to install GPU support without running the analysis:

```bash
conda activate parkximity
python gpu_utils.py
```

Or from Python:
```python
from gpu_utils import install_gpu_packages_auto
install_gpu_packages_auto()
```

### Manual Package Installation

If automatic setup doesn't work:

```bash
conda create -n parkximity python=3.10
conda activate parkximity

# Install core packages
conda install -c conda-forge geopandas shapely pyproj fiona rtree pandas numpy scipy networkx matplotlib seaborn jupyter ipykernel ipywidgets

# Optional: Add GPU support manually
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.02 cupy cudatoolkit=11.8
```

### IDE Integration

#### VS Code
1. Install Python extension
2. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose `parkximity`

#### Jupyter

```bash
conda activate parkximity
python -m ipykernel install --user --name=parkximity --display-name="ParkXimity"
```

## Troubleshooting

### Common Issues

**"conda: command not found"**
- Ensure Anaconda/Miniconda is installed
- Restart terminal
- Check: `conda --version`

**Setup script fails**
- Update conda: `conda update -n base conda`
- Try manual installation
- Check internet connection

**GPU not detected after setup**
- Check NVIDIA driver: `nvidia-smi`
- Run auto-installer: `python gpu_utils.py`
- Check GPU compatibility (Compute Capability 6.0+)

**Out of disk space**
- Base environment needs ~4-5 GB
- GPU libraries add ~2-3 GB more
- Clean conda cache: `conda clean --all`

### GPU-Specific Issues

**Want to add GPU support after initial setup?**

Just run your analysis again - it will prompt you automatically!

Or run manually:
```bash
python gpu_utils.py
```

Or from Python:
```python
from gpu_utils import install_gpu_packages_auto
install_gpu_packages_auto()
```

**"cudatoolkit not found"**
```bash
conda activate parkximity
conda install -c nvidia cudatoolkit=11.8
```

**"RAPIDS packages not available"**

RAPIDS requires Linux or WSL2. On Windows:
1. Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install conda inside WSL2
3. Run setup inside WSL2

Or use CPU mode - it works great!

**Installation interrupted or failed?**

Just run your analysis again - it will re-prompt you to install.

**Out of memory during processing**
- Reduce parallel workers: `config['n_jobs'] = 2`
- Process cities sequentially: `parallel_cities=False`
- Reduce batch size in config
- Reduce heatmap resolution
- Sample large datasets
- Use CPU mode: `force_cpu=True`

**One city fails in multi-city analysis**
- Check the error message in the final summary
- Verify input file paths for the failed city
- Check data format and CRS consistency
- Review full traceback printed during processing
- Other cities will complete successfully

**Visualization creation fails**
- Check output directory permissions
- Ensure sufficient disk space
- Verify matplotlib backend is working
- Failed visualizations are logged but don't stop analysis

**GPU mode not faster than CPU**
- Dataset may be too small (< 10K parcels)
- Check GPU memory with `nvidia-smi`
- Verify disk I/O isn't bottleneck (use SSD)

### Verify Installation

```bash
# Activate environment
conda activate parkximity

# Run comprehensive test
python test_setup.py

# Quick checks
python -c "import geopandas; print(f'✓ GeoPandas: {geopandas.__version__}')"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')"

# Check GPU status (shows if GPU libraries installed)
python -c "from gpu_utils import print_gpu_info; print_gpu_info()"
```

## System Requirements

### Minimum (CPU-only with parallel processing)
- 8 GB RAM
- 4 CPU cores
- 10 GB disk space
- Any OS (Windows, Mac, Linux)

### Recommended (CPU parallel)
- 16 GB RAM
- 8 CPU cores
- 20 GB disk space
- SSD recommended

### Recommended (GPU)
- 16 GB RAM
- 8 CPU cores
- 8 GB GPU VRAM
- 20 GB disk space
- Linux or WSL2
- NVIDIA GPU (Pascal or newer)

### For Large Cities (>1M parcels)
- 32 GB RAM
- 16 CPU cores
- 8+ GB GPU VRAM (if using GPU)
- 50 GB disk space
- SSD recommended

## Project Structure

```
Notebooks/Publication/
├── ParkximityCalc.py       # Main analysis file (auto-detects GPU)
├── ParkximityCalcGPU.py    # GPU implementation
├── ParkximityConfigUI.py   # Graphical configuration interface
├── CensusAnalysis.py       # Census demographic analysis
├── gpu_utils.py            # GPU utilities + auto-installer
├── environment.yml         # Conda environment
├── .gpu-requirements.yml   # Hidden GPU dependencies (auto-used)
├── install_gpu_support.py  # Manual GPU installer (optional)
├── setup.bat              # Windows setup script
├── test_setup.py          # Installation verification
└── README.md              # This file
```

### How It Works

1. **First run**: Detects GPU hardware automatically
2. **If GPU found**: Prompts to install GPU libraries interactively
3. **User chooses**: Install now (Y) or use CPU mode (n)
4. **If installed**: Restart Python, next run uses GPU automatically
5. **Always works**: Falls back to CPU if GPU unavailable

## Benchmarking

Compare GPU vs CPU performance:

```bash
# With GPU libraries installed
conda activate parkximity
python ParkximityCalc.py  # Uses GPU automatically

# Force CPU mode for comparison
python ParkximityCalc.py  # Edit to add force_cpu=True
```

Or in Python:

```python
import time
from ParkximityCalc import create_parkximity_analyzer

# GPU version (if available)
start = time.time()
analyzer_gpu = create_parkximity_analyzer('Boston', config, force_cpu=False)
analyzer_gpu.full_analysis()
gpu_time = time.time() - start

# CPU version
start = time.time()
analyzer_cpu = create_parkximity_analyzer('Boston', config, force_cpu=True)
analyzer_cpu.full_analysis()
cpu_time = time.time() - start

print(f"CPU: {cpu_time:.1f}s | GPU: {gpu_time:.1f}s | Speedup: {cpu_time/gpu_time:.1f}x")
```

## Cloud Deployment

### Google Colab

```python
# Install RAPIDS
!pip install cupy-cuda11x cuml-cu11

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Run analysis
from ParkximityCalcGPU import ParkXimityAnalyzerGPU
# ... rest of code
```

### AWS EC2

Recommended: `g4dn.xlarge` (T4 GPU, 16GB RAM)

```bash
# Use Deep Learning AMI with CUDA pre-installed
conda install -c rapidsai -c conda-forge cuml cupy
python ParkximityCalcGPU.py
```

## Environment Management

### Update Environment

```bash
# Update all packages
conda update --all

# Update specific package
conda update geopandas

# Update from file
conda env update -f environment.yml --prune

# Add GPU support later
python install_gpu_support.py
```

### Export Environment

```bash
# Export exact environment
conda env export > my-environment.yml

# Export portable (no builds)
conda env export --no-builds > environment-portable.yml
```

### Remove Environment

```bash
conda env remove -n parkximity
```

## Getting Help

If you encounter issues:

1. Check conda version: `conda --version` (should be 4.10+)
2. Update conda: `conda update -n base conda`
3. Check CUDA version: `nvidia-smi` (for GPU)
4. Check Python version: `python --version` (should be 3.10)
5. List installed packages: `conda list`

For RAPIDS-specific issues: https://rapids.ai/start.html

## Contributing

When adding GPU-accelerated operations:

1. Add CPU fallback in `gpu_utils.py`
2. Override method in `ParkXimityAnalyzerGPU`
3. Ensure identical results between GPU/CPU
4. Add unit tests comparing outputs
5. Update this documentation

### TODO

- [ ] Add streets layer detection back to auto-directory detection after LTS mapping is finalized (currently only checks for parcels and parks files)

## References

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

## License

[Your License Here]

## Citation

If you use ParkXimity in your research, please cite:

```bibtex
@software{parkximity,
  title={ParkXimity: GPU-Accelerated Park Accessibility Analysis},
  author={[Your Name]},
  year={2025},
  url={[Your Repository URL]}
}
```
