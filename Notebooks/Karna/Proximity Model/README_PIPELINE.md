# Proximity Model Pipeline

## Overview

The Proximity Model pipeline processes OpenStreetMap (OSM) data and government datasets to generate comprehensive street network parquet files with detailed sidewalk, bikeway, crosswalk, and curb ramp information.

## Quick Start

### Option 1: Using the Batch Script (Windows - Recommended)

Simply double-click `run_pipeline.bat` and choose from the menu:

1. **Launch Configuration UI** - For first-time setup or changing settings
2. **Run Analysis + Verification** - For pre-configured setups
3. **Run Verification Only** - To verify existing output files

### Option 2: Manual Execution

#### Step 1: Configure (First Time Only)
```bash
python ProximityModelUI.py
```
- Add cities to analyze
- Set file paths for government data
- Configure column mappings
- Click "Save Config" then "Run Analysis"

#### Step 2: Run Analysis
The UI will automatically launch the analysis in a separate console window, or you can run manually:
```bash
python ProximityModel.py
```

#### Step 3: Verify Output
```bash
python verify_parquet.py
```

## Files

### Core Scripts

- **`ProximityModel.py`** - Main analysis pipeline (14 stages)
  - Fetches OSM data
  - Computes sanity buffers
  - Merges government data
  - Generates offset geometries
  - Detects blocks and intersections
  - Processes curb ramps and crosswalks
  - Exports parquet files

- **`ProximityModelUI.py`** - Configuration GUI
  - Visual interface for setting up analysis
  - Manages city configurations
  - Handles column mappings for government data
  - Saves configuration to ProximityModel.py
  - Launches analysis in separate window

- **`verify_parquet.py`** - Output verification and diagnostics
  - Validates parquet file structure
  - Checks data completeness
  - Generates diagnostic maps
  - Creates summary statistics

- **`run_pipeline.bat`** - Convenience launcher (Windows)
  - Menu-driven interface
  - Activates conda environment
  - Runs full pipeline or individual steps

### Documentation

- **`OSMNetworkDescription.md`** - Detailed specification of the 21-step pipeline
- **`README_PIPELINE.md`** - This file

## Workflow Changes (v2.0)

### What Changed?

The pipeline has been restructured for better maintainability and clarity:

**Old Workflow:**
- UI created a flag file (`.run_analysis`)
- Batch script monitored flag file
- Batch script ran ProximityModel.py
- UI closed after starting analysis

**New Workflow:**
- UI directly launches ProximityModel.py in a new console
- No flag files needed
- UI stays open for further configuration
- Batch script provides menu options instead of sequential execution

### Migration Notes

If you have existing configurations:
1. The UI can load existing configurations from ProximityModel.py
2. Click "Load Existing Config" on the startup screen
3. Your settings will be imported automatically

## Output Files

### Generated Files

For each city, the pipeline generates:

1. **`{CityName}_network.parquet`** - Full street network with 206 columns
   - Street geometry and attributes
   - Sidewalk geometries (left/right)
   - Bikeway geometries (left/right, lanes 1-2)
   - Curb ramp locations (start/end, positions 1-3)
   - Crosswalk geometries and attributes
   - Block IDs and block sides
   - Feature geometries (street/sidewalk/bikeway furniture)

2. **`{CityName}_sanity.parquet`** - Sanity buffer map
   - osmid
   - max_offset_width (meters)

### Diagnostic Output

After running `verify_parquet.py`:

- **`Output/Diagnostic Maps/`** - Visual maps showing:
  - Street network coverage
  - Sidewalk presence
  - Bikeway presence
  - Curb ramp locations
  - Crosswalk locations
  - Block structure

## Configuration

### Global Settings

- **Output Directory** - Where parquet files are saved
- **Default Max Speed** - Default speed limit (mph)
- **Curb Ramp Buffers** - Inner/outer trustworthiness buffers (meters)
- **GPU Usage** - Enable/disable GPU acceleration
- **Python Environment** - Path to Python executable

### City Settings

For each city:
- **Name** - City name (used for OSM query)
- **Government Data Paths** - Optional file paths for:
  - Parcels (required for parcel-based sanity buffer)
  - Street centerlines
  - Intersection nodes
  - Sidewalks
  - Bikelanes
  - Curb ramps
  - Crosswalks
  - Street/sidewalk/bikeway features
- **Column Mappings** - Map government data columns to output schema
- **Curb Ramp Trustworthy** - Whether to trust government curb ramp geometry

## Troubleshooting

### Common Issues

**Issue: GPU screen flickering**
- Solution: Disable GPU in Global Settings (use_gpu = False)

**Issue: Conda environment not found**
- Solution: Ensure ParkximityENV is created and activated
- Check: `conda env list`

**Issue: OSM data fetch fails**
- Solution: Check internet connection and city name spelling
- Try: Use exact city name from OSM (e.g., "San Francisco County, CA")

**Issue: Column mapping errors**
- Solution: Use "Column Config" button to map government data columns
- Ensure geometry columns are auto-detected (don't map geometry manually)

**Issue: Parquet file validation fails**
- Solution: Check console output for specific errors
- Verify input data files exist and are readable
- Check that government data CRS matches (EPSG:4326)

## Performance

### Typical Processing Times

- Small city (< 10k streets): 5-15 minutes
- Medium city (10k-50k streets): 15-60 minutes  
- Large city (> 50k streets): 1-3 hours

### GPU Acceleration

When enabled, GPU acceleration speeds up:
- Nearest neighbor searches (curb ramp assignment)
- Spatial distance calculations
- Parcel distance computations

Note: GPU acceleration may cause screen flickering on some systems. Disable if this occurs.

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Review OSMNetworkDescription.md for pipeline specification
3. Verify input data formats and column mappings
4. Check that all required Python packages are installed

## Version History

- **v2.0** - Restructured pipeline with 14 stages, improved UI workflow
- **v1.0** - Initial release with flag-file based workflow
