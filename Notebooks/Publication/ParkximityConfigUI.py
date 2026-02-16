"""
ParkXimity Configuration UI

A graphical interface for configuring parkximity analysis parameters.
This UI provides a layer on top of the text-based configuration in ParkximityCalc.py,
allowing users to configure analysis settings through a visual interface while
maintaining the ability to edit the config file directly if desired.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import json
import subprocess
import sys
from pathlib import Path

# Set appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Parameter help text dictionary
HELP_TEXT = {
    # City-specific parameters
    'parcels_path': (
        "Parcels File\n\n"
        "Property parcel geometries or centroids.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/JSON: Polygon or Point geometries\n"
        "• CSV: Must have 'centroid_latitude' and 'centroid_longitude' columns\n\n"
        "The parcels represent individual properties or land units for which park "
        "accessibility will be calculated."
    ),
    'parks_path': (
        "Parks File\n\n"
        "Park polygon geometries.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/JSON: Polygon or MultiPolygon features\n\n"
        "Must contain polygon geometries representing park boundaries. The analysis "
        "will generate entrance points where streets intersect park boundaries."
    ),
    'streets_path': (
        "Streets File\n\n"
        "Street network with Level of Traffic Stress (LTS) values.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/JSON: LineString geometries\n\n"
        "Must contain a column with LTS values (specified in 'LTS Column Name'). "
        "LTS values weight the network distance calculation - higher stress streets "
        "are less desirable for cycling/walking."
    ),
    'boundary_path': (
        "Boundary File (Optional)\n\n"
        "City boundary for clipping data to study area.\n\n"
        "Accepted formats:\n"
        "• GeoJSON/JSON: Polygon or MultiPolygon\n"
        "• CSV: WKT geometry column\n\n"
        "Can be a single polygon or multiple neighborhood polygons. If multiple "
        "polygons, enable 'Combine Boundaries' to merge them into one city boundary."
    ),
    'census_block_shapefile': (
        "Census Block Shapefile\n\n"
        "TIGER/Line census block shapefile for spatial join.\n\n"
        "Accepted formats:\n"
        "• Shapefile (.shp)\n\n"
        "Download from Census Bureau TIGER/Line files or use provided state shapefiles. "
        "Used to spatially join census demographic data to parcels. Required only if "
        "'Run Census Analysis' is enabled."
    ),
    'exports_output_dir': (
        "Exports Output Directory\n\n"
        "Global folder where all analysis outputs will be saved.\n\n"
        "Structure created automatically:\n"
        "  Exports/\n"
        "  ├── Data/\n"
        "  │   ├── {CityName}/\n"
        "  │   │   ├── parcels_with_parkximity.geojson\n"
        "  │   │   ├── parkximity_data.json\n"
        "  │   │   └── parcels_with_demographics.geojson (if census enabled)\n"
        "  └── Visualizations/\n"
        "      └── {CityName}/\n"
        "          ├── entrance_map.png\n"
        "          ├── heatmap.png\n"
        "          └── histograms.png\n\n"
        "Example: Exports\n"
        "  Creates: Exports/Data/Boston/, Exports/Visualizations/Boston/, etc."
    ),
    'lts_column': (
        "LTS Column Name\n\n"
        "Column name in streets file containing Level of Traffic Stress values.\n\n"
        "Common names:\n"
        "• PC2_norm (normalized principal component)\n"
        "• LTS (direct LTS values)\n"
        "• bike_stress\n\n"
        "This column is used to weight network distances - higher values indicate "
        "less comfortable cycling/walking conditions."
    ),
    'boundary_county_column': (
        "Boundary County Column\n\n"
        "Column name in boundary file for filtering features.\n\n"
        "Common names:\n"
        "• name\n"
        "• COUNTY\n"
        "• NAME\n\n"
        "Used when boundary file contains multiple features (e.g., multiple counties "
        "or neighborhoods). Specify which feature(s) to use via 'Boundary Name' setting."
    ),
    'census_state_fips': (
        "Census State FIPS Code\n\n"
        "Two-digit state FIPS code for census data.\n\n"
        "Examples:\n"
        "• 25 = Massachusetts\n"
        "• 36 = New York\n"
        "• 06 = California\n"
        "• 48 = Texas\n\n"
        "Find codes at: https://www.census.gov/library/reference/code-lists/ansi.html\n\n"
        "Required only if 'Run Census Analysis' is enabled."
    ),
    'census_county_fips': (
        "Census County FIPS Codes\n\n"
        "Comma-separated list of three-digit county FIPS codes.\n\n"
        "Examples:\n"
        "• 025 = Suffolk County, MA (Boston)\n"
        "• 037 = Los Angeles County, CA\n"
        "• 005,047,061,081,085 = NYC 5 boroughs\n\n"
        "Find codes at: https://www.census.gov/library/reference/code-lists/ansi.html\n\n"
        "Required only if 'Run Census Analysis' is enabled."
    ),
    'combine_boundaries': (
        "Combine Boundaries\n\n"
        "If enabled, merges multiple boundary polygons into a single city boundary.\n\n"
        "Use when:\n"
        "• Boundary file contains multiple neighborhoods\n"
        "• You want to analyze the entire city as one unit\n\n"
        "Disable when:\n"
        "• Boundary file has a single polygon\n"
        "• You want to analyze specific neighborhoods only"
    ),
    'run_census_analysis': (
        "Run Census Analysis\n\n"
        "Enable to spatially join ACS demographic data to parcels.\n\n"
        "When enabled:\n"
        "• Downloads demographic data from Census API\n"
        "• Joins data to parcels based on spatial location\n"
        "• Exports parcels with demographics to separate file\n\n"
        "Requirements:\n"
        "• Census API key (free from census.gov)\n"
        "• Census State FIPS\n"
        "• Census County FIPS\n"
        "• Census Block Shapefile"
    ),
    
    # Geographic data type specifications
    'parcels_geo_type': (
        "Parcels Geographic Data Type\n\n"
        "Specifies how geographic data is stored in the parcels file.\n\n"
        "Options:\n"
        "• auto: Automatically detect based on file type\n"
        "  - GeoJSON/JSON → geometry column\n"
        "  - CSV → lat/long columns\n"
        "• geometry: Use geometry column (GeoJSON/JSON)\n"
        "• latlong: Use separate latitude/longitude columns (CSV)\n\n"
        "Default: auto (recommended)"
    ),
    
    'parcels_lat_column': (
        "Parcels Latitude Column\n\n"
        "Column name containing latitude values in CSV files.\n\n"
        "Common names:\n"
        "• centroid_latitude\n"
        "• latitude\n"
        "• lat\n"
        "• y\n\n"
        "Default: centroid_latitude\n\n"
        "Only used when parcels_geo_type is 'latlong'."
    ),
    
    'parcels_long_column': (
        "Parcels Longitude Column\n\n"
        "Column name containing longitude values in CSV files.\n\n"
        "Common names:\n"
        "• centroid_longitude\n"
        "• longitude\n"
        "• long\n"
        "• lon\n"
        "• x\n\n"
        "Default: centroid_longitude\n\n"
        "Only used when parcels_geo_type is 'latlong'."
    ),
    
    'parcels_geometry_column': (
        "Parcels Geometry Column\n\n"
        "Column name containing geometry data in GeoJSON/JSON files.\n\n"
        "Common names:\n"
        "• geometry (standard GeoJSON)\n"
        "• geom\n"
        "• shape\n\n"
        "Default: geometry\n\n"
        "Only used when parcels_geo_type is 'geometry'."
    ),
    
    'parks_geo_type': (
        "Parks Geographic Data Type\n\n"
        "Specifies how geographic data is stored in the parks file.\n\n"
        "Options:\n"
        "• auto: Automatically detect (uses geometry column)\n"
        "• geometry: Use geometry column (GeoJSON/JSON)\n\n"
        "Default: auto (recommended)\n\n"
        "Note: Parks must be polygons, so lat/long is not supported."
    ),
    
    'parks_geometry_column': (
        "Parks Geometry Column\n\n"
        "Column name containing geometry data in GeoJSON/JSON files.\n\n"
        "Common names:\n"
        "• geometry (standard GeoJSON)\n"
        "• geom\n"
        "• shape\n\n"
        "Default: geometry\n\n"
        "Only used when parks_geo_type is 'geometry'."
    ),
    
    'boundary_geo_type': (
        "Boundary Geographic Data Type\n\n"
        "Specifies how geographic data is stored in the boundary file.\n\n"
        "Options:\n"
        "• auto: Automatically detect based on file type\n"
        "  - GeoJSON/JSON → geometry column\n"
        "  - CSV → WKT column\n"
        "• geometry: Use geometry column (GeoJSON/JSON)\n"
        "• wkt: Use Well-Known Text column (CSV)\n\n"
        "Default: auto (recommended)"
    ),
    
    'boundary_geometry_column': (
        "Boundary Geometry Column\n\n"
        "Column name containing geometry data in GeoJSON/JSON files.\n\n"
        "Common names:\n"
        "• geometry (standard GeoJSON)\n"
        "• geom\n"
        "• shape\n\n"
        "Default: geometry\n\n"
        "Only used when boundary_geo_type is 'geometry'."
    ),
    
    'boundary_wkt_column': (
        "Boundary WKT Column\n\n"
        "Column name containing Well-Known Text geometry in CSV files.\n\n"
        "Common names:\n"
        "• the_geom\n"
        "• wkt\n"
        "• geometry\n"
        "• geom_wkt\n\n"
        "Default: the_geom\n\n"
        "Only used when boundary_geo_type is 'wkt'."
    ),
    
    # Global analysis parameters
    'park_buffer': (
        "Park Buffer Distance\n\n"
        "Buffer distance (in meters) around park boundaries for entrance detection.\n\n"
        "Range: 1-100m\n"
        "Default: 20m\n\n"
        "Larger values:\n"
        "• Catch streets near but not exactly touching parks\n"
        "• Handle topology mismatches between datasets\n"
        "• May create false entrances if too large\n\n"
        "Smaller values:\n"
        "• More precise entrance detection\n"
        "• May miss legitimate entrances due to data gaps"
    ),
    'entrance_tolerance': (
        "Entrance Deduplication Tolerance\n\n"
        "Distance (in meters) for merging duplicate entrance points within each park.\n\n"
        "Range: 1-20m\n"
        "Default: 2m\n\n"
        "Points closer than this distance are merged into one entrance. This prevents "
        "duplicate entrances from topology issues while preserving legitimate multiple "
        "access points.\n\n"
        "Note: Deduplication is per-park only. Adjacent parks can have separate entrances "
        "at the same street corner."
    ),
    'entrance_batch_size': (
        "Entrance Batch Size\n\n"
        "Number of parks processed together in parallel batches.\n\n"
        "Range: 10-500\n"
        "Default: 200\n\n"
        "Larger values:\n"
        "• Reduce task overhead\n"
        "• Faster processing (2-3x speedup)\n"
        "• Use more RAM\n\n"
        "Recommended by system RAM:\n"
        "• 8 GB: 50-100\n"
        "• 16 GB: 200-400\n"
        "• 32 GB+: 500-1000"
    ),
    'heatmap_resolution': (
        "Heatmap Grid Resolution\n\n"
        "Grid cell size (in meters) for heatmap interpolation.\n\n"
        "Range: 10-500m\n"
        "Default: 100m\n\n"
        "Smaller values:\n"
        "• Finer detail and smoother gradients\n"
        "• Slower processing and larger memory usage\n"
        "• Recommended: 50-75m for high detail\n\n"
        "Larger values:\n"
        "• Faster processing\n"
        "• Coarser appearance\n"
        "• Recommended: 100-150m for quick previews"
    ),
    'max_parcels_for_heatmap': (
        "Max Parcels for Heatmap\n\n"
        "Maximum number of parcels to sample for heatmap generation.\n\n"
        "Range: 10,000-200,000\n"
        "Default: 50,000\n\n"
        "Uses adaptive stratified spatial sampling to ensure equitable geographic "
        "representation. The system automatically adjusts sampling based on density "
        "variation to prevent dense areas from dominating the visualization.\n\n"
        "Higher values:\n"
        "• More accurate heatmaps\n"
        "• Slower processing\n"
        "• Recommended: 100k-200k for publication quality\n\n"
        "Lower values:\n"
        "• Faster processing\n"
        "• Still maintains spatial equity\n"
        "• Recommended: 50k for quick analysis"
    ),
    'heatmap_neighbors': (
        "Heatmap IDW Neighbors\n\n"
        "Number of nearest neighbors for inverse distance weighting (IDW) interpolation.\n\n"
        "Range: 1-20\n"
        "Default: 5\n\n"
        "More neighbors:\n"
        "• Smoother gradients\n"
        "• More averaging of nearby values\n"
        "• Slower processing\n\n"
        "Fewer neighbors:\n"
        "• Sharper transitions\n"
        "• More local variation\n"
        "• Faster processing"
    ),
    'heatmap_smoothing': (
        "Heatmap Gaussian Smoothing\n\n"
        "Gaussian smoothing sigma for heatmap appearance.\n\n"
        "Range: 0-10\n"
        "Default: 3\n\n"
        "Higher values:\n"
        "• Smoother, more blurred appearance\n"
        "• Better for presentation\n"
        "• Hides local variation\n\n"
        "Lower values:\n"
        "• Sharper, more detailed\n"
        "• Shows local patterns\n"
        "• May appear noisy\n\n"
        "0 = No smoothing (raw interpolation)"
    ),
    
    # Font settings
    'font_family': (
        "Font Family\n\n"
        "Font family for all visualization text.\n\n"
        "Options:\n"
        "• serif: Times-like fonts (traditional, formal)\n"
        "• sans-serif: Arial-like fonts (modern, clean)\n"
        "• monospace: Courier-like fonts (technical)\n\n"
        "Default: serif\n\n"
        "Serif fonts are typically preferred for academic publications."
    ),
    'font_serif': (
        "Font Serif List\n\n"
        "Comma-separated list of serif fonts to try in order.\n\n"
        "Default: Times New Roman, Times, DejaVu Serif\n\n"
        "The system will use the first available font from this list. This ensures "
        "consistent appearance across different operating systems.\n\n"
        "Common serif fonts:\n"
        "• Times New Roman (Windows)\n"
        "• Times (Mac/Linux)\n"
        "• DejaVu Serif (Linux)\n"
        "• Liberation Serif (Linux)"
    ),
    'font_stretch': (
        "Font Stretch\n\n"
        "Font width style for visualization text.\n\n"
        "Options:\n"
        "• normal: Standard width\n"
        "• condensed: Narrower (fits more text)\n"
        "• expanded: Wider (more readable)\n\n"
        "Default: condensed\n\n"
        "Condensed fonts are useful for fitting labels in tight spaces on maps."
    ),
    'font_size': (
        "Font Size\n\n"
        "Base font size (in points) for all visualization text.\n\n"
        "Range: 6-20\n"
        "Default: 10\n\n"
        "Larger values:\n"
        "• More readable\n"
        "• Better for presentations\n"
        "• May crowd small visualizations\n\n"
        "Smaller values:\n"
        "• Fits more information\n"
        "• Better for detailed maps\n"
        "• May be hard to read when printed"
    ),
    
    # Census settings
    'census_api_key': (
        "Census API Key\n\n"
        "Free API key from the U.S. Census Bureau.\n\n"
        "Get your key at:\n"
        "https://api.census.gov/data/key_signup.html\n\n"
        "The key is free and takes 1 minute to obtain. No credit card required.\n\n"
        "Required only if 'Run Census Analysis' is enabled for any city.\n\n"
        "The API key allows downloading American Community Survey (ACS) demographic "
        "data for spatial analysis."
    ),
    'census_acs_year': (
        "Census ACS Year\n\n"
        "American Community Survey (ACS) year to use for demographic data.\n\n"
        "Range: 2010-2030\n"
        "Default: 2022\n\n"
        "The most recent available data is typically 2-3 years behind the current year. "
        "For example, in 2025, the most recent data is usually 2022 or 2023.\n\n"
        "ACS 5-Year Estimates provide the most reliable data for small areas."
    ),
    'census_n_jobs': (
        "Census Parallel Jobs\n\n"
        "Number of parallel workers for census data processing.\n\n"
        "Range: 1-16\n"
        "Default: 4\n\n"
        "More workers:\n"
        "• Faster processing (4-6x speedup)\n"
        "• Higher RAM usage\n"
        "• Recommended for systems with 16GB+ RAM\n\n"
        "Fewer workers:\n"
        "• Lower RAM usage\n"
        "• Slower processing\n"
        "• Recommended for systems with 8GB RAM"
    ),
    'census_output_dir': (
        "Census Output Directory (Deprecated)\n\n"
        "This setting is deprecated. Census data is now saved in:\n"
        "  {Exports}/Data/{CityName}/\n\n"
        "All data outputs (parkximity and census) are consolidated in the Data folder."
    ),
    
    # UI-specific
    'create_data_directory': (
        "Create Data Directory\n\n"
        "Creates an organized folder structure for your analysis.\n\n"
        "What it creates:\n"
        "• Data/Raw/{CityName}/ - For input data files\n"
        "• Data/Processed/LTS/ - For LTS-processed street networks\n"
        "• Data/Processed/Census/ - For census block data\n"
        "• Data/Processed/Demographics/ - For census analysis results\n"
        "• Visualizations/{CityName}/ - For output visualizations\n\n"
        "Before creating:\n"
        "• Checks if directories already exist\n"
        "• Prompts for base location\n"
        "• Creates subdirectories for each city entered above\n\n"
        "⚠ Important:\n"
        "The Raw data folders (Data/Raw/{CityName}/) must be populated manually "
        "with your input files (parcels, parks, streets, boundaries). The tool "
        "only creates the folder structure.\n\n"
        "This helps organize your data consistently across multiple cities."
    ),
    
    'standardize_file_names': (
        "Standardize Raw Data File Names\n\n"
        "When enabled, renames input files to a consistent format before running analysis.\n\n"
        "Naming pattern:\n"
        "{city}_{layer}_{date}.{extension}\n\n"
        "Examples:\n"
        "• boston_parcels_2.16.26.geojson\n"
        "• nyc_parks_2.16.26.geojson\n"
        "• la_streets_2.16.26.geojson\n"
        "• sf_boundary_2.16.26.geojson\n\n"
        "Layers:\n"
        "• parcels - Property parcels\n"
        "• parks - Park polygons\n"
        "• streets - Street network with LTS\n"
        "• boundary - City boundary\n"
        "• census_blocks - Census block shapefile\n\n"
        "Date format: M.D.YY (e.g., 2.16.26 for Feb 16, 2026)\n\n"
        "What happens:\n"
        "• Creates a copy with standardized name\n"
        "• Original file is preserved\n"
        "• Config paths are updated to use new names\n"
        "• Helps track data versions and maintain consistency\n\n"
        "⚠ Note: Files are renamed when you click 'Save & Run'"
    ),
    
    'use_detected_directory': (
        "Use Detected Data Directory\n\n"
        "A standardized data directory structure has been detected!\n\n"
        "What this does:\n"
        "• Automatically scans the detected directory for standardized files\n"
        "• Matches files to cities based on naming pattern:\n"
        "  {city}_{layer}_{date}.{extension}\n"
        "• Auto-fills all file paths in the configuration\n"
        "• Sets output directories to Visualizations/{CityName}/\n\n"
        "Detected structure:\n"
        "• Data/Raw/{CityName}/ - Input files\n"
        "• Data/Processed/LTS/ - LTS street networks\n"
        "• Data/Processed/Census/ - Census blocks\n"
        "• Visualizations/{CityName}/ - Outputs\n\n"
        "Files detected:\n"
        "• {city}_parcels_*.geojson or .csv\n"
        "• {city}_parks_*.geojson\n"
        "• {city}_streets_*.geojson\n"
        "• {city}_boundary_*.geojson (optional)\n"
        "• {city}_census_blocks_*.shp (optional)\n\n"
        "This saves time by automatically configuring paths for all cities."
    ),
}


class ToolTip:
    """Simple, reliable tooltip using tkinter Toplevel."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.hide_job = None
        
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event):
        # Cancel any pending hide
        if self.hide_job:
            self.widget.after_cancel(self.hide_job)
            self.hide_job = None
        
        # Don't create if already exists
        if self.tooltip:
            return
            
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 30
        
        # Use regular tkinter Toplevel (more reliable than CTkToplevel)
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        self.tooltip.attributes('-topmost', True)
        
        # Use regular tkinter Label with dark styling and wraplength
        label = tk.Label(
            self.tooltip,
            text=self.text,
            justify="left",
            background="#1a1a1a",
            foreground="#ffffff",
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Inter", 9),
            wraplength=400  # Wrap text at 400 pixels
        )
        label.pack()
    
    def on_leave(self, event):
        # Schedule destruction with small delay
        if self.hide_job:
            self.widget.after_cancel(self.hide_job)
        self.hide_job = self.widget.after(200, self.destroy_tooltip)
    
    def destroy_tooltip(self):
        if self.tooltip:
            try:
                self.tooltip.destroy()
            except:
                pass
            self.tooltip = None
        self.hide_job = None


# RoundedButton class removed - using CTkButton instead


def create_info_button(parent, help_key):
    """
    Create an info button with tooltip for a parameter.
    
    Parameters
    ----------
    parent : ctk.CTkFrame
        Parent widget
    help_key : str
        Key in HELP_TEXT dictionary
    
    Returns
    -------
    ctk.CTkLabel
        Info button with tooltip
    """
    # Create small info button
    info_btn = ctk.CTkLabel(
        parent,
        text="ⓘ",
        font=("Inter", 14, "bold"),
        text_color=("#666666", "#999999"),
        cursor="hand2",
        width=20
    )
    
    # Add tooltip
    if help_key in HELP_TEXT:
        ToolTip(info_btn, HELP_TEXT[help_key])
    
    return info_btn


class ParkximityConfigUI:
    """Main configuration UI for parkximity analysis."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ParkXimity Configuration")
        
        # Update to ensure window is ready
        self.root.update_idletasks()
        
        # Force maximize on Windows
        try:
            self.root.state('zoomed')
            self.root.update()  # Force the state to apply
        except:
            # Fallback for other platforms
            try:
                self.root.attributes('-zoomed', True)
                self.root.update()
            except:
                # Last resort - set to large size
                self.root.geometry("1600x900")
        
        # Configuration data
        self.cities = []
        self.current_city = None
        self.city_configs = {}
        self.global_settings = {}
        
        # Detected data directory
        self.detected_data_dir = None
        
        # Default global settings
        self.default_global_settings = {
            "park_buffer": 20.0,
            "entrance_tolerance": 2.0,
            "entrance_batch_size": 200,
            "heatmap_resolution": 100,
            "max_parcels_for_heatmap": 50000,
            "heatmap_neighbors": 5,
            "heatmap_smoothing": 3,
            "font_family": "serif",
            "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font_stretch": "condensed",
            "font_size": 10,
            "census_api_key": "",
            "census_acs_year": 2022,
            "census_n_jobs": 4,
            "exports_output_dir": "Exports",
        }
        
        # Default city config template
        self.default_city_config = {
            'parcels_path': '',
            'parks_path': '',
            'streets_path': '',
            'boundary_path': '',
            'lts_column': 'PC2_norm',
            'boundary_county_column': 'name',
            'combine_boundaries': False,
            'run_census_analysis': False,
            'census_state_fips': '',
            'census_county_fips': [],
            'census_block_shapefile': '',
            'census_parcels_path': '',
            # Parcels geographic data column specifications
            'parcels_geo_type': 'auto',  # 'auto', 'geometry', or 'latlong'
            'parcels_lat_column': 'centroid_latitude',
            'parcels_long_column': 'centroid_longitude',
            'parcels_geometry_column': 'geometry',
            # Parks geographic data column specifications
            'parks_geo_type': 'auto',  # 'auto' or 'geometry'
            'parks_geometry_column': 'geometry',
            # Boundary geographic data column specifications
            'boundary_geo_type': 'auto',  # 'auto', 'geometry', or 'wkt'
            'boundary_geometry_column': 'geometry',
            'boundary_wkt_column': 'the_geom',
        }
        
        # Start with city selection
        self.show_city_selection()
    
    def show_city_selection(self):
        """Show initial dialog to select cities for analysis."""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Re-maximize after clearing widgets
        try:
            self.root.state('zoomed')
        except:
            pass
        
        frame = ctk.CTkFrame(self.root, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Title
        title_label = ctk.CTkLabel(
            frame, 
            text="PARKXIMITY CONFIGURATION", 
            font=("Inter", 24, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            frame,
            text="Select Cities for Analysis",
            font=("Inter", 16)
        )
        subtitle_label.pack(pady=(0, 20))
        
        instruction_label = ctk.CTkLabel(
            frame,
            text="Enter city names (one per line):",
            font=("Inter", 12, "bold")
        )
        instruction_label.pack(pady=5)
        
        self.city_text = ctk.CTkTextbox(
            frame, 
            height=300, 
            width=400,
            font=("Inter", 12),
            corner_radius=10
        )
        self.city_text.pack(pady=10)
        self.city_text.insert("1.0", "Boston\nHouston\nNYC\nLA\nSF")
        
        # Detect standardized data directory
        self.detected_data_dir = self.detect_data_directory()
        
        # Create data directory button with info icon
        data_dir_frame = ctk.CTkFrame(frame, fg_color="transparent")
        data_dir_frame.pack(pady=10)
        
        create_btn = ctk.CTkButton(
            data_dir_frame,
            text="CREATE DATA DIRECTORY",
            command=self.create_data_directory,
            font=("Inter", 12, "bold"),
            height=40,
            corner_radius=10
        )
        create_btn.pack(side="left", padx=5)
        
        info_btn = create_info_button(data_dir_frame, 'create_data_directory')
        info_btn.pack(side="left", padx=3)
        
        # Show "Use Detected Directory" button if found
        if self.detected_data_dir:
            detected_frame = ctk.CTkFrame(frame, fg_color="transparent")
            detected_frame.pack(pady=10)
            
            # Abbreviate the path for display
            abbreviated_path = self.abbreviate_path(self.detected_data_dir, max_length=50)
            
            detected_btn = ctk.CTkButton(
                detected_frame,
                text=f"USE DIRECTORY: {abbreviated_path}",
                command=self.use_detected_directory,
                font=("Inter", 12, "bold"),
                height=40,
                corner_radius=10,
                fg_color=("#1f538d", "#1f538d")
            )
            detected_btn.pack(side="left", padx=5)
            
            info_btn = create_info_button(detected_frame, 'use_detected_directory')
            info_btn.pack(side="left", padx=3)
        
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.pack(pady=20)
        
        continue_btn = ctk.CTkButton(
            button_frame,
            text="CONTINUE",
            command=self.process_cities,
            font=("Inter", 13, "bold"),
            height=45,
            width=150,
            corner_radius=10
        )
        continue_btn.pack(side="left", padx=5)
        
        load_btn = ctk.CTkButton(
            button_frame,
            text="LOAD EXISTING CONFIG",
            command=self.load_existing_config,
            font=("Inter", 13, "bold"),
            height=45,
            width=200,
            corner_radius=10,
            fg_color="transparent",
            border_width=2
        )
        load_btn.pack(side="left", padx=5)
    
    def process_cities(self):
        """Process the entered city names and show main config UI."""
        city_text = self.city_text.get("1.0", "end").strip()
        self.cities = [c.strip() for c in city_text.split("\n") if c.strip()]
        
        if not self.cities:
            messagebox.showerror("Error", "Please enter at least one city name.")
            return
        
        # Initialize configs for each city
        for city in self.cities:
            if city not in self.city_configs:
                self.city_configs[city] = self.default_city_config.copy()
        
        # Initialize global settings
        self.global_settings = self.default_global_settings.copy()
        
        self.show_main_config()
    
    def create_data_directory(self):
        """Create organized data directory structure for all cities."""
        # Get city names from text box
        city_text = self.city_text.get("1.0", "end").strip()
        cities = [c.strip() for c in city_text.split("\n") if c.strip()]
        
        if not cities:
            messagebox.showerror("Error", "Please enter at least one city name before creating directories.")
            return
        
        # Ask user for base directory
        base_dir = filedialog.askdirectory(
            title="Select Base Directory for Data Folders",
            initialdir="."
        )
        
        if not base_dir:
            return  # User cancelled
        
        base_path = Path(base_dir)
        
        # Define directory structure
        directories_to_create = []
        existing_dirs = []
        
        for city in cities:
            city_dirs = [
                base_path / "Data" / "Raw" / city,
                base_path / "Data" / "Processed" / "LTS",
                base_path / "Data" / "Processed" / "Census",
                base_path / "Data" / "Processed" / "Demographics" / city,
                base_path / "Visualizations" / city,
            ]
            
            for dir_path in city_dirs:
                if dir_path.exists():
                    existing_dirs.append(str(dir_path))
                else:
                    directories_to_create.append(dir_path)
        
        # Show summary
        summary_msg = f"Base directory: {base_path}\n\n"
        summary_msg += f"Cities: {', '.join(cities)}\n\n"
        
        if existing_dirs:
            summary_msg += f"Already exist ({len(existing_dirs)} directories):\n"
            for dir_path in existing_dirs[:5]:  # Show first 5
                summary_msg += f"  ✓ {dir_path}\n"
            if len(existing_dirs) > 5:
                summary_msg += f"  ... and {len(existing_dirs) - 5} more\n"
            summary_msg += "\n"
        
        if directories_to_create:
            summary_msg += f"Will create ({len(directories_to_create)} directories):\n"
            for dir_path in directories_to_create[:5]:  # Show first 5
                summary_msg += f"  + {dir_path}\n"
            if len(directories_to_create) > 5:
                summary_msg += f"  ... and {len(directories_to_create) - 5} more\n"
        else:
            summary_msg += "All required directories already exist!"
        
        # Confirm with user
        if directories_to_create:
            result = messagebox.askyesno(
                "Create Directories?",
                summary_msg + "\n\nProceed with creation?"
            )
            
            if not result:
                return
            
            # Create directories
            try:
                created_count = 0
                for dir_path in directories_to_create:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_count += 1
                
                messagebox.showinfo(
                    "Success",
                    f"Created {created_count} directories successfully!\n\n"
                    f"Directory structure:\n"
                    f"  Data/Raw/{'{CityName}'}/\n"
                    f"  Data/Processed/LTS/\n"
                    f"  Data/Processed/Census/\n"
                    f"  Data/Processed/Demographics/{'{CityName}'}/\n"
                    f"  Visualizations/{'{CityName}'}/\n\n"
                    f"⚠ Note: You must manually populate the Raw data folders\n"
                    f"(Data/Raw/{'{CityName}'}/) with your input files:\n"
                    f"  • Parcels (GeoJSON/CSV)\n"
                    f"  • Parks (GeoJSON)\n"
                    f"  • Streets (GeoJSON)\n"
                    f"  • Boundaries (GeoJSON/CSV, optional)"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create directories: {e}")
        else:
            messagebox.showinfo("Already Exists", summary_msg)
    
    def detect_data_directory(self):
        """Detect if a standardized data directory exists."""
        search_paths = [Path("."), Path("..")]
        
        for base_path in search_paths:
            data_dir = base_path / "Data"
            if not data_dir.exists():
                continue
            
            raw_dir = data_dir / "Raw"
            processed_dir = data_dir / "Processed"
            
            if not (raw_dir.exists() and processed_dir.exists()):
                continue
            
            # Check if any city subdirectory has valid files
            city_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
            if any(self._has_standardized_files(d) for d in city_dirs):
                return str(data_dir.resolve())
        
        return None
    
    def _has_standardized_files(self, city_dir):
        """Check if a city directory contains valid parks and parcels files."""
        patterns = [
            ('parcel', {'.geojson', '.json', '.csv'}),
            ('park', {'.geojson', '.json'})
        ]
        
        found_files = self._find_files_by_pattern(city_dir, patterns)
        
        # Check if we have at least one parcels and one parks file
        has_parcels = any('parcel' in f.name.lower() for f in found_files)
        has_parks = any('park' in f.name.lower() for f in found_files)
        
        return has_parcels and has_parks
    
    def abbreviate_path(self, path_str, max_length=60):
        """Abbreviate a path to fit within max_length characters."""
        if len(path_str) <= max_length:
            return path_str
        
        path = Path(path_str)
        parts = list(path.parts)
        
        if len(parts) <= 2:
            return path_str
        
        # Keep drive and last 3 parts (grandparent/parent/file)
        if len(parts) > 4:
            abbreviated = str(Path(parts[0]) / "..." / parts[-3] / parts[-2] / parts[-1])
        else:
            abbreviated = str(Path(parts[0]) / "..." / parts[-2] / parts[-1])
        
        # If still too long, progressively simplify
        if len(abbreviated) > max_length:
            abbreviated = f".../{parts[-2]}/{parts[-1]}"
        if len(abbreviated) > max_length:
            abbreviated = f".../{parts[-1]}"
        
        return abbreviated
    
    def abbreviate_path(self, path_str, max_length=60):
        """Abbreviate a path to fit within max_length characters."""
        if len(path_str) <= max_length:
            return path_str
        
        path = Path(path_str)
        parts = list(path.parts)
        
        if len(parts) <= 2:
            return path_str
        
        # Keep drive and last 3 parts (grandparent/parent/file)
        if len(parts) > 4:
            abbreviated = str(Path(parts[0]) / "..." / parts[-3] / parts[-2] / parts[-1])
        else:
            abbreviated = str(Path(parts[0]) / "..." / parts[-2] / parts[-1])
        
        # If still too long, progressively simplify
        if len(abbreviated) > max_length:
            abbreviated = f".../{parts[-2]}/{parts[-1]}"
        if len(abbreviated) > max_length:
            abbreviated = f".../{parts[-1]}"
        
        return abbreviated
    
    def _validate_file_path(self, path_str, label, valid_extensions):
        """Validate a file path exists and has correct extension."""
        errors = []
        if not path_str:
            return errors  # Optional files can be empty
        
        path = Path(path_str)
        if not path.exists():
            errors.append(f"  • {label} file not found: {path_str}")
        elif not any(path_str.lower().endswith(ext) for ext in valid_extensions):
            valid_ext_str = ', '.join(valid_extensions)
            errors.append(
                f"  • {label} file has invalid extension.\n"
                f"    Expected: {valid_ext_str}\n"
                f"    Got: {path.suffix}"
            )
        return errors
    
    def _find_files_by_pattern(self, directory, patterns):
        """Find files matching any of the given patterns."""
        if not directory.exists():
            return []
        
        try:
            files = [f for f in directory.iterdir() if f.is_file()]
        except (OSError, PermissionError):
            return []
        
        matching_files = []
        for file in files:
            file_name_lower = file.name.lower()
            for pattern, extensions in patterns:
                if pattern in file_name_lower and file.suffix.lower() in extensions:
                    matching_files.append(file)
                    break
        
        return matching_files
    
    def _create_labeled_field(self, parent, row, key, label, widget_creator):
        """Create a labeled field with info button - DRY helper."""
        label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
        info_btn = create_info_button(label_frame, key)
        info_btn.pack(side="left", padx=3)
        
        widget = widget_creator(parent, row)
        return widget
    
    def _create_file_field(self, parent, row, key, label, config):
        """Create a file path field with browse button."""
        def create_widget(parent, row):
            entry = ctk.CTkEntry(parent, width=400)
            entry.grid(row=row, column=1, padx=5, pady=5)
            
            path_value = config.get(key, '')
            if path_value:
                abbreviated = self.abbreviate_path(path_value, max_length=60)
                entry.insert(0, abbreviated)
                entry.full_path = path_value
            else:
                entry.insert(0, '')
                entry.full_path = ''
            
            btn = ctk.CTkButton(parent, text="Browse...", 
                               command=lambda e=entry: self.browse_file(e))
            btn.grid(row=row, column=2, padx=5, pady=5)
            
            return entry
        
        return self._create_labeled_field(parent, row, key, label, create_widget)
        
    def use_detected_directory(self):
        """Auto-populate configuration from detected data directory."""
        if not self.detected_data_dir:
            messagebox.showerror("Error", "No data directory detected.")
            return
        
        # detected_data_dir now points to the Data folder itself
        data_path = Path(self.detected_data_dir)
        
        # Get city names from text box
        city_text = self.city_text.get("1.0", "end").strip()
        cities = [c.strip() for c in city_text.split("\n") if c.strip()]
        
        if not cities:
            messagebox.showerror("Error", "Please enter at least one city name.")
            return
        
        # Scan for files and auto-populate configs
        found_files = {}
        missing_cities = []
        
        for city in cities:
            city_lower = city.lower().replace(' ', '_')
            city_config = self.default_city_config.copy()
            
            # Search for files in Raw directory
            raw_dir = data_path / "Raw" / city
            if not raw_dir.exists():
                missing_cities.append(city)
                continue
            
            files_found = []
            
            # Look for ANY file with "parcel" in the name
            parcels_files = [f for f in raw_dir.glob("*") if f.is_file() and 'parcel' in f.name.lower() 
                           and f.suffix.lower() in {'.geojson', '.json', '.csv'}]
            if parcels_files:
                city_config['parcels_path'] = str(parcels_files[0])
                files_found.append('parcels')
            
            # Look for ANY file with "park" in the name
            parks_files = [f for f in raw_dir.glob("*") if f.is_file() and 'park' in f.name.lower() 
                          and f.suffix.lower() in {'.geojson', '.json'}]
            if parks_files:
                city_config['parks_path'] = str(parks_files[0])
                files_found.append('parks')
            
            # Only consider city valid if both parks and parcels are found
            if len(files_found) < 2:
                missing_cities.append(city)
                continue
            
            # Streets (optional - check Raw first, then Processed/LTS, but don't report in summary)
            streets_files = [f for f in raw_dir.glob("*") if f.is_file() and 
                           ('street' in f.name.lower() or 'bike' in f.name.lower()) 
                           and f.suffix.lower() in {'.geojson', '.json'}]
            if not streets_files:
                lts_dir = data_path / "Processed" / "LTS"
                if lts_dir.exists():
                    streets_files = list(lts_dir.glob(f"lts_{city_lower[:3]}.*"))
                    if not streets_files:
                        streets_files = [f for f in lts_dir.glob("*") if f.is_file() and 
                                       city_lower[:3] in f.name.lower()]
            
            if streets_files:
                city_config['streets_path'] = str(streets_files[0])
            
            # Boundary (optional - don't report in summary)
            boundary_files = [f for f in raw_dir.glob("*") if f.is_file() and 
                            ('boundary' in f.name.lower() or 'limit' in f.name.lower()) 
                            and f.suffix.lower() in {'.geojson', '.json', '.csv'}]
            if boundary_files:
                city_config['boundary_path'] = str(boundary_files[0])
            
            # Census blocks (optional - search in Raw folder first, then Processed/Census)
            census_shp = None
            
            # Look for folders with "blocks" or "census" in the name within the city's Raw folder
            try:
                for item in raw_dir.iterdir():
                    if item.is_dir():
                        folder_name_lower = item.name.lower()
                        if 'block' in folder_name_lower or 'census' in folder_name_lower:
                            # Found a census/blocks folder, look for .shp file inside
                            shp_files = list(item.glob("*.shp"))
                            if shp_files:
                                census_shp = str(shp_files[0])
                                break
            except (OSError, PermissionError):
                pass
            
            # If not found in Raw, check Processed/Census
            if not census_shp:
                census_dir = data_path / "Processed" / "Census"
                if census_dir.exists():
                    census_files = list(census_dir.glob(f"*{city_lower}*.shp"))
                    if not census_files:
                        # Try state-level census files
                        census_files = list(census_dir.glob("*.shp"))
                    if census_files:
                        census_shp = str(census_files[0])
            
            if census_shp:
                city_config['census_block_shapefile'] = census_shp
            
            # Store config and found files
            self.city_configs[city] = city_config
            found_files[city] = files_found
        
        # Show summary - only report parks and parcels
        summary_msg = f"Auto-populated configuration from:\n{self.detected_data_dir}\n\n"
        
        if found_files:
            summary_msg += "Cities with valid data (parks + parcels):\n"
            for city, files in found_files.items():
                summary_msg += f"  ✓ {city}\n"
        
        if missing_cities:
            summary_msg += f"\n\nCities missing required files:\n"
            for city in missing_cities:
                summary_msg += f"  ✗ {city} (needs parks and parcels files)\n"
            summary_msg += "\nYou'll need to add the files or configure these manually."
        
        messagebox.showinfo("Auto-Configuration Complete", summary_msg)
        
        # Update cities list to only include found cities
        self.cities = list(found_files.keys())
        
        # Initialize global settings and set exports output directory
        self.global_settings = self.default_global_settings.copy()
        # Set exports output directory to parent of Data folder
        self.global_settings['exports_output_dir'] = str(data_path.parent / "Exports")
        
        # Show main config
        if self.cities:
            self.show_main_config()
        else:
            messagebox.showerror("Error", "No cities found in the detected directory.")
    
    def load_existing_config(self):
        """Load configuration from ParkximityCalc.py."""
        try:
            calc_file = Path("Notebooks/Publication/ParkximityCalc.py")
            
            if not calc_file.exists():
                messagebox.showerror("Error", "ParkximityCalc.py not found")
                return
            
            # Read the file
            with open(calc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the configuration
            parsed_config = self.parse_config_from_file(content)
            
            if not parsed_config:
                messagebox.showerror("Error", "Could not parse configuration from file")
                return
            
            # Load the parsed config
            self.global_settings = parsed_config.get('global_settings', self.default_global_settings.copy())
            self.city_configs = parsed_config.get('cities_config', {})
            self.cities = list(self.city_configs.keys())
            
            if not self.cities:
                messagebox.showwarning("Warning", "No cities found in config. Starting with defaults.")
                self.process_cities()
                return
            
            messagebox.showinfo("Success", f"Loaded configuration for {len(self.cities)} cities")
            self.show_main_config()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            import traceback
            traceback.print_exc()
    
    def parse_config_from_file(self, content):
        """Parse configuration from ParkximityCalc.py content."""
        import ast
        import re
        
        try:
            # Find the main section
            main_match = re.search(r'if __name__ == "__main__":(.*)', content, re.DOTALL)
            if not main_match:
                return None
            
            main_content = main_match.group(1)
            
            # Extract global_settings dictionary
            global_match = re.search(r'global_settings\s*=\s*(\{[^}]*\})', main_content, re.DOTALL)
            global_settings = {}
            if global_match:
                try:
                    # Use ast.literal_eval to safely parse the dictionary
                    global_dict_str = global_match.group(1)
                    # Handle multi-line dictionaries by finding the matching closing brace
                    brace_count = 0
                    start_pos = main_content.find('global_settings')
                    if start_pos != -1:
                        dict_start = main_content.find('{', start_pos)
                        if dict_start != -1:
                            pos = dict_start
                            for i, char in enumerate(main_content[dict_start:]):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        global_dict_str = main_content[dict_start:dict_start + i + 1]
                                        break
                    
                    global_settings = ast.literal_eval(global_dict_str)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse global_settings: {e}")
                    global_settings = self.default_global_settings.copy()
            
            # Extract cities_config dictionary
            cities_match = re.search(r'cities_config\s*=\s*\{', main_content)
            cities_config = {}
            if cities_match:
                try:
                    # Find the cities_config dictionary
                    dict_start = cities_match.end() - 1
                    brace_count = 0
                    pos = dict_start
                    for i, char in enumerate(main_content[dict_start:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                cities_dict_str = main_content[dict_start:dict_start + i + 1]
                                cities_config = ast.literal_eval(cities_dict_str)
                                break
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse cities_config: {e}")
                    cities_config = {}
            
            return {
                'global_settings': global_settings,
                'cities_config': cities_config
            }
            
        except Exception as e:
            print(f"Error parsing config: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def show_main_config(self):
        """Show main configuration interface with sidebar and config panels."""
        # Clear root
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Re-maximize after clearing widgets
        try:
            self.root.state('zoomed')
        except:
            pass
        
        # Create main layout
        # Sidebar for city list
        sidebar = ctk.CTkFrame(self.root, width=200)
        sidebar.pack(side="left", fill="y", padx=5, pady=5)
        sidebar.pack_propagate(False)
        
        ctk.CTkLabel(sidebar, text="Cities", font=("Inter", 16, "bold")).pack(pady=10)
        
        # City buttons frame
        city_buttons_frame = ctk.CTkScrollableFrame(sidebar)
        city_buttons_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create button for each city
        self.city_buttons = {}
        for city in self.cities:
            btn = ctk.CTkButton(
                city_buttons_frame,
                text=city,
                command=lambda c=city: self.select_city(c),
                height=35,
                corner_radius=8
            )
            btn.pack(fill="x", pady=2)
            self.city_buttons[city] = btn
        
        # Add City button
        add_city_btn = ctk.CTkButton(
            city_buttons_frame,
            text="+ Add City",
            command=self.add_new_city,
            height=35,
            corner_radius=8,
            fg_color="transparent",
            border_width=2,
            hover_color=("#3a7ebf", "#1f538d")
        )
        add_city_btn.pack(fill="x", pady=10)
        
        # Buttons at bottom of sidebar
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # Standardize file names checkbox with info button
        standardize_frame = ctk.CTkFrame(btn_frame, fg_color="transparent")
        standardize_frame.pack(fill="x", pady=5)
        
        self.standardize_names_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            standardize_frame, 
            text="Standardize Files",
            variable=self.standardize_names_var,
            font=("Inter", 10)
        ).pack(side="left")
        
        info_btn = create_info_button(standardize_frame, 'standardize_file_names')
        info_btn.pack(side="left", padx=3)
        
        ctk.CTkButton(btn_frame, text="Validate Config", 
                  command=self.validate_and_show_results, height=32).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="Global Settings", 
                  command=self.show_global_settings, height=32).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="Save & Run", 
                  command=self.save_and_run, height=32).pack(fill="x", pady=2)
        
        # Main config area
        self.config_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.config_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Select first city by default
        if self.cities:
            self.select_city(self.cities[0])
    
    def select_city(self, city_name):
        """Handle city selection."""
        self.current_city = city_name
        
        # Update button states
        for city, btn in self.city_buttons.items():
            if city == city_name:
                btn.configure(fg_color=("#1f538d", "#1f538d"))
            else:
                btn.configure(fg_color=("#3b8ed0", "#1f6aa5"))
        
        self.show_city_config(city_name)
    
    def add_new_city(self):
        """Add a new city to the configuration."""
        # Create a simple dialog to get city name
        dialog = ctk.CTkInputDialog(
            text="Enter city name:",
            title="Add New City"
        )
        city_name = dialog.get_input()
        
        if city_name and city_name.strip():
            city_name = city_name.strip()
            
            # Check if city already exists
            if city_name in self.cities:
                messagebox.showwarning("Duplicate City", f"{city_name} already exists in the list.")
                return
            
            # Add city to list
            self.cities.append(city_name)
            
            # Initialize config for new city
            self.city_configs[city_name] = self.default_city_config.copy()
            
            # Refresh the main config view to show new city
            self.show_main_config()
            
            # Select the new city
            self.select_city(city_name)
    
    def on_city_select(self, event):
        """Handle city selection from sidebar (legacy method)."""
        # This method is kept for compatibility but not used
        pass
    
    def show_city_config(self, city_name):
        """Show configuration panel for selected city."""
        # Clear config frame
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable frame - CTk has built-in scrollable frame!
        scrollable_frame = ctk.CTkScrollableFrame(self.config_frame, fg_color="transparent")
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            scrollable_frame, 
            text=f"Configuration for {city_name}", 
            font=("Inter", 18, "bold")
        ).grid(row=0, column=0, columnspan=3, pady=10)
        
        # Ensure city config exists
        if city_name not in self.city_configs:
            self.city_configs[city_name] = self.default_city_config.copy()
        
        config = self.city_configs[city_name]
        self.city_widgets = {}
        
        row = 1
        
        # File path fields
        file_fields = [
            ('parcels_path', 'Parcels File'),
            ('parks_path', 'Parks File'),
            ('streets_path', 'Streets File'),
            ('boundary_path', 'Boundary File'),
            ('census_block_shapefile', 'Census Block Shapefile'),
        ]
        
        for key, label in file_fields:
            # Label with info button
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            entry = ctk.CTkEntry(scrollable_frame, width=400)
            entry.grid(row=row, column=1, padx=5, pady=5)
            
            # Abbreviate path for display
            path_value = config.get(key, '')
            if path_value:
                abbreviated = self.abbreviate_path(path_value, max_length=60)
                entry.insert(0, abbreviated)
                # Store full path in a custom attribute
                entry.full_path = path_value
            else:
                entry.insert(0, '')
                entry.full_path = ''
            
            btn = ctk.CTkButton(scrollable_frame, text="Browse...", 
                           command=lambda e=entry: self.browse_file(e))
            btn.grid(row=row, column=2, padx=5, pady=5)
            
            self.city_widgets[key] = entry
            row += 1
        
        # Text fields
        text_fields = [
            ('lts_column', 'LTS Column Name'),
            ('boundary_county_column', 'Boundary County Column'),
            ('census_state_fips', 'Census State FIPS'),
        ]
        
        for key, label in text_fields:
            # Label with info button
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            entry = ctk.CTkEntry(scrollable_frame, width=400)
            entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
            entry.insert(0, config.get(key, ''))
            
            self.city_widgets[key] = entry
            row += 1
        
        # Census county FIPS (list field)
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Census County FIPS (comma-separated):", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'census_county_fips')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        fips_list = config.get('census_county_fips', [])
        entry.insert(0, ', '.join(fips_list) if isinstance(fips_list, list) else str(fips_list))
        
        self.city_widgets['census_county_fips'] = entry
        row += 1
        
        # Boolean fields
        bool_fields = [
            ('combine_boundaries', 'Combine Boundaries'),
            ('run_census_analysis', 'Run Census Analysis'),
        ]
        
        for key, label in bool_fields:
            # Label with info button
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            var = ctk.BooleanVar(value=config.get(key, False))
            check = ctk.CTkCheckBox(scrollable_frame, variable=var, text="")
            check.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            
            self.city_widgets[key] = var
            row += 1
        
        # Save button
        ctk.CTkButton(scrollable_frame, text="Save City Config", 
                  command=lambda: self.save_city_config(city_name), height=35).grid(
            row=row, column=0, columnspan=3, pady=20)
        row += 1
        
        # Add visual separator before geographic data specifications
        separator = ctk.CTkFrame(scrollable_frame, height=2, fg_color=("#3a3a3a", "#3a3a3a"))
        separator.grid(row=row, column=0, columnspan=3, sticky='ew', pady=15, padx=5)
        row += 1
        
        ctk.CTkLabel(
            scrollable_frame, 
            text="Geographic Data Column Specifications", 
            font=("Inter", 14, "bold")
        ).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # === PARCELS GEOGRAPHIC DATA ===
        ctk.CTkLabel(scrollable_frame, text="Parcels:", font=("Inter", 12)).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        row += 1
        
        # Parcels geo type
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Data Type:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parcels_geo_type')
        info_btn.pack(side="left", padx=3)
        
        parcels_geo_var = ctk.StringVar(value=config.get('parcels_geo_type', 'auto'))
        parcels_geo_combo = ctk.CTkComboBox(
            scrollable_frame,
            values=['auto', 'geometry', 'latlong'],
            variable=parcels_geo_var,
            width=400,
            state="readonly"
        )
        parcels_geo_combo.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        parcels_geo_combo.set(config.get('parcels_geo_type', 'auto'))
        self.city_widgets['parcels_geo_type'] = parcels_geo_var
        row += 1
        
        # Parcels geometry column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Geometry Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parcels_geometry_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('parcels_geometry_column', 'geometry'))
        self.city_widgets['parcels_geometry_column'] = entry
        row += 1
        
        # Parcels latitude column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Latitude Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parcels_lat_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('parcels_lat_column', 'centroid_latitude'))
        self.city_widgets['parcels_lat_column'] = entry
        row += 1
        
        # Parcels longitude column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Longitude Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parcels_long_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('parcels_long_column', 'centroid_longitude'))
        self.city_widgets['parcels_long_column'] = entry
        row += 1
        
        # === PARKS GEOGRAPHIC DATA ===
        ctk.CTkLabel(scrollable_frame, text="Parks:", font=("Inter", 12)).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(15, 5))
        row += 1
        
        # Parks geo type
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Data Type:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parks_geo_type')
        info_btn.pack(side="left", padx=3)
        
        parks_geo_var = ctk.StringVar(value=config.get('parks_geo_type', 'auto'))
        parks_geo_combo = ctk.CTkComboBox(
            scrollable_frame,
            values=['auto', 'geometry'],
            variable=parks_geo_var,
            width=400,
            state="readonly"
        )
        parks_geo_combo.set(config.get('parks_geo_type', 'auto'))
        parks_geo_combo.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        self.city_widgets['parks_geo_type'] = parks_geo_var
        row += 1
        
        # Parks geometry column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Geometry Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'parks_geometry_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('parks_geometry_column', 'geometry'))
        self.city_widgets['parks_geometry_column'] = entry
        row += 1
        
        # === BOUNDARY GEOGRAPHIC DATA ===
        ctk.CTkLabel(scrollable_frame, text="Boundary:", font=("Inter", 12)).grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(15, 5))
        row += 1
        
        # Boundary geo type
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Data Type:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'boundary_geo_type')
        info_btn.pack(side="left", padx=3)
        
        boundary_geo_var = ctk.StringVar(value=config.get('boundary_geo_type', 'auto'))
        boundary_geo_combo = ctk.CTkComboBox(
            scrollable_frame,
            values=['auto', 'geometry', 'wkt'],
            variable=boundary_geo_var,
            width=400,
            state="readonly"
        )
        boundary_geo_combo.set(config.get('boundary_geo_type', 'auto'))
        boundary_geo_combo.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        self.city_widgets['boundary_geo_type'] = boundary_geo_var
        row += 1
        
        # Boundary geometry column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Geometry Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'boundary_geometry_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('boundary_geometry_column', 'geometry'))
        self.city_widgets['boundary_geometry_column'] = entry
        row += 1
        
        # Boundary WKT column
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="WKT Column:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'boundary_wkt_column')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        entry.insert(0, config.get('boundary_wkt_column', 'the_geom'))
        self.city_widgets['boundary_wkt_column'] = entry
        row += 1
        
        # Save button at bottom
        ctk.CTkButton(scrollable_frame, text="Save City Config", 
                  command=lambda: self.save_city_config(city_name)).grid(
            row=row, column=0, columnspan=3, pady=20)
    
    def browse_file(self, entry_widget):
        """Open file browser and update entry widget."""
        filename = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("GeoJSON files", "*.geojson"),
                ("JSON files", "*.json"),
                ("Shapefile", "*.shp"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            entry_widget.delete(0, "end")
            # Display abbreviated path
            abbreviated = self.abbreviate_path(filename, max_length=60)
            entry_widget.insert(0, abbreviated)
            # Store full path
            entry_widget.full_path = filename
    
    def browse_directory(self, entry_widget):
        """Open directory browser and update entry widget."""
        dirname = filedialog.askdirectory(title="Select Directory")
        if dirname:
            entry_widget.delete(0, "end")
            # Display abbreviated path
            abbreviated = self.abbreviate_path(dirname, max_length=60)
            entry_widget.insert(0, abbreviated)
            # Store full path
            entry_widget.full_path = dirname
    
    def _get_widget_value(self, widget, key):
        """Extract value from a widget based on its type."""
        if isinstance(widget, (ctk.BooleanVar, ctk.StringVar, ctk.DoubleVar)):
            return widget.get()
        elif key == 'census_county_fips':
            # Parse comma-separated list
            value = widget.get().strip()
            return [v.strip() for v in value.split(',') if v.strip()]
        else:
            # For Entry widgets, use full_path if available
            return getattr(widget, 'full_path', widget.get())
    
    def save_city_config(self, city_name):
        """Save current city configuration."""
        config = {key: self._get_widget_value(widget, key) 
                  for key, widget in self.city_widgets.items()}
        
        self.city_configs[city_name] = config
        
        # Don't show success message if called from validation
        if not hasattr(self, '_validating'):
            messagebox.showinfo("Success", f"Configuration saved for {city_name}")
    
    def validate_configuration(self):
        """
        Validate all configuration settings before running analysis.
        Returns a list of error messages (empty if valid).
        """
        errors = []
        
        # Validate each city configuration
        for city_name, config in self.city_configs.items():
            city_errors = []
            
            # Check required file paths
            required_paths = {
                'parcels_path': ('Parcels', ['.geojson', '.json', '.csv']),
                'parks_path': ('Parks', ['.geojson', '.json']),
                'streets_path': ('Streets', ['.geojson', '.json']),
            }
            
            for key, (label, valid_extensions) in required_paths.items():
                path_str = config.get(key, '').strip()
                
                if not path_str:
                    city_errors.append(f"  • {label} file path is required")
                else:
                    path = Path(path_str)
                    
                    # Check if file exists
                    if not path.exists():
                        city_errors.append(f"  • {label} file not found: {path_str}")
                    # Check file extension
                    elif not any(path_str.lower().endswith(ext) for ext in valid_extensions):
                        valid_ext_str = ', '.join(valid_extensions)
                        city_errors.append(
                            f"  • {label} file has invalid extension.\n"
                            f"    Expected: {valid_ext_str}\n"
                            f"    Got: {path.suffix}"
                        )
            
            # Check optional boundary path if provided
            boundary_path = config.get('boundary_path', '').strip()
            if boundary_path:
                path = Path(boundary_path)
                if not path.exists():
                    city_errors.append(f"  • Boundary file not found: {boundary_path}")
                elif not any(boundary_path.lower().endswith(ext) for ext in ['.geojson', '.json', '.csv']):
                    city_errors.append(
                        f"  • Boundary file has invalid extension.\n"
                        f"    Expected: .geojson, .json, or .csv\n"
                        f"    Got: {path.suffix}"
                    )
            
            # Check census configuration if enabled
            if config.get('run_census_analysis', False):
                # Check census API key
                api_key = self.global_settings.get('census_api_key', '').strip()
                if not api_key or api_key == 'YOUR_API_KEY_HERE':
                    city_errors.append(
                        f"  • Census analysis enabled but API key is missing.\n"
                        f"    Get a free key at: https://api.census.gov/data/key_signup.html"
                    )
                
                # Check census state FIPS
                state_fips = config.get('census_state_fips', '').strip()
                if not state_fips:
                    city_errors.append(f"  • Census State FIPS is required when census analysis is enabled")
                
                # Check census county FIPS
                county_fips = config.get('census_county_fips', [])
                if not county_fips or (isinstance(county_fips, list) and len(county_fips) == 0):
                    city_errors.append(f"  • Census County FIPS is required when census analysis is enabled")
                
                # Check census block shapefile
                shapefile_path = config.get('census_block_shapefile', '').strip()
                if not shapefile_path:
                    city_errors.append(f"  • Census Block Shapefile is required when census analysis is enabled")
                else:
                    path = Path(shapefile_path)
                    if not path.exists():
                        city_errors.append(f"  • Census Block Shapefile not found: {shapefile_path}")
                    elif not shapefile_path.lower().endswith('.shp'):
                        city_errors.append(
                            f"  • Census Block Shapefile must be a .shp file.\n"
                            f"    Got: {path.suffix}"
                        )
            
            # Check LTS column name
            lts_column = config.get('lts_column', '').strip()
            if not lts_column:
                city_errors.append(f"  • LTS Column Name is required")
            
            # Add city errors to main error list
            if city_errors:
                errors.append(f"❌ {city_name}:\n" + "\n".join(city_errors))
        
        # Validate global settings
        global_errors = []
        
        # Check exports output directory
        exports_output = self.global_settings.get('exports_output_dir', '').strip()
        if not exports_output:
            global_errors.append("  • Exports Output Directory is required")
        
        # Check census API key if any city has census enabled
        any_census_enabled = any(
            config.get('run_census_analysis', False) 
            for config in self.city_configs.values()
        )
        
        if any_census_enabled:
            # Census data now goes to Exports/Data/{city}/ - no separate setting needed
            pass
        
        if global_errors:
            errors.append("❌ Global Settings:\n" + "\n".join(global_errors))
        
        return errors
    
    def show_global_settings(self):
        """Show global settings configuration window."""
        # Save current city config first
        if self.current_city:
            self.save_city_config(self.current_city)
        
        # Ensure global_settings has all required keys
        for key, value in self.default_global_settings.items():
            if key not in self.global_settings:
                self.global_settings[key] = value
        
        # Create new window
        global_window = ctk.CTkToplevel(self.root)
        global_window.title("Global Settings")
        
        # Set size to fit screen with taskbar - use 90% of screen height to avoid taskbar
        screen_width = global_window.winfo_screenwidth()
        screen_height = global_window.winfo_screenheight()
        
        # Account for taskbar (typically 40-60px on Windows)
        window_width = min(1000, int(screen_width * 0.8))
        window_height = min(850, int(screen_height * 0.85))  # 85% to leave room for taskbar
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = max(0, (screen_height - window_height) // 2 - 30)  # Shift up slightly
        
        global_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        global_window.minsize(800, 600)  # Set minimum size
        
        # Handle window close button (X) - same as cancel
        def on_close():
            global_window.grab_release()
            global_window.destroy()
        
        global_window.protocol("WM_DELETE_WINDOW", on_close)
        
        # Make it modal and bring to front
        global_window.transient(self.root)
        global_window.grab_set()
        global_window.focus_force()
        global_window.lift()
        global_window.attributes('-topmost', True)
        global_window.after(100, lambda: global_window.attributes('-topmost', False))
        
        # Force update to ensure window is ready
        global_window.update_idletasks()
        
        # Create main container frame with padding
        main_container = ctk.CTkFrame(global_window, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title at the top (fixed, not scrollable)
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            title_frame, 
            text="Global Settings", 
            font=("Inter", 18, "bold")
        ).pack()
        
        # Create button frame FIRST (at bottom) so it doesn't get hidden
        button_frame = ctk.CTkFrame(main_container, fg_color=("#2b2b2b", "#2b2b2b"), height=60)
        button_frame.pack(side="bottom", fill="x", pady=(10, 0))
        button_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Define button functions first
        def save_global_settings():
            # This will be populated later with the actual save logic
            pass
        
        def cancel_global_settings():
            global_window.grab_release()
            global_window.destroy()
        
        # Create buttons immediately
        save_btn = ctk.CTkButton(
            button_frame, 
            text="Save & Close", 
            command=save_global_settings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            corner_radius=8
        )
        save_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        cancel_btn = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=cancel_global_settings,
            width=150,
            height=40,
            font=("Inter", 13, "bold"),
            fg_color="transparent",
            border_width=2,
            corner_radius=8
        )
        cancel_btn.pack(side="left", padx=20, pady=10, expand=True)
        
        # Create scrollable frame for settings (takes remaining space)
        scrollable_frame = ctk.CTkScrollableFrame(
            main_container, 
            fg_color="transparent"
        )
        scrollable_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Configure grid column weights for proper expansion
        scrollable_frame.grid_columnconfigure(1, weight=1)
        
        global_widgets = {}
        row = 0
        
        # === ANALYSIS PARAMETERS SECTION ===
        separator = ctk.CTkFrame(scrollable_frame, height=2, fg_color=("#3a3a3a", "#3a3a3a"))
        separator.grid(row=row, column=0, columnspan=3, sticky='ew', pady=10, padx=5)
        row += 1
        
        ctk.CTkLabel(scrollable_frame, text="Analysis Parameters", font=("Inter", 14, "bold")).grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Analysis numeric sliders
        analysis_fields = [
            ('park_buffer', 'Park Buffer (m)', 1.0, 100.0, 1.0),
            ('entrance_tolerance', 'Entrance Tolerance (m)', 1.0, 20.0, 0.5),
            ('entrance_batch_size', 'Entrance Batch Size', 10, 500, 10),
            ('heatmap_resolution', 'Heatmap Resolution (m)', 10, 500, 10),
            ('max_parcels_for_heatmap', 'Max Parcels for Heatmap', 10000, 200000, 10000),
            ('heatmap_neighbors', 'Heatmap Neighbors (IDW)', 1, 20, 1),
            ('heatmap_smoothing', 'Heatmap Smoothing (sigma)', 0, 10, 0.5),
        ]
        
        for key, label, min_val, max_val, resolution in analysis_fields:
            # Label with info button
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            # Get current value, default to min_val if not set
            current_value = self.global_settings.get(key, min_val)
            # Ensure value is within range
            current_value = max(min_val, min(max_val, float(current_value)))
            
            var = ctk.DoubleVar(value=current_value)
            
            slider = ctk.CTkSlider(
                scrollable_frame, 
                from_=min_val, 
                to=max_val,
                variable=var,
                width=300
            )
            slider.grid(row=row, column=1, padx=5, pady=5)
            
            value_label = ctk.CTkLabel(scrollable_frame, text=f"{var.get():.1f}")
            value_label.grid(row=row, column=2, padx=5, pady=5)
            
            # Update label when slider moves
            def update_label(val, lbl=value_label, v=var):
                lbl.configure(text=f"{v.get():.1f}")
            
            var.trace('w', lambda *args, u=update_label: u(None))
            
            global_widgets[key] = var
            row += 1
        
        # === FONT SETTINGS SECTION ===
        separator = ctk.CTkFrame(scrollable_frame, height=2, fg_color=("#3a3a3a", "#3a3a3a"))
        separator.grid(row=row, column=0, columnspan=3, sticky='ew', pady=10, padx=5)
        row += 1
        
        ctk.CTkLabel(scrollable_frame, text="Font Settings", font=("Inter", 14, "bold")).grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Font family dropdown
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Font Family:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'font_family')
        info_btn.pack(side="left", padx=3)
        
        font_var = ctk.StringVar(value=self.global_settings.get('font_family', 'serif'))
        font_combo = ctk.CTkComboBox(
            scrollable_frame,
            values=['serif', 'sans-serif', 'monospace'],
            variable=font_var,
            width=300,
            state="readonly"
        )
        font_combo.set(self.global_settings.get('font_family', 'serif'))
        font_combo.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        global_widgets['font_family'] = font_var
        row += 1
        
        # Font serif list
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Font Serif List (comma-separated):", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'font_serif')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=40)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        font_list = self.global_settings.get('font_serif', ["Times New Roman", "Times", "DejaVu Serif"])
        entry.insert(0, ', '.join(font_list))
        global_widgets['font_serif'] = entry
        row += 1
        
        # Font stretch dropdown
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Font Stretch:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'font_stretch')
        info_btn.pack(side="left", padx=3)
        
        stretch_var = ctk.StringVar(value=self.global_settings.get('font_stretch', 'condensed'))
        stretch_combo = ctk.CTkComboBox(
            scrollable_frame,
            values=['normal', 'condensed', 'expanded'],
            variable=stretch_var,
            width=300,
            state="readonly"
        )
        stretch_combo.set(self.global_settings.get('font_stretch', 'condensed'))
        stretch_combo.grid(row=row, column=1, padx=5, pady=5, columnspan=2)
        global_widgets['font_stretch'] = stretch_var
        row += 1
        
        # Font size slider
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Font Size:", font=("Inter", 12)).pack(side="left")
        info_btn = create_info_button(label_frame, 'font_size')
        info_btn.pack(side="left", padx=3)
        
        current_value = self.global_settings.get('font_size', 10)
        current_value = max(6, min(20, float(current_value)))
        
        font_size_var = ctk.DoubleVar(value=current_value)
        
        font_size_slider = ctk.CTkSlider(
            scrollable_frame, 
            from_=6, 
            to=20,
            variable=font_size_var,
            width=300
        )
        font_size_slider.grid(row=row, column=1, padx=5, pady=5)
        
        font_size_label = ctk.CTkLabel(scrollable_frame, text=f"{font_size_var.get():.1f}")
        font_size_label.grid(row=row, column=2, padx=5, pady=5)
        
        def update_font_size_label(val, lbl=font_size_label, v=font_size_var):
            lbl.configure(text=f"{v.get():.1f}")
        
        font_size_var.trace('w', lambda *args, u=update_font_size_label: u(None))
        
        global_widgets['font_size'] = font_size_var
        row += 1
        
        # === CENSUS SETTINGS SECTION ===
        separator = ctk.CTkFrame(scrollable_frame, height=2, fg_color=("#3a3a3a", "#3a3a3a"))
        separator.grid(row=row, column=0, columnspan=3, sticky='ew', pady=10, padx=5)
        row += 1
        
        ctk.CTkLabel(scrollable_frame, text="Output & Census Settings", font=("Inter", 14, "bold")).grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Census text fields
        # Census API Key (text field only)
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Census API Key:").pack(side="left")
        info_btn = create_info_button(label_frame, 'census_api_key')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, columnspan=2, sticky="ew")
        entry.insert(0, self.global_settings.get('census_api_key', ''))
        
        global_widgets['census_api_key'] = entry
        row += 1
        
        # Exports Output Directory with browse button
        label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(label_frame, text="Exports Output Directory:").pack(side="left")
        info_btn = create_info_button(label_frame, 'exports_output_dir')
        info_btn.pack(side="left", padx=3)
        
        entry = ctk.CTkEntry(scrollable_frame, width=400)
        entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        entry.insert(0, self.global_settings.get('exports_output_dir', 'Exports'))
        
        # Browse button for directory
        browse_btn = ctk.CTkButton(
            scrollable_frame, 
            text="Browse...",
            command=lambda e=entry: self.browse_directory(e),
            width=100,
            height=28,
            corner_radius=6
        )
        browse_btn.grid(row=row, column=2, padx=5, pady=5, sticky="w")
        
        global_widgets['exports_output_dir'] = entry
        row += 1
        
        # Census numeric fields
        census_fields = [
            ('census_acs_year', 'Census ACS Year', 2010, 2030, 1),
            ('census_n_jobs', 'Census Parallel Jobs', 1, 16, 1),
        ]
        
        for key, label, min_val, max_val, resolution in census_fields:
            # Label with info button
            label_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
            label_frame.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            ctk.CTkLabel(label_frame, text=f"{label}:").pack(side="left")
            info_btn = create_info_button(label_frame, key)
            info_btn.pack(side="left", padx=3)
            
            current_value = self.global_settings.get(key, min_val)
            current_value = max(min_val, min(max_val, float(current_value)))
            
            var = ctk.DoubleVar(value=current_value)
            
            slider = ctk.CTkSlider(scrollable_frame, from_=min_val, to=max_val, 
                             variable=var, orientation="horizontal", length=300)
            slider.grid(row=row, column=1, padx=5, pady=5)
            
            value_label = ctk.CTkLabel(scrollable_frame, text=f"{var.get():.1f}")
            value_label.grid(row=row, column=2, padx=5, pady=5)
            
            def update_label(val, lbl=value_label, v=var):
                lbl.configure(text=f"{v.get():.1f}")
            
            var.trace('w', lambda *args, u=update_label: u(None))
            
            global_widgets[key] = var
            row += 1
        
        # Save and Cancel functions - update the button command
        def save_global():
            for key, widget in global_widgets.items():
                if isinstance(widget, (ctk.DoubleVar, ctk.StringVar)):
                    value = widget.get()
                    # Convert to int for integer fields
                    if key in ['entrance_batch_size', 'max_parcels_for_heatmap', 
                              'heatmap_neighbors', 'font_size', 'census_acs_year', 'census_n_jobs']:
                        value = int(value)
                    self.global_settings[key] = value
                elif key == 'font_serif':
                    value = widget.get().strip()
                    self.global_settings[key] = [v.strip() for v in value.split(',') if v.strip()]
                else:
                    self.global_settings[key] = widget.get()
            
            messagebox.showinfo("Success", "Global settings saved")
            global_window.grab_release()
            global_window.destroy()
        
        # Update the save button command now that we have the actual function
        save_btn.configure(command=save_global)
    
    def validate_and_show_results(self):
        """Validate configuration and show results to user."""
        # Save current city config first (suppress success message)
        if self.current_city:
            self._validating = True
            self.save_city_config(self.current_city)
            delattr(self, '_validating')
        
        # Standardize file names if requested
        if hasattr(self, 'standardize_names_var') and self.standardize_names_var.get():
            try:
                self.standardize_file_names()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to standardize file names: {e}")
                return
        
        validation_errors = self.validate_configuration()
        
        if validation_errors:
            error_msg = "Configuration validation found issues:\n\n" + "\n\n".join(validation_errors)
            messagebox.showwarning("Validation Issues", error_msg)
        else:
            messagebox.showinfo(
                "Validation Passed", 
                "✓ All configuration settings are valid!\n\n"
                "You can now save and run the analysis."
            )
    
    def save_and_run(self):
        """Save configuration to ParkximityCalc.py and run analysis."""
        # Save current city config
        if self.current_city:
            self.save_city_config(self.current_city)
        
        # Standardize file names if requested
        if hasattr(self, 'standardize_names_var') and self.standardize_names_var.get():
            try:
                self.standardize_file_names()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to standardize file names: {e}")
                return
        
        # Validate configuration before proceeding
        validation_errors = self.validate_configuration()
        
        if validation_errors:
            # Show validation errors
            error_msg = "Configuration validation failed:\n\n" + "\n\n".join(validation_errors)
            messagebox.showerror("Validation Error", error_msg)
            return
        
        # Confirm with user
        result = messagebox.askyesno(
            "Confirm", 
            "This will update ParkximityCalc.py and run the analysis. Continue?"
        )
        
        if not result:
            return
        
        try:
            self.write_config_to_file()
            messagebox.showinfo("Success", "Configuration saved to ParkximityCalc.py")
            
            # Ask if user wants to run now
            run_now = messagebox.askyesno(
                "Run Analysis", 
                "Configuration saved. Run analysis now?"
            )
            
            if run_now:
                self.run_analysis()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def standardize_file_names(self):
        """
        Standardize raw data file names to {city}_{layer}_{date}.{extension} format.
        Renames files to standardized names and updates config paths.
        """
        from datetime import datetime
        import shutil
        import os
        
        # Get current date in M.D.YY format
        today = datetime.now()
        date_str = f"{today.month}.{today.day}.{today.year % 100}"
        
        # Layer mapping
        layer_map = {
            'parcels_path': 'parcels',
            'parks_path': 'parks',
            'streets_path': 'streets',
            'boundary_path': 'boundary',
            'census_block_shapefile': 'census_blocks',
        }
        
        renamed_files = []
        errors = []
        
        for city_name, config in self.city_configs.items():
            city_lower = city_name.lower().replace(' ', '_')
            
            for config_key, layer_name in layer_map.items():
                file_path_str = config.get(config_key, '').strip()
                
                if not file_path_str:
                    continue  # Skip empty paths
                
                file_path = Path(file_path_str)
                
                if not file_path.exists():
                    continue  # Skip non-existent files (will be caught by validation)
                
                # Get extension
                extension = file_path.suffix
                
                # Handle shapefiles specially (rename all associated files)
                if extension == '.shp':
                    # Standardized base name
                    new_base_name = f"{city_lower}_{layer_name}_{date_str}"
                    new_dir = file_path.parent
                    new_shp_path = new_dir / f"{new_base_name}.shp"
                    
                    # Check if already standardized
                    if file_path.stem == new_base_name:
                        continue  # Already standardized
                    
                    # Rename all shapefile components
                    shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml', '.shp.xml', '.shp.ea.iso.xml', '.shp.iso.xml']
                    try:
                        for ext in shapefile_extensions:
                            src = file_path.parent / f"{file_path.stem}{ext}"
                            if src.exists():
                                dst = new_dir / f"{new_base_name}{ext}"
                                os.rename(src, dst)
                        
                        # Update config
                        config[config_key] = str(new_shp_path)
                        renamed_files.append(f"{city_name}: {file_path.name} → {new_shp_path.name}")
                    except Exception as e:
                        errors.append(f"{city_name} - {layer_name}: {e}")
                
                else:
                    # Regular file (GeoJSON, JSON, CSV)
                    new_name = f"{city_lower}_{layer_name}_{date_str}{extension}"
                    new_path = file_path.parent / new_name
                    
                    # Check if already standardized
                    if file_path.name == new_name:
                        continue  # Already standardized
                    
                    # Rename file
                    try:
                        os.rename(file_path, new_path)
                        
                        # Update config
                        config[config_key] = str(new_path)
                        renamed_files.append(f"{city_name}: {file_path.name} → {new_name}")
                    except Exception as e:
                        errors.append(f"{city_name} - {layer_name}: {e}")
        
        # Refresh the UI to show updated paths
        if self.current_city and renamed_files:
            self.show_city_config(self.current_city)
        
        # Show results
        if renamed_files or errors:
            result_msg = ""
            
            if renamed_files:
                result_msg += f"Renamed {len(renamed_files)} files:\n\n"
                for item in renamed_files[:10]:  # Show first 10
                    result_msg += f"  ✓ {item}\n"
                if len(renamed_files) > 10:
                    result_msg += f"  ... and {len(renamed_files) - 10} more\n"
                result_msg += "\n✓ Configuration paths updated automatically."
            
            if errors:
                result_msg += f"\n\nErrors ({len(errors)}):\n"
                for error in errors[:5]:  # Show first 5
                    result_msg += f"  ✗ {error}\n"
                if len(errors) > 5:
                    result_msg += f"  ... and {len(errors) - 5} more\n"
            
            if errors:
                messagebox.showwarning("Standardization Complete with Errors", result_msg)
            else:
                messagebox.showinfo("Files Renamed", result_msg)
        else:
            messagebox.showinfo("No Changes", "All files are already using standardized names.")
    
    def write_config_to_file(self):
        """Write configuration back to ParkximityCalc.py."""
        calc_file = Path("Notebooks/Publication/ParkximityCalc.py")
        
        if not calc_file.exists():
            raise FileNotFoundError("ParkximityCalc.py not found")
        
        # Read the file
        with open(calc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the if __name__ == "__main__": section
        main_start = content.find('if __name__ == "__main__":')
        if main_start == -1:
            raise ValueError("Could not find main section in ParkximityCalc.py")
        
        # Generate new config code
        config_code = self.generate_config_code()
        
        # Replace everything after if __name__ == "__main__":
        new_content = content[:main_start] + config_code
        
        # Write back
        with open(calc_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    def generate_config_code(self):
        """Generate Python code for configuration."""
        lines = ['if __name__ == "__main__":\n']
        lines.append('    # Global visualization settings applied to all cities\n')
        lines.append('    global_settings = {\n')
        
        # Write global settings
        for key, value in self.global_settings.items():
            if isinstance(value, str):
                lines.append(f'        "{key}": "{value}",\n')
            elif isinstance(value, list):
                # Format list nicely
                list_str = ', '.join([f'"{v}"' for v in value])
                lines.append(f'        "{key}": [{list_str}],\n')
            else:
                lines.append(f'        "{key}": {value},\n')
        
        lines.append('    }\n\n')
        
        # Write city configs
        lines.append('    cities_config = {\n')
        
        for city_name, config in self.city_configs.items():
            lines.append(f'        "{city_name}": {{\n')
            
            for key, value in config.items():
                if isinstance(value, str):
                    lines.append(f'            "{key}": "{value}",\n')
                elif isinstance(value, bool):
                    lines.append(f'            "{key}": {value},\n')
                elif isinstance(value, list):
                    list_str = ', '.join([f'"{v}"' for v in value])
                    lines.append(f'            "{key}": [{list_str}],\n')
                else:
                    lines.append(f'            "{key}": {value},\n')
            
            lines.append('        },\n')
        
        lines.append('    }\n\n')
        
        # Add execution code
        lines.append('    parkximity_results, census_results = multi_city_analysis(cities_config, global_settings=global_settings)\n\n')
        lines.append('    for city_name, parcels_gdf in parkximity_results.items():\n')
        lines.append('        print(f"\\n{city_name} Results:")\n')
        lines.append('        print(f"  Total parcels: {len(parcels_gdf)}")\n')
        lines.append('        valid = parcels_gdf[parcels_gdf["park_distance"] < float("inf")]\n')
        lines.append('        if len(valid) > 0:\n')
        lines.append('            print(f"  Mean parkximity: {valid[\'park_distance\'].mean():.2f} m")\n')
        lines.append('            print(f"  Median parkximity: {valid[\'park_distance\'].median():.2f} m")\n')
        
        return ''.join(lines)
    
    def run_analysis(self):
        """Run the parkximity analysis."""
        try:
            # Run ParkximityCalc.py
            calc_file = Path("Notebooks/Publication/ParkximityCalc.py")
            
            messagebox.showinfo("Running", "Starting analysis... This may take a while. Check console for progress.")
            
            # Run in subprocess
            subprocess.Popen([sys.executable, str(calc_file)])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run analysis: {e}")


def main():
    """Main entry point for the UI."""
    root = ctk.CTk()
    app = ParkximityConfigUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
