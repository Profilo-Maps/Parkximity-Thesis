ParkXimityAnalyzer Class:
    def __init__(self, city_name, config, target_crs):
        Config Parameters:
        city_name : str
            Name of the city being analyzed
        config : dict
            Configuration dictionary containing:
                - parcels_path
                - parks_path
                - streets_path
                - boundary_path (optional)
                - combine_boundaries_toggle: Boolean, if True combines neighborhood polygons into city boundary (optional)
                - output_dir: Directory for output files
    Methods:
    def load_data():Should load parcels, parks, streets, and boundaries data to prepare it for conversion into a graph. Should detect whether source files are csv or geojson, and should have 2 configurations for detecting geometry accordingly. All sources should have their crs reprojected to the target_crs
        def _clip_data_to_boundary(): if city boundary is specified, clips parcels and parks to city boundary
        def _combine_boundaries(): if combine boundaries is on, merges individual neighborhood boundaries into one larger polygon
    def generate_park_entrances():find locations where streets network intersects with park boundaries and marks them as park entrance nodes. 
        def _batch_park_entrances(): batches entrance calculations
        def _extract_points_from_intersection(): finds streetnetwork and park boundary intersections
        def _vectorized_deduplication_park_entrances(): ensures park entrances are unique
    def build_network(): builds graph for analysis
    def find_parkximity(): creates json dataset of parks, parcels, and the calculated routes between them. Each parcel should have a parkximity (LTS weighted distance to park) value stored. Each street segment should have an LTS value. Each Park should have a list of parcels it is the closest to. Total parkximity should be the distance from a parcel to the nearest network node added to the distance from that nearest network node to the nearest park. 
        def _assign_parcels_to_nodes(): finds the distance between parcels and the nearest network node to be added to the Dijkstra's distance to ensure true distance is calculated. 
        def _run_dijkstra(): runs custom Dijkstra's on park, parcel, and streets network. For tiebreak logic, whatever street has the lowest cumulative LTS score is selected. 
    def create_visualizations(): Uses Parkximity data to create heatmaps and histograms
        def _calculate_density_cv(): Coeffient of Variation = std(parcels_per_cell) / mean(parcels_per_cell)
        def _calculate_adaptive_alpha(): 
        CV:    0.0    0.5    1.0    1.5    2.0    2.5+
        Alpha: 1.0    0.85   0.7    0.55   0.4    0.3
        def _stratified_spatial_sample(): samples_per_cell = (parcels_in_cell)^α
        def _create_heatmap(): Colors each parcel according to its stored parkximity value and then applies a gaussian blur to create a smooth distance gradient across the city. Parks should be colored according to the number of parcels that they are the most parkximate to. GPU-accelerated version uses CuPy for IDW interpolation and Gaussian smoothing (15-40x speedup). 
        def _create_distance_histograms(): Creates histograms showing the bin distributions of parkximity distances from parks and of the number of parcels that a park is closest to. 
    def full_analysis(): Loads data, generates graph, calculates parkximities, and creates visualizations

def multi_city_analysis(cities_config, target_crs): initializes analysis for multiple cities in one config file. 