"""
Quick test script to verify ParkXimity environment setup.

Run this after creating your conda environment to check that
all dependencies are installed correctly.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 60)
    
    packages = {
        'Core Python': ['sys', 'json', 'pathlib'],
        'Data Processing': ['pandas', 'numpy'],
        'Geospatial': ['geopandas', 'shapely', 'pyproj', 'fiona'],
        'Scientific': ['scipy', 'networkx'],
        'Visualization': ['matplotlib'],
        'GPU (optional)': ['cupy'],
    }
    
    results = {}
    
    for category, pkg_list in packages.items():
        print(f"\n{category}:")
        for pkg in pkg_list:
            try:
                if pkg == 'cupy':
                    import cupy as cp
                    version = cp.__version__
                else:
                    mod = __import__(pkg)
                    version = getattr(mod, '__version__', 'unknown')
                
                print(f"  ✓ {pkg:20s} {version}")
                results[pkg] = True
            except ImportError:
                if category == 'GPU (optional)':
                    print(f"  ○ {pkg:20s} (not installed - GPU mode disabled)")
                else:
                    print(f"  ✗ {pkg:20s} MISSING!")
                results[pkg] = False
    
    return results


def test_gpu():
    """Test GPU availability and capabilities."""
    print("\n" + "=" * 60)
    print("GPU Detection Test")
    print("=" * 60)
    
    try:
        from gpu_utils import print_gpu_info
        print_gpu_info()
        return True
    except ImportError:
        print("⚠ gpu_utils.py not found")
        print("  This is normal if you haven't copied the GPU files yet")
        return False
    except Exception as e:
        print(f"⚠ GPU detection failed: {e}")
        return False


def test_geospatial():
    """Test basic geospatial operations."""
    print("\n" + "=" * 60)
    print("Geospatial Operations Test")
    print("=" * 60)
    
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import numpy as np
        
        # Create simple test data
        points = [Point(x, y) for x, y in zip(np.random.rand(5), np.random.rand(5))]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        gdf_projected = gdf.to_crs("EPSG:6350")
        
        print(f"✓ Created GeoDataFrame with {len(gdf)} points")
        print(f"✓ Reprojected from EPSG:4326 to EPSG:6350")
        print(f"✓ Geospatial operations working correctly")
        return True
    except Exception as e:
        print(f"✗ Geospatial test failed: {e}")
        return False


def test_network():
    """Test network analysis capabilities."""
    print("\n" + "=" * 60)
    print("Network Analysis Test")
    print("=" * 60)
    
    try:
        import networkx as nx
        
        # Create simple test graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        path = nx.shortest_path(G, 1, 3)
        
        print(f"✓ Created graph with {G.number_of_nodes()} nodes")
        print(f"✓ Found shortest path: {path}")
        print(f"✓ Network analysis working correctly")
        return True
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PARKXIMITY ENVIRONMENT TEST")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Run tests
    import_results = test_imports()
    gpu_available = test_gpu()
    geospatial_ok = test_geospatial()
    network_ok = test_network()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    required_packages = ['pandas', 'numpy', 'geopandas', 'shapely', 
                        'scipy', 'networkx', 'matplotlib']
    all_required = all(import_results.get(pkg, False) for pkg in required_packages)
    
    if all_required and geospatial_ok and network_ok:
        print("✓ All required packages installed correctly")
        print("✓ Core functionality working")
        
        if gpu_available and import_results.get('cupy', False):
            print("✓ GPU acceleration available")
            print("\n🚀 Your environment is ready for GPU-accelerated analysis!")
        else:
            print("○ GPU acceleration not available (CPU mode only)")
            print("\n💻 Your environment is ready for CPU-based analysis!")
        
        print("\nNext steps:")
        print("  1. Run: python Notebooks/Karna/ParkximityCalc.py")
        print("  2. Check the output for GPU/CPU mode confirmation")
        
        return 0
    else:
        print("✗ Some required packages are missing")
        print("\nPlease install missing packages:")
        print("  conda install -c conda-forge <package-name>")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
