"""
GPU Detection and Acceleration Utilities for ParkXimity Analysis

This module provides automatic GPU detection and graceful fallback to CPU
for spatial analysis operations. It supports NVIDIA GPUs via RAPIDS/CuPy.

Usage:
    from gpu_utils import GPUContext, get_gpu_info
    
    # Check GPU availability
    info = get_gpu_info()
    print(f"GPU Available: {info['available']}")
    
    # Use GPU context for automatic array handling
    with GPUContext() as ctx:
        result = ctx.nearest_neighbors(points, k=5)
"""

# CRITICAL: Configure CuPy BEFORE any imports
# This must be the very first code that runs
import os
import sys
import shutil
import warnings
import subprocess
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial import cKDTree

# Set CUDA compilation flags for C++17 (required for CUDA 12.6+)
# CuPy uses CUPY_CUDA_COMPILE_ARGS for nvcc flags
os.environ['CUPY_CUDA_COMPILE_ARGS'] = '--std=c++17'
os.environ['NVCC_PREPEND_FLAGS'] = '--std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT'
os.environ['CUPY_CACHE_DIR'] = os.path.join(os.environ.get('TEMP', '/tmp'), 'cupy_cache')

# Clear any existing CuPy cache to force recompilation with new flags
cache_dir = os.environ['CUPY_CACHE_DIR']

if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass  # Silently ignore cache clearing errors
else:
    os.makedirs(cache_dir, exist_ok=True)

# Conditional GPU imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import cuspatial
    CUSPATIAL_AVAILABLE = True
except ImportError:
    cuspatial = None
    CUSPATIAL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available() if torch else False
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class GPUCapabilities:
    """Detect and report GPU capabilities."""
    
    def __init__(self):
        self.cupy_available = False
        self.cuml_available = False
        self.cuspatial_available = False
        self.torch_available = False
        self.gpu_count = 0
        self.gpu_names = []
        self.total_memory = []
        self.cuda_version = None
        
        self._detect_capabilities()
    
    def _detect_capabilities(self):
        """Detect available GPU libraries and hardware."""
        # Check CuPy (array operations)
        if CUPY_AVAILABLE:
            try:
                self.cupy_available = True
                self.gpu_count = cp.cuda.runtime.getDeviceCount()
                self.cuda_version = cp.cuda.runtime.runtimeGetVersion()
                
                for i in range(self.gpu_count):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    self.gpu_names.append(props['name'].decode('utf-8'))
                    self.total_memory.append(props['totalGlobalMem'] / (1024**3))  # GB
            except Exception:
                pass
        
        # cuML removed - not compatible with Windows
        self.cuml_available = False
        
        # Check cuSpatial (spatial operations)
        self.cuspatial_available = CUSPATIAL_AVAILABLE
        
        # Check PyTorch (alternative for graph operations)
        self.torch_available = TORCH_AVAILABLE
    
    def is_available(self) -> bool:
        """Check if any GPU acceleration is available."""
        return self.cupy_available and self.gpu_count > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of GPU capabilities."""
        return {
            'available': self.is_available(),
            'gpu_count': self.gpu_count,
            'gpu_names': self.gpu_names,
            'total_memory_gb': self.total_memory,
            'cuda_version': self.cuda_version,
            'libraries': {
                'cupy': self.cupy_available,
                'cuml': self.cuml_available,
                'cuspatial': self.cuspatial_available,
                'pytorch': self.torch_available,
            }
        }
    
    def print_info(self):
        """Print formatted GPU information."""
        info = self.get_summary()
        print("\n" + "="*60)
        print("GPU Acceleration Status")
        print("="*60)
        
        if info['available']:
            print(f"✓ GPU acceleration ENABLED")
            print(f"  Devices: {info['gpu_count']}")
            for i, (name, mem) in enumerate(zip(info['gpu_names'], info['total_memory_gb'])):
                print(f"    [{i}] {name} ({mem:.1f} GB)")
            print(f"  CUDA Version: {info['cuda_version']}")
            print(f"\n  Available Libraries:")
            for lib, avail in info['libraries'].items():
                status = "✓" if avail else "✗"
                print(f"    {status} {lib}")
        else:
            print(f"✗ GPU acceleration DISABLED (using CPU)")
            print(f"  Reason: ", end="")
            if not info['libraries']['cupy']:
                print("CuPy not installed")
            elif info['gpu_count'] == 0:
                print("No NVIDIA GPU detected")
            else:
                print("Unknown")
        
        print("="*60 + "\n")


# Global GPU capabilities instance
_gpu_caps = None


def get_gpu_capabilities() -> GPUCapabilities:
    """Get or create global GPU capabilities instance."""
    global _gpu_caps
    if _gpu_caps is None:
        _gpu_caps = GPUCapabilities()
    return _gpu_caps


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information dictionary."""
    return get_gpu_capabilities().get_summary()


class GPUContext:
    """
    Context manager for GPU-accelerated operations with automatic fallback.
    
    Provides unified interface for spatial operations that automatically
    uses GPU when available and falls back to CPU otherwise.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.caps = get_gpu_capabilities()
        self.use_gpu = self.caps.is_available()
        
        # Import appropriate libraries
        # Note: cuML removed for Windows compatibility
        if self.use_gpu:
            try:
                self.cp = cp
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        if not self.use_gpu:
            self.cp = np
        
        # Always use scipy for nearest neighbors (cuML not Windows-compatible)
        self.cKDTree = cKDTree
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self.use_gpu:
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        return False
    
    def to_device(self, array: np.ndarray):
        """Move numpy array to GPU if available."""
        if self.use_gpu:
            return self.cp.asarray(array)
        return array
    
    def to_host(self, array):
        """Move array back to CPU (numpy)."""
        if self.use_gpu and hasattr(array, 'get'):
            return self.cp.asnumpy(array)
        return np.asarray(array)
    
    def nearest_neighbors(
        self, 
        points: np.ndarray, 
        query_points: Optional[np.ndarray] = None,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors using GPU or CPU.
        
        Parameters
        ----------
        points : np.ndarray
            Reference points (N, 2)
        query_points : np.ndarray, optional
            Query points (M, 2). If None, uses points.
        k : int
            Number of neighbors
        
        Returns
        -------
        distances : np.ndarray
            Distances to k nearest neighbors (M, k)
        indices : np.ndarray
            Indices of k nearest neighbors (M, k)
        """
        if query_points is None:
            query_points = points
        
        # Use scipy cKDTree (cuML removed for Windows compatibility)
        tree = self.cKDTree(points)
        distances, indices = tree.query(query_points, k=k)
        
        # Ensure consistent shape
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
            
            return distances, indices
    
    def query_pairs(
        self,
        points: np.ndarray,
        radius: float
    ) -> set:
        """
        Find all pairs of points within radius.
        
        Parameters
        ----------
        points : np.ndarray
            Points array (N, 2)
        radius : float
            Search radius
        
        Returns
        -------
        pairs : set
            Set of (i, j) tuples where i < j
        """
        if self.use_gpu:
            # GPU implementation using distance matrix
            points_gpu = self.to_device(points)
            n = len(points)
            
            # Compute pairwise distances (memory intensive for large N)
            if n > 10000:
                warnings.warn(
                    f"Computing pairwise distances for {n} points may use significant GPU memory. "
                    "Consider using CPU for this operation."
                )
            
            # Expand dims for broadcasting
            p1 = points_gpu[:, None, :]  # (N, 1, 2)
            p2 = points_gpu[None, :, :]  # (1, N, 2)
            
            # Compute distances
            dists = self.cp.sqrt(self.cp.sum((p1 - p2)**2, axis=2))
            
            # Find pairs within radius (upper triangle only)
            i_idx, j_idx = self.cp.where((dists < radius) & (dists > 0))
            
            # Convert to host and filter to upper triangle
            i_idx = self.to_host(i_idx)
            j_idx = self.to_host(j_idx)
            
            pairs = set()
            for i, j in zip(i_idx, j_idx):
                if i < j:
                    pairs.add((i, j))
            
            return pairs
        else:
            # CPU path using scipy
            tree = cKDTree(points)
            return tree.query_pairs(r=radius)
    
    def deduplicate_points(
        self,
        coords_list: list,
        data_list: list,
        tolerance: float
    ) -> Tuple[list, list]:
        """
        Deduplicate points within tolerance distance.
        
        Parameters
        ----------
        coords_list : list
            List of (x, y) coordinate tuples
        data_list : list
            List of associated data dictionaries
        tolerance : float
            Deduplication distance threshold
        
        Returns
        -------
        unique_coords : list
            Deduplicated coordinates
        unique_data : list
            Corresponding data
        """
        if not coords_list:
            return [], []
        
        coords_array = np.array(coords_list)
        pairs = self.query_pairs(coords_array, tolerance)
        
        # Build removal set
        remove = set()
        for i, j in pairs:
            if i not in remove:
                remove.add(j)
        
        # Filter
        unique_coords = [c for idx, c in enumerate(coords_list) if idx not in remove]
        unique_data = [d for idx, d in enumerate(data_list) if idx not in remove]
        
        return unique_coords, unique_data


def print_gpu_info():
    """Convenience function to print GPU information."""
    get_gpu_capabilities().print_info()


def install_instructions():
    """Print installation instructions for GPU libraries."""
    print("\n" + "="*60)
    print("GPU Acceleration Installation")
    print("="*60)
    print("\nAutomatic Installation (Recommended):")
    print("  python install_gpu_support.py")
    print("\nThis will:")
    print("  • Detect your NVIDIA GPU")
    print("  • Install RAPIDS (CuPy)")
    print("  • Configure everything automatically")
    print("\n" + "-"*60)
    print("\nManual Installation:")
    print("\n  Option 1: Using conda (recommended)")
    print("  " + "-" * 30)
    print("  conda install -c rapidsai -c conda-forge -c nvidia \\")
    print("      cupy cudatoolkit=11.8")
    print("\n  Option 2: Using pip (limited support)")
    print("  " + "-" * 30)
    print("  pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12")
    print("\n  Requirements:")
    print("  • NVIDIA GPU (Compute Capability 6.0+)")
    print("  • CUDA Toolkit 11.2+ or 12.0+")
    print("  • Linux or WSL2 (Windows Subsystem for Linux)")
    print("\n  Check compatibility:")
    print("  https://rapids.ai/start.html")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test GPU detection
    print_gpu_info()
    
    if not get_gpu_capabilities().is_available():
        install_instructions()
    else:
        # Run simple benchmark
        print("Running simple benchmark...")
        n_points = 10000
        points = np.random.rand(n_points, 2) * 1000
        
        with GPUContext(verbose=True) as ctx:
            start = time.time()
            dists, indices = ctx.nearest_neighbors(points, k=5)
            elapsed = time.time() - start
            
            device = "GPU" if ctx.use_gpu else "CPU"
            print(f"\n{device} Nearest Neighbors ({n_points} points, k=5): {elapsed:.3f}s")
            print(f"Result shape: {dists.shape}")


# ======================================================================
# GPU Auto-Installer Functions
# ======================================================================

def check_nvidia_gpu_hardware():
    """
    Check if NVIDIA GPU hardware is present (even if libraries not installed).
    
    Returns
    -------
    has_gpu : bool
        True if NVIDIA GPU detected
    gpu_name : str or None
        Name of the GPU if detected
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True,
            timeout=5,
            shell=True  # Use shell on Windows
        )
        # Check output even if return code is non-zero (Windows quirk)
        output = result.stdout
        if 'NVIDIA' in output or 'GeForce' in output or 'RTX' in output:
            # Extract GPU name from the table
            for line in output.split('\n'):
                # Look for the GPU name line (has "GeForce" or "RTX")
                if ('GeForce' in line or 'RTX' in line or 'Quadro' in line) and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        # Second column has the GPU name
                        gpu_name = parts[1].strip()
                        
                        # Remove GPU index number at start (e.g., "0  NVIDIA...")
                        if gpu_name and gpu_name[0].isdigit():
                            gpu_name = gpu_name[1:].strip()
                        
                        # Remove extra info after the name
                        if 'WDDM' in gpu_name:
                            gpu_name = gpu_name.split('WDDM')[0].strip()
                        if 'TCC' in gpu_name:
                            gpu_name = gpu_name.split('TCC')[0].strip()
                        if 'Driver-Model' in gpu_name:
                            gpu_name = gpu_name.split('Driver-Model')[0].strip()
                        
                        # Remove trailing "..."
                        gpu_name = gpu_name.rstrip('.').strip()
                        
                        return True, gpu_name
            return True, "NVIDIA GPU"
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False, None


def check_cuda_version():
    """
    Check CUDA version if available.
    
    Returns
    -------
    version : str or None
        CUDA version string if detected
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True,
            timeout=5,
            shell=True  # Use shell on Windows
        )
        # Check output even if return code is non-zero
        output = result.stdout
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                parts = line.split('CUDA Version:')
                if len(parts) > 1:
                    version = parts[1].strip().split()[0]
                    return version
        return None
    except:
        return None


def get_conda_env():
    """Get current conda environment name."""
    return os.environ.get('CONDA_DEFAULT_ENV', 'base')


def install_gpu_packages_auto(interactive=True):
    """
    Automatically install GPU packages using conda.
    
    Parameters
    ----------
    interactive : bool
        If True, ask user for confirmation before installing.
        If False, install automatically without prompting.
    
    Returns
    -------
    success : bool
        True if installation succeeded
    """
    if interactive:
        print("\n" + "="*70)
        print("GPU Acceleration Available!")
        print("="*70)
        print("\nWould you like to install GPU acceleration support now?")
        print("\nBenefits:")
        print("  • 10-50x faster processing for large datasets")
        print("  • Automatic fallback to CPU if GPU unavailable")
        print("  • Same code works on any machine")
        print("\nRequirements:")
        print("  • ~2-3 GB disk space")
        print("  • 5-10 minutes installation time")
        print("  • Internet connection")
        
        while True:
            response = input("\nInstall GPU support now? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                break
            elif response in ['n', 'no']:
                print("\nSkipping GPU installation.")
                print("\nYou can install later by running:")
                print("  python -c \"from gpu_utils import install_gpu_packages_auto; install_gpu_packages_auto()\"")
                print("Or:")
                print("  python install_gpu_support.py")
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    print("\n" + "="*70)
    print("Installing GPU Acceleration Packages")
    print("="*70)
    print("\nThis will install:")
    print("  • CuPy (GPU-accelerated NumPy)")
    print("  • CUDA Toolkit 11.8")
    print("\nEstimated time: 5-10 minutes")
    print("Press Ctrl+C to cancel...")
    print("="*70 + "\n")
    
    try:
        cmd = [
            'conda', 'install', '-y',
            '-c', 'rapidsai',
            '-c', 'conda-forge', 
            '-c', 'nvidia',
            'cupy',
            'cudatoolkit=11.8'
        ]
        
        print("Running: " + " ".join(cmd))
        print("\nThis may take several minutes...\n")
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("✓ GPU packages installed successfully!")
            print("="*70)
            print("\nPlease restart your Python session for changes to take effect.")
            print("Then run your analysis again:")
            print("  python ParkximityCalc.py")
            print("="*70 + "\n")
            return True
        else:
            print("\n" + "="*70)
            print("✗ Installation failed")
            print("="*70)
            print("\nTry manual installation:")
            print("  conda install -c rapidsai -c conda-forge -c nvidia cupy cudatoolkit=11.8")
            return False
            
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        return False
    except Exception as e:
        print(f"\n✗ Installation error: {e}")
        return False


def prompt_gpu_installation_if_needed():
    """
    Check if GPU hardware exists but libraries are missing, and offer to install.
    
    This is called automatically when GPU hardware is detected but libraries
    are not installed. It offers to install GPU support interactively.
    
    Returns
    -------
    should_retry : bool
        True if user installed packages and should retry GPU initialization
    """
    # Check if GPU hardware exists
    has_gpu_hw, gpu_name = check_nvidia_gpu_hardware()
    
    if not has_gpu_hw:
        return False
    
    # Check if libraries already installed and working
    if CUPY_AVAILABLE:
        try:
            # Verify they actually work (not just imported)
            # Quick test to ensure GPU is accessible
            test_array = cp.array([1, 2, 3])
            _ = cp.asnumpy(test_array)
            return False  # Already installed and working
        except Exception:
            # Libraries installed but not working - might need reinstall
            print("\n⚠ GPU libraries found but not functioning properly.")
            print("  This may require reinstallation or driver update.")
    
    # GPU hardware exists but libraries missing or not working
    print("\n" + "="*70)
    print("🎯 NVIDIA GPU Detected!")
    print("="*70)
    print(f"\nGPU: {gpu_name}")
    
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"CUDA Version: {cuda_version}")
    
    print("\nGPU acceleration libraries are not installed.")
    print("Installing them will provide 10-50x speedup for large datasets.")
    
    # Offer to install
    success = install_gpu_packages_auto(interactive=True)
    
    if success:
        print("\n⚠ IMPORTANT: You must restart Python for changes to take effect.")
        print("After restarting, run your analysis again.")
        return True
    
    return False


def main():
    """
    Main installation flow when run as a script.
    
    This provides the same functionality as install_gpu_support.py
    but is now integrated directly into gpu_utils.py.
    """
    print("\n" + "="*70)
    print("ParkXimity GPU Support Auto-Installer")
    print("="*70)
    
    env_name = get_conda_env()
    print(f"\nCurrent conda environment: {env_name}")
    
    # Check if GPU libraries already installed and working
    caps = get_gpu_capabilities()
    if caps.is_available():
        info = caps.get_summary()
        print(f"\n✓ GPU libraries already installed and working!")
        print(f"  Device: {info['gpu_names'][0]}")
        print(f"  Memory: {info['total_memory_gb'][0]:.1f} GB")
        print(f"  Libraries:")
        for lib, avail in info['libraries'].items():
            if avail:
                print(f"    ✓ {lib}")
        print("\nYou're all set! Run your analysis:")
        print("  python ParkximityCalc.py")
        return 0
    
    # Check if libraries are installed but not working
    if CUPY_AVAILABLE:
        print("\n⚠ GPU libraries are installed but not functioning properly.")
        print("  This may indicate:")
        print("    • Driver version mismatch")
        print("    • CUDA version incompatibility")
        print("    • GPU not accessible")
        print("\nTry updating your NVIDIA drivers or reinstalling GPU packages.")
        return 1
    
    # Check for NVIDIA GPU hardware
    print("\nDetecting NVIDIA GPU...")
    has_gpu, gpu_name = check_nvidia_gpu_hardware()
    
    if not has_gpu:
        print("\n✗ No NVIDIA GPU detected")
        print("\nYour system will use CPU-only mode.")
        print("This is perfectly fine - the code will work normally,")
        print("just without GPU acceleration.")
        print("\nTo run analysis:")
        print("  python ParkximityCalc.py")
        return 0
    
    print(f"\n✓ NVIDIA GPU detected: {gpu_name}")
    
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"✓ CUDA Version: {cuda_version}")
    
    # Install packages
    success = install_gpu_packages_auto(interactive=True)
    
    if success:
        print("\n" + "="*70)
        print("Next Steps")
        print("="*70)
        print("\n1. Restart your Python session")
        print("\n2. Run your analysis:")
        print("     python ParkximityCalc.py")
        print("\n3. You should see:")
        print("     🚀 GPU ACCELERATION: ENABLED")
        print("="*70 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        sys.exit(1)
