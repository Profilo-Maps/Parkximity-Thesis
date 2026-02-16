# ParkXimity

Evaluating LTS-weighted distance from parcels to parks with optional GPU acceleration.

## 📁 Publication-Ready Code

All publication-ready code has been moved to:

```
Notebooks/Publication/
```

**To get started:**

1. Navigate to the publication folder:
   ```bash
   cd Notebooks/Publication
   ```

2. Follow the instructions in `Notebooks/Publication/README.md`

## Quick Start

```bash
# Navigate to publication folder
cd Notebooks/Publication

# Create environment
conda env create -f environment.yml
conda activate parkximity

# Run analysis
python ParkximityCalc.py
```

## What's in This Repository

- `Notebooks/Publication/` - **Publication-ready code** (start here!)
- `Notebooks/Karna/` - Development notebooks
- `Notebooks/Alex/` - Development notebooks
- `Data/` - Input data files (not included in publication)
- `Visualizations/` - Output visualizations

## Documentation

See `Notebooks/Publication/README.md` for complete documentation including:
- Installation instructions
- GPU acceleration setup
- Usage examples
- Troubleshooting
- API reference

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
