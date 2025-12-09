"""
PCA Analysis for Parcel Data with Demographics and Park Proximity
=================================================================
This script analyzes relationships between variables in parcel GeoJSON files
containing park distance and census demographic data.

Usage:
    python parcel_pca_analysis.py <input_geojson> [--output-dir <dir>]
    python parcel_pca_analysis.py Data/Processed/Demographics/boston_parcels_with_demographics.geojson
    python parcel_pca_analysis.py "Data/Processed/Demographics/*.geojson" --output-dir results/
"""

import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob


def load_geojson(filepath):
    """Load a GeoJSON file and extract properties into a DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract properties from each feature
    records = []
    for feature in data['features']:
        props = feature['properties'].copy()
        # Add coordinates if available
        if feature.get('geometry') and feature['geometry'].get('coordinates'):
            coords = feature['geometry']['coordinates']
            props['longitude'] = coords[0]
            props['latitude'] = coords[1]
        records.append(props)
    
    df = pd.DataFrame(records)
    return df, data.get('name', Path(filepath).stem)


def clean_data(df, null_values=None):
    """
    Clean the data by handling null/placeholder values.
    
    Args:
        df: DataFrame with parcel data
        null_values: List of values to treat as null (default includes -666666666, -222222222)
    """
    if null_values is None:
        # Common placeholder values for missing data in census data
        null_values = [-666666666, -666666666.0, -222222222, -222222222.0, -999999999, -999999999.0]
    
    df_clean = df.copy()
    
    # Replace placeholder values with NaN
    for val in null_values:
        df_clean = df_clean.replace(val, np.nan)
    
    return df_clean


def select_numeric_columns(df, exclude_cols=None):
    """Select numeric columns suitable for PCA, excluding coordinates and MOE columns."""
    if exclude_cols is None:
        exclude_cols = ['longitude', 'latitude', 'park_distance']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns and margin of error columns
    pca_cols = [col for col in numeric_cols 
                if col not in exclude_cols 
                and not col.endswith('_moe')]
    
    return pca_cols


def run_pca_analysis(df, columns, n_components=None):
    """
    Run PCA analysis on the specified columns.
    
    Args:
        df: DataFrame with data
        columns: List of column names to include in PCA
        n_components: Number of components (default: min of n_features, n_samples)
    
    Returns:
        Dictionary with PCA results
    """
    # Prepare data - drop rows with any NaN in the selected columns
    df_subset = df[columns].dropna()
    
    if len(df_subset) < 2:
        raise ValueError("Not enough valid data points for PCA after removing NaN values")
    
    if n_components is None:
        n_components = min(len(columns), len(df_subset))
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_subset)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(data_scaled)
    
    # Create results dictionary
    results = {
        'pca': pca,
        'scaler': scaler,
        'transformed_data': pca_transformed,
        'columns': columns,
        'n_samples': len(df_subset),
        'n_features': len(columns),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'feature_names': columns,
        'loadings': pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=columns
        )
    }
    
    return results


def print_pca_summary(results, dataset_name="Dataset"):
    """Print a summary of PCA results."""
    print(f"\n{'='*60}")
    print(f"PCA Analysis Results: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\nData Summary:")
    print(f"  - Number of samples (parcels): {results['n_samples']:,}")
    print(f"  - Number of features: {results['n_features']}")
    print(f"  - Features analyzed: {', '.join(results['feature_names'])}")
    
    print(f"\nExplained Variance by Component:")
    print("-" * 40)
    for i, (var, cum_var) in enumerate(zip(
        results['explained_variance_ratio'],
        results['cumulative_variance_ratio']
    )):
        print(f"  PC{i+1}: {var*100:6.2f}% (cumulative: {cum_var*100:6.2f}%)")
    
    print(f"\nComponent Loadings (correlations with original variables):")
    print("-" * 40)
    print(results['loadings'].round(4).to_string())
    
    # Interpret components
    print(f"\nComponent Interpretation:")
    print("-" * 40)
    for i in range(min(3, len(results['explained_variance_ratio']))):
        loadings = results['loadings'][f'PC{i+1}']
        sorted_loadings = loadings.abs().sort_values(ascending=False)
        
        print(f"\n  PC{i+1} ({results['explained_variance_ratio'][i]*100:.1f}% variance):")
        print(f"    Strongest contributors:")
        for feat in sorted_loadings.head(3).index:
            val = loadings[feat]
            direction = "+" if val > 0 else "-"
            print(f"      {direction} {feat}: {val:.3f}")


def create_visualizations(results, output_dir, dataset_name):
    """Create and save PCA visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Scree plot - Explained variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_components = len(results['explained_variance_ratio'])
    x = range(1, n_components + 1)
    
    # Individual variance
    axes[0].bar(x, results['explained_variance_ratio'] * 100, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Variance Explained by Each Component')
    axes[0].set_xticks(x)
    
    # Cumulative variance
    axes[1].plot(x, results['cumulative_variance_ratio'] * 100, 'o-', color='steelblue', linewidth=2)
    axes[1].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[1].axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance (%)')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].set_ylim(0, 105)
    
    plt.suptitle(f'PCA Scree Plot - {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_scree_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Loadings heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(results['feature_names']) * 0.5)))
    
    sns.heatmap(
        results['loadings'],
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    ax.set_title(f'PCA Loadings Heatmap - {dataset_name}')
    ax.set_ylabel('Features')
    ax.set_xlabel('Principal Components')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_loadings_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Biplot (if we have at least 2 components)
    if results['transformed_data'].shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot samples
        transformed = results['transformed_data']
        ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.3, s=10, c='gray')
        
        # Plot feature vectors
        loadings = results['loadings']
        scale = max(abs(transformed[:, 0].max()), abs(transformed[:, 1].max())) * 0.8
        
        for i, feature in enumerate(results['feature_names']):
            ax.arrow(0, 0, 
                    loadings.iloc[i, 0] * scale, 
                    loadings.iloc[i, 1] * scale,
                    head_width=0.1 * scale/5, 
                    head_length=0.05 * scale/5, 
                    fc='red', ec='red', alpha=0.8)
            ax.text(loadings.iloc[i, 0] * scale * 1.1, 
                   loadings.iloc[i, 1] * scale * 1.1, 
                   feature, fontsize=9, ha='center', va='center')
        
        ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)')
        ax.set_title(f'PCA Biplot - {dataset_name}')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name}_biplot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Correlation matrix of original variables
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # We need the original data for correlation
    # This will be added as a parameter if needed
    
    print(f"  Visualizations saved to: {output_dir}/")


def create_correlation_matrix(df, columns, output_dir, dataset_name):
    """Create and save correlation matrix visualization."""
    output_dir = Path(output_dir)
    
    df_subset = df[columns].dropna()
    corr_matrix = df_subset.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True
    )
    ax.set_title(f'Correlation Matrix - {dataset_name}')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return corr_matrix


def save_results_to_csv(results, output_dir, dataset_name):
    """Save PCA results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save loadings
    results['loadings'].to_csv(output_dir / f'{dataset_name}_loadings.csv')
    
    # Save variance explained
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(results['explained_variance_ratio']))],
        'Explained_Variance': results['explained_variance_ratio'],
        'Cumulative_Variance': results['cumulative_variance_ratio']
    })
    variance_df.to_csv(output_dir / f'{dataset_name}_variance_explained.csv', index=False)
    
    print(f"  Results saved to: {output_dir}/")


def process_single_file(filepath, output_dir, create_plots=True):
    """Process a single GeoJSON file."""
    print(f"\nProcessing: {filepath}")
    
    # Load data
    df, dataset_name = load_geojson(filepath)
    print(f"  Loaded {len(df):,} features")
    
    # Clean data
    df_clean = clean_data(df)
    
    # Select columns for PCA
    pca_columns = select_numeric_columns(df_clean)
    print(f"  Selected {len(pca_columns)} variables for PCA: {pca_columns}")
    
    if len(pca_columns) < 2:
        print("  ERROR: Need at least 2 numeric variables for PCA")
        return None
    
    # Run PCA
    try:
        results = run_pca_analysis(df_clean, pca_columns)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return None
    
    # Print summary
    print_pca_summary(results, dataset_name)
    
    # Create visualizations and save results
    if output_dir:
        save_results_to_csv(results, output_dir, dataset_name)
        
        if create_plots:
            create_visualizations(results, output_dir, dataset_name)
            corr_matrix = create_correlation_matrix(df_clean, pca_columns, output_dir, dataset_name)
            print(f"\nCorrelation Matrix:")
            print(corr_matrix.round(3).to_string())
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run PCA analysis on parcel GeoJSON files with demographics and park proximity data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parcel_pca_analysis.py boston_parcels.geojson
  python parcel_pca_analysis.py data/*.geojson --output-dir results/
  python parcel_pca_analysis.py city_data.geojson --no-plots
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input GeoJSON file(s). Supports glob patterns like "data/*.geojson"'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='pca_results',
        help='Output directory for results and plots (default: pca_results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    input_files = []
    for pattern in args.input_files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)
    
    # Process each file
    all_results = {}
    for filepath in input_files:
        if not Path(filepath).exists():
            print(f"WARNING: File not found: {filepath}")
            continue
        
        results = process_single_file(
            filepath,
            args.output_dir,
            create_plots=not args.no_plots
        )
        
        if results:
            all_results[filepath] = results
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Successfully processed {len(all_results)} of {len(input_files)} files")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()