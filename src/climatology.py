"""Climatology module for threshold calculation.

Computes P90 and P99 percentile thresholds from daily precipitation data.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml


@dataclass
class ThresholdConfig:
    """Configuration for threshold calculation."""
    input_path: str
    output_path: str
    wet_day_threshold: float = 1.0  # mm
    min_events: int = 10
    absolute_floor: float = 10.0  # mm
    chunk_y: int = 100
    chunk_x: int = 100


def load_config(config_path: str) -> ThresholdConfig:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Validated ThresholdConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required parameters are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError("Configuration file is empty")

    # Get threshold_calculation section
    config_section = raw_config.get('threshold_calculation', raw_config)
    
    # Validate required parameters
    required_params = ['input_path', 'output_path']
    for param in required_params:
        if param not in config_section:
            raise ValueError(f"Missing required configuration parameter: {param}")
    
    # Build config with defaults for optional parameters
    return ThresholdConfig(
        input_path=config_section['input_path'],
        output_path=config_section['output_path'],
        wet_day_threshold=config_section.get('wet_day_threshold', 1.0),
        min_events=config_section.get('min_events', 10),
        absolute_floor=config_section.get('absolute_floor', 10.0),
        chunk_y=config_section.get('chunk_y', 100),
        chunk_x=config_section.get('chunk_x', 100),
    )


def compute_wet_percentiles(
    precip_1d: np.ndarray,
    wet_day_threshold: float,
    min_events: int
) -> np.ndarray:
    """Compute 90th and 99th percentiles from wet days.
    
    This is the core kernel function applied to each pixel's time series.
    
    Args:
        precip_1d: 1D array of precipitation values (full time series)
        wet_day_threshold: Minimum precipitation to count as wet day
        min_events: Minimum wet days required for valid percentiles
        
    Returns:
        Array of shape (2,) containing [p90, p99], or [NaN, NaN] if arid
    """
    wet = precip_1d[precip_1d >= wet_day_threshold]
    if len(wet) < min_events:
        return np.array([np.nan, np.nan])
    return np.percentile(wet, [90, 99])


def calculate_thresholds(
    precip: xr.DataArray,
    config: ThresholdConfig
) -> xr.Dataset:
    """Calculate P90 and P99 threshold maps.
    
    Args:
        precip: Dask-backed precipitation DataArray
        config: Threshold configuration
        
    Returns:
        Dataset containing p90_map and p99_map variables
    """
    # Apply kernel using apply_ufunc with dask parallelization
    result = xr.apply_ufunc(
        compute_wet_percentiles,
        precip,
        config.wet_day_threshold,
        config.min_events,
        input_core_dims=[['time'], [], []],
        output_core_dims=[['percentile']],
        output_dtypes=[np.float64],
        dask='parallelized',
        vectorize=True,
        dask_gufunc_kwargs={'output_sizes': {'percentile': 2}},
    )
    
    # Split into p90 and p99
    p90 = result.isel(percentile=0).drop_vars('percentile')
    p99 = result.isel(percentile=1).drop_vars('percentile')
    
    # Apply absolute floor to P99 only
    p99 = xr.where(p99 < config.absolute_floor, config.absolute_floor, p99)
    # Preserve NaN values (don't apply floor to arid pixels)
    p99 = xr.where(np.isnan(result.isel(percentile=1)), np.nan, p99)
    
    # Create output dataset
    ds = xr.Dataset({
        'p90_map': p90.astype(np.float32),
        'p99_map': p99.astype(np.float32),
    })
    
    return ds


def run_threshold_calculation(config_path: str) -> None:
    """Run the complete threshold calculation workflow.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Check input file exists
    if not os.path.exists(config.input_path):
        raise FileNotFoundError(f"Input file not found: {config.input_path}")
    
    # Open NetCDF with chunking (time=-1 for full time series per pixel)
    ds = xr.open_dataset(
        config.input_path,
        chunks={'time': -1, 'y': config.chunk_y, 'x': config.chunk_x}
    )
    
    # Find precipitation variable
    precip_var = None
    for var in ['precipitation', 'precip', 'pr', 'PRCP']:
        if var in ds.data_vars:
            precip_var = var
            break
    
    if precip_var is None:
        raise ValueError(
            f"Precipitation variable not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )
    
    precip = ds[precip_var]
    
    # Compute thresholds
    thresholds = calculate_thresholds(precip, config)
    
    # Preserve spatial coordinates
    for coord in ['y', 'x', 'lat', 'lon', 'latitude', 'longitude']:
        if coord in ds.coords:
            thresholds.coords[coord] = ds.coords[coord]
    
    # Create output directory if needed
    output_dir = Path(config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to Zarr
    thresholds.to_zarr(config.output_path, mode='w')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.climatology <config_path>")
        sys.exit(1)
    run_threshold_calculation(sys.argv[1])
