"""Test climatology module with synthetic data."""

import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from src.climatology import (
    ThresholdConfig,
    load_config,
    compute_wet_percentiles,
    calculate_thresholds,
    run_threshold_calculation,
)


def test_compute_wet_percentiles_basic():
    """Test basic percentile computation."""
    # Create simple test data: 100 days with values 0-99
    precip = np.arange(100, dtype=np.float64)
    
    # All days are wet (>= 1.0), so should compute percentiles
    result = compute_wet_percentiles(precip, wet_day_threshold=1.0, min_events=10)
    
    # Expected: 90th and 99th percentiles of values 1-99
    wet_values = precip[precip >= 1.0]
    expected = np.percentile(wet_values, [90, 99])
    
    np.testing.assert_array_almost_equal(result, expected)
    print("✓ Basic percentile computation works")


def test_compute_wet_percentiles_arid():
    """Test arid mask (insufficient wet days)."""
    # Only 5 wet days (< min_events=10)
    precip = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    result = compute_wet_percentiles(precip, wet_day_threshold=1.0, min_events=10)
    
    # Should return NaN for both percentiles
    assert np.isnan(result[0]) and np.isnan(result[1])
    print("✓ Arid mask works correctly")


def test_end_to_end_with_synthetic_data():
    """Test complete workflow with synthetic NetCDF data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic precipitation data
        # 365 days, 10x10 spatial grid
        time = np.arange(365)
        y = np.arange(10)
        x = np.arange(10)
        
        # Create precipitation with spatial variation
        # Some pixels wet, some arid
        precip_data = np.random.rand(365, 10, 10) * 50  # 0-50 mm
        
        # Make one corner arid (mostly zeros)
        precip_data[:, 0:2, 0:2] = np.random.rand(365, 2, 2) * 0.5
        
        # Make another corner very wet
        precip_data[:, 8:10, 8:10] = np.random.rand(365, 2, 2) * 100 + 20
        
        ds = xr.Dataset({
            'precipitation': (['time', 'y', 'x'], precip_data)
        }, coords={
            'time': time,
            'y': y,
            'x': x,
        })
        
        # Save to NetCDF
        input_path = tmpdir / "test_precip.nc"
        ds.to_netcdf(input_path)
        
        # Create config file
        config_path = tmpdir / "test_config.yaml"
        config_data = {
            'threshold_calculation': {
                'input_path': str(input_path),
                'output_path': str(tmpdir / "test_thresholds.zarr"),
                'wet_day_threshold': 1.0,
                'min_events': 10,
                'absolute_floor': 10.0,
                'chunk_y': 5,
                'chunk_x': 5,
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run threshold calculation
        run_threshold_calculation(str(config_path))
        
        # Load and verify output
        output_path = tmpdir / "test_thresholds.zarr"
        assert output_path.exists()
        
        result = xr.open_zarr(output_path)
        
        # Check structure
        assert 'p90_map' in result
        assert 'p99_map' in result
        assert result['p90_map'].shape == (10, 10)
        assert result['p99_map'].shape == (10, 10)
        
        # Check arid corner has NaN
        assert np.isnan(result['p90_map'].values[0, 0])
        assert np.isnan(result['p99_map'].values[0, 0])
        
        # Check wet corner has valid values
        assert not np.isnan(result['p90_map'].values[9, 9])
        assert not np.isnan(result['p99_map'].values[9, 9])
        
        # Check absolute floor applied to P99
        # For pixels with valid data, P99 should be >= 10.0
        valid_p99 = result['p99_map'].values[~np.isnan(result['p99_map'].values)]
        assert np.all(valid_p99 >= 10.0)
        
        print("✓ End-to-end workflow works correctly")
        print(f"  - Output saved to: {output_path}")
        print(f"  - P90 range: {np.nanmin(result['p90_map'].values):.2f} - {np.nanmax(result['p90_map'].values):.2f} mm")
        print(f"  - P99 range: {np.nanmin(result['p99_map'].values):.2f} - {np.nanmax(result['p99_map'].values):.2f} mm")
        print(f"  - Arid pixels (NaN): {np.sum(np.isnan(result['p90_map'].values))}/{10*10}")


if __name__ == '__main__':
    print("Testing threshold calculation implementation...\n")
    
    test_compute_wet_percentiles_basic()
    test_compute_wet_percentiles_arid()
    test_end_to_end_with_synthetic_data()
    
    print("\n✅ All tests passed!")
