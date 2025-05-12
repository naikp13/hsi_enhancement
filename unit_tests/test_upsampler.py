import pytest
import numpy as np
from src.upsampler import HSIUpsampler

@pytest.fixture
def synthetic_upsampler_data():
    """Create synthetic MSI, HSI, and MSI guide data."""
    hsi = np.random.rand(50, 50, 10).astype(np.float32)
    msi = np.random.rand(100, 100, 3).astype(np.float32)
    msi_guide = np.random.rand(100, 100, 3).astype(np.float32)
    return hsi, msi, msi_guide

def test_enhanced_hsi_upsampling(synthetic_upsampler_data):
    """Test HSIUpsampler's enhanced_hsi_upsampling method."""
    hsi, msi, msi_guide = synthetic_upsampler_data
    upsampler = HSIUpsampler()
    
    hsi_upsampled = upsampler.enhanced_hsi_upsampling(hsi, msi, msi_guide, detail_weight=2.0)
    
    # Check output shape
    assert hsi_upsampled.shape == (100, 100, 10)
    
    # Check that output is finite
    assert np.all(np.isfinite(hsi_upsampled))
    
    # Check that output preserves approximate mean of original HSI
    original_mean = np.mean(hsi, axis=(0, 1))
    upsampled_mean = np.mean(hsi_upsampled, axis=(0, 1))
    assert np.allclose(original_mean, upsampled_mean, rtol=0.1, atol=0.1)

def test_upsampling_invalid_dimensions():
    """Test upsampling with invalid input dimensions."""
    upsampler = HSIUpsampler()
    hsi = np.random.rand(50, 50, 0).astype(np.float32)  # Empty channels
    msi = np.random.rand(100, 100, 3).astype(np.float32)
    msi_guide = np.random.rand(100, 100, 3).astype(np.float32)
    
    with pytest.raises(ValueError, match="Inconsistent input dimensions"):
        upsampler.enhanced_hsi_upsampling(hsi, msi, msi_guide)