import pytest
import numpy as np
from src.enhancer import HSIEnhancer
from src.data_loader import HSIDataLoader

@pytest.fixture
def synthetic_enhancer_data(tmp_path):
    """Create synthetic MSI and HSI data and save to temporary files."""
    msi = np.random.rand(100, 100, 3).astype(np.float32)
    hsi = np.random.rand(50, 50, 10).astype(np.float32)
    
    import rasterio
    msi_path = tmp_path / "msi.tif"
    hsi_path = tmp_path / "hsi.dat"
    
    with rasterio.open(msi_path, 'w', driver='GTiff', height=msi.shape[0], width=msi.shape[1], count=msi.shape[2], dtype='float32') as dst:
        for i in range(msi.shape[2]):
            dst.write(msi[:, :, i], i + 1)
    with rasterio.open(hsi_path, 'w', driver='GTiff', height=hsi.shape[0], width=hsi.shape[1], count=hsi.shape[2], dtype='float32') as dst:
        for i in range(hsi.shape[2]):
            dst.write(hsi[:, :, i], i + 1)
    
    return str(msi_path), str(hsi_path)

def test_fuse_to_enhance(synthetic_enhancer_data):
    """Test HSIEnhancer's fuse_to_enhance method."""
    msi_path, hsi_path = synthetic_enhancer_data
    enhancer = HSIEnhancer(msi_path, hsi_path, n_components=3, n_atoms=3, lambda_reg=0.0005)
    
    hsi_enhanced = enhancer.fuse_to_enhance(patch_size=8, stride=4, guide_radius=1, detail_weight=2.0)
    
    # Check output shape
    assert hsi_enhanced.shape == (100, 100, 10)
    
    # Check that output is finite
    assert np.all(np.isfinite(hsi_enhanced))
    
    # Check that output contains non-zero values
    assert np.any(hsi_enhanced != 0)

def test_enhancer_invalid_file():
    """Test HSIEnhancer with invalid file paths."""
    with pytest.raises(FileNotFoundError):
        enhancer = HSIEnhancer("nonexistent_msi.tif", "nonexistent_hsi.dat")