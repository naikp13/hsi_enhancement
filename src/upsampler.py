import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

class HSIUpsampler:
    """Handles enhanced HSI upsampling with MSI detail injection."""
    
    @staticmethod
    def enhanced_hsi_upsampling(hsi, msi, msi_guide, detail_weight=3.5):
        """Upsample HSI while injecting MSI details."""
        if hsi.shape[2] < 1 or msi.shape[:2] != msi_guide.shape[:2]:
            raise ValueError("Inconsistent input dimensions.")

        original_means = np.mean(hsi, axis=(0, 1))
        original_stds = np.std(hsi, axis=(0, 1))

        hsi_upsampled = resize(
            hsi, (msi.shape[0], msi.shape[1], hsi.shape[2]),
            order=5, mode='edge', anti_aliasing=False, preserve_range=True
        ).astype(np.float32)

        msi_guide_gray = np.mean(msi_guide, axis=-1)
        msi_guide_gray = (msi_guide_gray - msi_guide_gray.min()) / \
                        (msi_guide_gray.max() - msi_guide_gray.min() + 1e-6)

        msi_low = gaussian_filter(msi_guide_gray, sigma=1, mode='reflect')
        msi_high = msi_guide_gray - msi_low

        for band in range(hsi_upsampled.shape[2]):
            enhanced = hsi_upsampled[..., band] + detail_weight * msi_high
            band_mean = np.mean(enhanced)
            band_std = np.std(enhanced)
            if band_std > 0:
                hsi_upsampled[..., band] = original_means[band] + \
                                         (enhanced - band_mean) * (original_stds[band] / band_std)
            else:
                hsi_upsampled[..., band] = enhanced

        return hsi_upsampled
