import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
from joblib import Parallel, delayed
from .decomposition import Decomposition
from .sparse_coding import SparseCoding
import logging

class PatchProcessor:
    """Handles patch-based processing for HSI enhancement."""
    
    def __init__(self, hsi, msi, n_components=5, n_atoms=5, lambda_reg=0.0005):
        self.hsi = hsi
        self.msi = msi
        self.n_components = n_components
        self.n_atoms = n_atoms
        self.lambda_reg = lambda_reg
        self.decomposition = Decomposition()
        self.sparse_coding = SparseCoding()

    def process_patch(self, x, y, patch_size):
        """Process a single patch."""
        try:
            hsi_patch = self.hsi[x:x + patch_size, y:y + patch_size, :]
            msi_patchLR = self.msi_lr[x:x + patch_size, y:y + patch_size, :]
            msi_patchHR = self.msi[x * self.f:x * self.f + patch_size * self.f,
                                  y * self.f:y * self.f + patch_size * self.f, :]

            valid = np.isfinite(hsi_patch).all(axis=-1) & np.isfinite(msi_patchLR).all(axis=-1)
            if np.sum(valid) > 5:
                hsi_patch_clean = np.nan_to_num(hsi_patch, nan=np.nanmean(hsi_patch[valid]))
                msi_patchLR_clean = np.nan_to_num(msi_patchLR, nan=np.nanmean(msi_patchLR[valid]))
                msi_patchHR_clean = np.nan_to_num(msi_patchHR, nan=np.nanmean(msi_patchHR))

                wt_components = self.decomposition.wavelet_3d_transform(hsi_patch_clean, self.n_components)
                ica_components = self.decomposition.fastica_decomposition(hsi_patch_clean, self.n_components)
                nmf_components = self.decomposition.nmf_decomposition(hsi_patch_clean, self.n_components)

                combined_components = np.hstack([wt_components, ica_components, nmf_components])

                hsi_patch_2d = hsi_patch_clean.reshape(patch_size, patch_size, -1)
                msi_patchLR_2d = msi_patchLR_clean.reshape(patch_size, patch_size, -1)
                msi_patchHR_2d = msi_patchHR_clean.reshape(patch_size * self.f, patch_size * self.f, -1)
                combined_components_2d = combined_components.reshape(patch_size, patch_size, -1)

                residual = self.sparse_coding.sparse_code_residual(
                    msi_patchLR_2d, msi_patchHR_2d, combined_components_2d, self.n_atoms, self.f, self.lambda_reg)

                return (x * self.f, y * self.f, residual)
            return None
        except Exception as e:
            logging.error(f"Patch ({x}, {y}) failed: {str(e)}")
            return None

    def run_parallel(self, patch_size=12, stride=1):
        """Run patch processing in parallel."""
        self.f = self.msi.shape[0] // self.hsi.shape[0]
        self.msi_lr = zoom(self.msi, (1/self.f, 1/self.f, 1), order=2, mode='nearest')

        coords = [(x, y) for x in range(0, self.hsi.shape[0] - patch_size + 1, stride)
                  for y in range(0, self.hsi.shape[1] - patch_size + 1, stride)]

        hsi_hr = np.zeros((self.msi.shape[0], self.msi.shape[1], self.hsi.shape[2]), dtype=np.float32)
        counts = np.zeros((self.msi.shape[0], self.msi.shape[1]), dtype=np.int32)

        backend = 'threading' if self.hsi.size < 1e6 else 'loky'
        results = Parallel(n_jobs=4, backend=backend)(
            delayed(self.process_patch)(x, y, patch_size) for x, y in tqdm(coords, desc="Processing patches")
        )

        for result in results:
            if result is not None:
                x_start, y_start, residual = result
                x_end = x_start + residual.shape[0]
                y_end = y_start + residual.shape[1]
                x_start = min(x_start, self.msi.shape[0] - (x_end - x_start))
                y_start = min(y_start, self.msi.shape[1] - (y_end - y_start))
                x_end = min(x_end, self.msi.shape[0])
                y_end = min(y_end, self.msi.shape[1])
                n_channels = min(residual.shape[-1], hsi_hr.shape[-1])
                hsi_hr[x_start:x_end, y_start:y_end, :n_channels] += residual[..., :n_channels]
                counts[x_start:x_end, y_start:y_end] += 1

        valid_mask = counts > 0
        hsi_hr[valid_mask] = hsi_hr[valid_mask] / counts[valid_mask, np.newaxis]

        return hsi_hr
