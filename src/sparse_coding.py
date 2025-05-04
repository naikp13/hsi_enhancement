import numpy as np
import numba as nb
from sklearn.decomposition import DictionaryLearning
from scipy.ndimage import zoom

class SparseCoding:
    """Handles dictionary learning and sparse coding with FISTA."""
    
    @staticmethod
    @nb.njit
    def fista(X, D, lambda_reg, max_iter=75, tol=1e-6):
        """Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)."""
        n_samples, n_features = X.shape
        n_atoms = D.shape[1]
        L = np.linalg.norm(D, ord=2)**2
        if L == 0:
            return np.zeros((n_samples, n_atoms))

        alpha = np.zeros((n_samples, n_atoms))
        y = np.zeros((n_samples, n_atoms))
        t = 1.0

        for k in range(max_iter):
            alpha_prev = alpha.copy()
            grad = np.dot(X - np.dot(y, D.T), D) / L
            alpha = nb.njit(lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0))(y - grad, lambda_reg / L)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = alpha + ((t - 1) / t_next) * (alpha - alpha_prev)
            t = t_next
            if k > 5 and np.max(np.abs(alpha - alpha_prev)) < tol:
                break

        return alpha

    @staticmethod
    def train_dictionary(msi_lr_flat, hsi_comp_flat, n_atoms):
        """Train dictionary for sparse coding."""
        data = np.hstack([msi_lr_flat.T, hsi_comp_flat.T]).T
        dict_learner = DictionaryLearning(n_components=n_atoms, alpha=1, max_iter=20)
        dict_learner.fit(data)
        return dict_learner.components_.T

    def sparse_code_residual(self, msi_lr_patch, msi_hr_patch, hsi_components, n_atoms, f, lambda_reg):
        """Compute residual using sparse coding."""
        msi_lr_flat = msi_lr_patch.reshape(-1, msi_lr_patch.shape[-1])
        msi_hr_flat = msi_hr_patch.reshape(-1, msi_hr_patch.shape[-1])
        hsi_comp_flat = hsi_components.reshape(-1, hsi_components.shape[-1])

        D = self.train_dictionary(msi_lr_flat, hsi_comp_flat, n_atoms)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)

        coeffs_hr = self.fista(msi_hr_flat, D, lambda_reg)
        pred_hr = np.dot(coeffs_hr, D.T).reshape(msi_hr_patch.shape[0], msi_hr_patch.shape[1], -1)

        hsi_mean_upsampled = zoom(hsi_components.mean(axis=-1, keepdims=True),
                                (f, f, pred_hr.shape[-1]), order=3, mode='nearest')
        residual = pred_hr - hsi_mean_upsampled

        return residual
