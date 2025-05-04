import numpy as np
import pywt
from sklearn.decomposition import FastICA, NMF

class Decomposition:
    """Handles signal decomposition methods for HSI data."""
    
    @staticmethod
    def wavelet_3d_transform(data, n_components):
        """Apply 3D wavelet transform and extract components."""
        coeffs = pywt.wavedecn(data, 'db1', level=3)
        approx = pywt.waverecn([coeffs[0]] + [None] * (len(coeffs) - 1), 'db1')
        W = approx.reshape(-1, data.shape[-1])[:, :n_components]
        if W.shape[1] < n_components:
            W = np.pad(W, ((0, 0), (0, n_components - W.shape[1])), mode='constant')
        return W / np.linalg.norm(W, axis=0, keepdims=True)

    @staticmethod
    def fastica_decomposition(data, n_components):
        """Apply FastICA decomposition."""
        transformer = FastICA(n_components=n_components, random_state=0, max_iter=200)
        data_2d = data.reshape(-1, data.shape[-1])
        W = transformer.fit_transform(data_2d.T).T
        if W.shape[0] < n_components:
            W = np.pad(W, ((0, n_components - W.shape[0]), (0, 0)), mode='constant')
        return W[:, :n_components] / np.linalg.norm(W[:, :n_components], axis=0, keepdims=True)

    @staticmethod
    def nmf_decomposition(data, n_components):
        """Apply NMF decomposition."""
        transformer = NMF(n_components=n_components, init='random', random_state=0, max_iter=200)
        data_2d = data.reshape(-1, data.shape[-1])
        data_2d = np.abs(data_2d)
        W = transformer.fit_transform(data_2d.T).T
        if W.shape[0] < n_components:
            W = np.pad(W, ((0, n_components - W.shape[0]), (0, 0)), mode='constant')
        return W[:, :n_components] / np.linalg.norm(W[:, :n_components], axis=0, keepdims=True)