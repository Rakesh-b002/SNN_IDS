import numpy as np
import joblib
import os

class SpikeEncoder:
    """
    Time-To-First-Spike (TTFS) encoder.
    Converts normalized feature vectors into spike timing matrices.

    Based on paper: Event-Driven IDS using SNN (ICSCSA-2025)
    Formula : t_spike = feature_value x T_window  (Eq. 2)
    Threshold: theta = mean of each feature on training data (Table III)
    Refractory: delta = 5ms
    """

    def __init__(self, T=100, delta=5):
        self.T          = T        # time window in ms (100ms from paper)
        self.delta      = delta    # refractory period in ms (5ms from paper)
        self.thresholds = None     # fitted per-feature thresholds
        self.n_features = None

    def fit(self, X_train):
        """
        Compute dynamic thresholds from training data.
        theta_i = mean of feature i across all training samples.
        """
        self.thresholds = np.mean(X_train, axis=0) * 0.5  # 0.5x mean — handles skewed distributions
        self.n_features = X_train.shape[1]
        print(f"SpikeEncoder fitted:")
        print(f"  Features  : {self.n_features}")
        print(f"  T window  : {self.T}ms")
        print(f"  Refractory: {self.delta}ms")
        print(f"  Thresholds (first 5): {self.thresholds[:5].round(4)}")
        return self

    def encode_single(self, sample):
        """
        Encode one feature vector into a spike matrix.

        Args:
            sample: 1D array of shape (n_features,) normalized [0,1]

        Returns:
            spike_matrix: 2D array of shape (n_features, T)
        """
        spike_matrix = np.zeros((self.n_features, self.T), dtype=np.float32)
        last_spike   = np.full(self.n_features, -self.delta - 1)

        for i in range(self.n_features):
            fi = sample[i]

            # Rule 1: Threshold check
            if fi < self.thresholds[i]:
                continue

            # Rule 2: Compute spike timing
            t_spike = int(fi * (self.T - 1))
            t_spike = min(t_spike, self.T - 1)

            # Rule 3: Refractory period check
            if (t_spike - last_spike[i]) > self.delta:
                spike_matrix[i, t_spike] = 1.0
                last_spike[i] = t_spike

        return spike_matrix

    def encode(self, X):
        """
        Encode entire dataset into spike matrices.

        Args:
            X: 2D array of shape (N, n_features)

        Returns:
            spikes: 3D array of shape (N, n_features, T)
        """
        assert self.thresholds is not None, "Call fit() before encode()"
        N      = X.shape[0]
        spikes = np.zeros((N, self.n_features, self.T), dtype=np.float32)

        for idx in range(N):
            spikes[idx] = self.encode_single(X[idx])
            if (idx + 1) % 10000 == 0:
                print(f"  Encoded {idx+1}/{N} samples...")

        return spikes

    def get_sparsity(self, spikes):
        """
        Sparsity = fraction of zeros in spike matrix.
        High sparsity (>90%) expected and desirable for energy efficiency.
        """
        total  = spikes.size
        active = np.count_nonzero(spikes)
        return 1.0 - (active / total)

    def save(self, path):
        joblib.dump({
            'thresholds': self.thresholds,
            'T'         : self.T,
            'delta'     : self.delta,
            'n_features': self.n_features
        }, path)
        print(f"Encoder saved to {path}")

    def load(self, path):
        data            = joblib.load(path)
        self.thresholds = data['thresholds']
        self.T          = data['T']
        self.delta      = data['delta']
        self.n_features = data['n_features']
        return self
