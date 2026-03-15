"""
Data Transforms for Lumina-ML
Essential preprocessing utilities for time-series sensor data.
"""

import numpy as np
from typing import List, Tuple

class TimeSeriesScaler:
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.min_val = None
        self.max_val = None
        self.feature_range = feature_range

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Scales input data based on a specified feature range."""
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        
        normalized = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        scaled = normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverts the scaling to return to original units."""
        normalized = (scaled_data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        original = normalized * (self.max_val - self.min_val) + self.min_val
        return original

def create_windows(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sliding windows of input data for time-series forecasting."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    scaler = TimeSeriesScaler()
    test_data = np.array([[10, 20], [15, 25], [20, 30]])
    scaled = scaler.fit_transform(test_data)
    print(f"Scaled data: {scaled}")
    print(f"Inversed data: {scaler.inverse_transform(scaled)}")
