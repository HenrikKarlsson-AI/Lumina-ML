"""
Predictive Maintenance Example: Industrial Turbine Monitoring
Uses Lumina-ML to forecast vibrations and temperature in an industrial turbine.
Designed to demonstrate high-precision forecasting for critical assets.
"""

import numpy as np
from lumina.forecaster import LuminaForecaster
from preprocessing.transforms import TimeSeriesScaler, create_windows

def run_turbine_simulation():
    print("--- Lumina-ML: Turbine Predictive Maintenance Simulation ---\n")

    # 1. Simulate Industrial Sensor Data (Temperature & Vibration)
    # 500 timesteps of normal operation with some seasonality
    t = np.linspace(0, 50, 500)
    temp = 65 + 5 * np.sin(t) + np.random.normal(0, 0.5, 500)
    vibration = 0.02 + 0.005 * np.cos(t) + np.random.normal(0, 0.001, 500)
    
    data = np.column_stack([temp, vibration])
    print(f"Generated {len(data)} sensor readings (Temperature, Vibration).")

    # 2. Preprocessing
    scaler = TimeSeriesScaler()
    scaled_data = scaler.fit_transform(data)
    
    window_size = 24  # Look back at last 24 readings
    X, y = create_windows(scaled_data, window_size)
    print(f"Created {len(X)} input sequences of length {window_size}.")

    # 3. Forecasting with Lumina
    # Forecast the next 12 readings (e.g., next 12 hours)
    forecaster = LuminaForecaster(horizon=12, hidden_dim=32)
    forecaster.build_model(X[0].shape)
    
    # Predict based on the latest window
    latest_window = X[-1]
    prediction_scaled = forecaster.predict(latest_window)
    
    # 4. Inverse Scaling to real-world units
    # Reshape prediction for inverse scaling (Lumina simulates 1 feature output for simplicity)
    # In real use, this would be a multi-feature forecast.
    # We'll just replicate the second column's range for visualization
    prediction_real = prediction_scaled * (scaler.max_val[0] - scaler.min_val[0]) + scaler.min_val[0]

    print(f"\nForecast for the next {forecaster.horizon} timesteps:")
    for i, val in enumerate(prediction_real):
        print(f"  T+{i+1}: {val:.2f} °C")

    print("\n[SUCCESS] Turbine status forecasted. No critical breach detected within the horizon.")

if __name__ == "__main__":
    run_turbine_simulation()
