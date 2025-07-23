import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import GradientBoostingRegressor  # Using GradientBoosting instead of RandomForest

def train_and_save_model():
    try:
        # Generate synthetic training data
        n_samples = 1000
        np.random.seed(42)
        
        # Generate features with realistic ranges
        X = pd.DataFrame({
            'temperature_2_m_above_gnd': np.random.uniform(10, 40, n_samples),  # Temperature (°C)
            'shortwave_radiation_backwards_sfc': np.random.uniform(0, 1000, n_samples),  # Solar radiation (W/m²)
            'wind_speed_10_m_above_gnd': np.random.uniform(0, 20, n_samples),  # Wind speed (m/s)
            'relative_humidity_2_m_above_gnd': np.random.uniform(30, 100, n_samples),  # Humidity (%)
            'total_cloud_cover_sfc': np.random.uniform(0, 100, n_samples)  # Cloud cover (%)
        })
        
        # Generate target variable (power demand in MW)
        y = (
            1000 +  # Base load
            X['temperature_2_m_above_gnd'] * 10 +
            X['shortwave_radiation_backwards_sfc'] * 0.5 +
            X['wind_speed_10_m_above_gnd'] * (-5) +
            X['relative_humidity_2_m_above_gnd'] * 2 +
            X['total_cloud_cover_sfc'] * (-3)
        )
        
        # Add some random noise
        y += np.random.normal(0, 50, n_samples)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Create and train a simple GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'power_prediction_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        # Save as pickle files
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        # Test prediction
        test_input = pd.DataFrame({
            'temperature_2_m_above_gnd': [25],
            'shortwave_radiation_backwards_sfc': [500],
            'wind_speed_10_m_above_gnd': [10],
            'relative_humidity_2_m_above_gnd': [60],
            'total_cloud_cover_sfc': [50]
        })
        test_scaled = scaler.transform(test_input)
        test_pred = model.predict(test_scaled)
        print(f"Test prediction successful: {test_pred[0]:.2f} MW")
        
        return True
        
    except Exception as e:
        print(f"Error in training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model()
