"""
This module contains the core logic for generating a prediction.
"""
import sys
import pandas as pd

from data_handler import DataHandler
from feature_engineering import FeatureEngineer
from modeler import Modeler


def get_prediction(sector, model_path):
    """
    Loads a pre-trained model and generates a prediction for a given sector.

    This function encapsulates the end-to-end process of loading a model,
    fetching the latest data, engineering features for the specific sector,
    and returning a downgrade probability.

    Args:
        sector (str): The sector to predict for (e.g., 'Technology').
        model_path (str): The path to the saved model file.

    Returns:
        float: The predicted probability of a downgrade (between 0.0 and 1.0).
               Returns None if an error occurs (e.g., model not found).
    """
    print("--- Loading Model and Data ---")
    # 1. Load the trained model and scaler
    try:
        model, scaler = Modeler.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please train a model first by running: python main.py train")
        return None

    # 2. Get the latest data and engineer features
    print("Fetching latest data and engineering features...")
    raw_data = DataHandler().get_full_dataset()
    if raw_data.empty:
        print("Error: Data could not be loaded.")
        return None

    X, _ = FeatureEngineer(raw_data).create_features()

    # 3. Get the most recent feature set for the specified sector
    try:
        latest_features = X.xs(sector, level='Sector').iloc[[-1]]
        print(f"Using data from date: {latest_features.index.get_level_values('Date')[0].date()}")
    except (KeyError, IndexError):
        print(f"Error: No data available for sector '{sector}' after feature engineering.")
        return None

    # 4. Scale the features using the loaded scaler
    scaled_features_np = scaler.transform(latest_features)
    scaled_features = pd.DataFrame(scaled_features_np, index=latest_features.index, columns=latest_features.columns)

    # 5. Make a prediction
    print("--- Generating Prediction ---")
    probability = model.predict_proba(scaled_features)[:, 1]

    return probability[0]
