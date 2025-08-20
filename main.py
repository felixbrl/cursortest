"""
Main entry point for the credit downgrade prediction pipeline.

This script provides a command-line interface to either train a new model
or run a prediction for a specific sector using a pre-trained model.
"""
import argparse
import configparser
import sys

import pandas as pd

from data_handler import DataHandler
from feature_engineering import FeatureEngineer
from modeler import Modeler

MODEL_PATH = 'downgrade_model.joblib'


def train():
    """
    Runs the full model training pipeline.
    1. Loads data from DataHandler.
    2. Engineers features.
    3. Splits data and trains the model.
    4. Evaluates the model and saves it to a file.
    """
    print("--- Starting Model Training Pipeline ---")

    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_date = config['MODEL_PARAMETERS']['TEST_START_DATE']

    # 1. Get and feature engineer the data
    print("\nStep 1: Loading and engineering features...")
    raw_data = DataHandler().get_full_dataset()
    if raw_data.empty:
        print("Error: DataHandler returned an empty DataFrame. Cannot proceed.")
        sys.exit(1)

    X, y = FeatureEngineer(raw_data).create_features()
    print("Feature engineering complete.")

    # 2. Initialize and run the modeling pipeline
    print("\nStep 2: Initializing model and splitting data...")
    modeler = Modeler(model_type='RandomForest')
    X_train, X_test, y_train, y_test = modeler.split_and_scale_data(X, y, test_date)

    # 3. Train and evaluate
    if X_train.empty or X_test.empty:
        print("\nError: Not enough data to train or test the model after splitting.")
        print(f"Please check the data and the split date ({test_date}).")
        sys.exit(1)

    print("\nStep 3: Training and evaluating the model...")
    modeler.train(X_train, y_train)
    modeler.evaluate(X_test, y_test)
    modeler.get_feature_importance(X_train.columns, top_n=5)

    # 4. Save the model
    print("\nStep 4: Saving the model...")
    modeler.save_model(MODEL_PATH)

    print("\n--- Model Training Pipeline Finished ---")


from predictor import get_prediction


def predict(sector, model_path):
    """
    Loads a pre-trained model and makes a prediction for a specific sector.

    Args:
        sector (str): The sector to predict for.
        model_path (str): The path to the saved model file.
    """
    print(f"--- Running Prediction for {sector} Sector ---")

    probability = get_prediction(sector, model_path)

    if probability is not None:
        # Output the result
        print("\n--- Prediction Result ---")
        print(f"The probability of a credit rating downgrade for the {sector} sector in the next 6-12 months is: {probability:.1%}")
    else:
        print("\nPrediction failed.")
        sys.exit(1)


def main():
    """
    Main function to parse arguments and call the appropriate pipeline.
    """
    # Get available sectors for the prediction choices
    config = configparser.ConfigParser()
    config.read('config.ini')
    available_sectors = [s.capitalize() for s in config['SECTORS'].keys()]

    parser = argparse.ArgumentParser(description="Credit Downgrade Prediction Pipeline")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-parser for the 'train' command
    train_parser = subparsers.add_parser('train', help="Run the full training pipeline.")

    # Sub-parser for the 'predict' command
    predict_parser = subparsers.add_parser('predict', help="Run a prediction for a single sector.")
    predict_parser.add_argument(
        '--sector',
        required=True,
        choices=available_sectors,
        help="The sector to make a prediction for."
    )
    predict_parser.add_argument(
        '--model_path',
        default=MODEL_PATH,
        help=f"Path to the trained model file (default: {MODEL_PATH})."
    )

    args = parser.parse_args()

    if args.command == 'train':
        train()
    elif args.command == 'predict':
        predict(args.sector, args.model_path)


if __name__ == '__main__':
    main()
