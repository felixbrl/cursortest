"""
This module handles the model training, evaluation, and persistence.
"""
import configparser
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

from data_handler import DataHandler
from feature_engineering import FeatureEngineer


class Modeler:
    """
    Handles the machine learning workflow, including data splitting, scaling,
    training, evaluation, and model persistence.

    Attributes:
        model: The scikit-learn model instance.
        scaler: The scikit-learn StandardScaler instance.
    """

    def __init__(self, model_type='RandomForest', model_params=None):
        """
        Initializes the Modeler with a specified model type.

        Args:
            model_type (str): The type of model to use.
                              Options: 'RandomForest', 'LogisticRegression'.
            model_params (dict, optional): Parameters for the model. Defaults to None.
        """
        self.model = None
        self.scaler = None
        self._model_type = model_type
        self._model_params = model_params if model_params is not None else {}
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the scikit-learn model object."""
        if self._model_type == 'RandomForest':
            # Params for imbalanced data and interpretability
            params = {
                'n_estimators': 100,
                'class_weight': 'balanced',
                'random_state': 42,
                **self._model_params
            }
            self.model = RandomForestClassifier(**params)
        elif self._model_type == 'LogisticRegression':
            params = {
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'random_state': 42,
                **self._model_params
            }
            self.model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")

        self.scaler = StandardScaler()

    def split_and_scale_data(self, X, y, test_start_date):
        """
        Splits data into time-based train/test sets and scales the features.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            test_start_date (str): The date to start the test set (e.g., '2020-01-01').

        Returns:
            tuple: A tuple containing (X_train, X_test, y_train, y_test).
        """
        split_date = pd.to_datetime(test_start_date)
        train_indices = X.index.get_level_values('Date') < split_date
        test_indices = X.index.get_level_values('Date') >= split_date

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame to retain index and columns
        X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Trains the model on the provided data.

        Args:
            X_train (pd.DataFrame): The training feature matrix.
            y_train (pd.Series): The training target vector.
        """
        print(f"Training {self._model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model and prints a comprehensive report.

        Args:
            X_test (pd.DataFrame): The test feature matrix.
            y_test (pd.Series): The test target vector.
        """
        print("\n--- Model Evaluation ---")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nConfusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                         index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive']))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def get_feature_importance(self, feature_names, top_n=5):
        """
        Identifies and prints the top N most important features.

        Args:
            feature_names (list): The list of feature names.
            top_n (int): The number of top features to display.
        """
        if self._model_type == 'RandomForest':
            importances = self.model.feature_importances_
        elif self._model_type == 'LogisticRegression':
            # Use the absolute value of coefficients for importance
            importances = abs(self.model.coef_[0])
        else:
            print("Feature importance not supported for this model type.")
            return

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n--- Top {top_n} Features ---")
        print(feature_importance_df.head(top_n))

    def save_model(self, path='model.joblib'):
        """
        Saves the trained model and scaler to a file.

        Args:
            path (str): The path to save the model file.
        """
        print(f"\nSaving model and scaler to {path}...")
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print("Save complete.")

    @staticmethod
    def load_model(path='model.joblib'):
        """
        Loads a model and scaler from a file.

        Args:
            path (str): The path to the model file.

        Returns:
            A tuple containing the loaded model and scaler.
        """
        print(f"Loading model and scaler from {path}...")
        data = joblib.load(path)
        return data['model'], data['scaler']


if __name__ == '__main__':
    print("--- Running Modeler standalone example ---")

    # 1. Load configuration for the test split date
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_date = config['MODEL_PARAMETERS']['TEST_START_DATE']

    # 2. Get and feature engineer the data
    raw_data = DataHandler().get_full_dataset()
    X, y = FeatureEngineer(raw_data).create_features()

    # 3. Initialize and run the modeling pipeline
    modeler = Modeler(model_type='RandomForest')
    X_train, X_test, y_train, y_test = modeler.split_and_scale_data(X, y, test_date)

    # Check if there is data to train and test on
    if not X_train.empty and not X_test.empty:
        modeler.train(X_train, y_train)
        modeler.evaluate(X_test, y_test)
        modeler.get_feature_importance(X_train.columns, top_n=5)
        modeler.save_model('downgrade_model.joblib')

        # Example of loading the model
        loaded_model, loaded_scaler = Modeler.load_model('downgrade_model.joblib')
        print("\nModel loaded successfully:")
        print(loaded_model)
    else:
        print("\nNot enough data to train or test the model after splitting.")
