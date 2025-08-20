"""
This module is responsible for creating features from the processed data
that will be used in the predictive model.
"""

import numpy as np
import pandas as pd
from data_handler import DataHandler


class FeatureEngineer:
    """
    Engineers features from the raw data for model training and prediction.

    This class takes a DataFrame from the DataHandler, generates time-series
    features like moving averages and momentum, and creates the final
    target variable for the specified prediction horizon.

    Attributes:
        data (pd.DataFrame): The dataset to be featured.
    """

    def __init__(self, data):
        """
        Initializes the FeatureEngineer.

        Args:
            data (pd.DataFrame): The input DataFrame from DataHandler, indexed
                                 by 'Date' and 'Sector'.
        """
        self.data = data.copy()

    def create_features(self):
        """
        Generates all features for the model.

        This orchestrator method performs the following steps:
        1. Generates time-series features (moving averages and rates of change).
        2. Engineers the forward-looking target variable.
        3. Cleans the resulting DataFrame by handling NaNs and infinite values.

        Returns:
            pd.DataFrame: A DataFrame with engineered features and the target variable,
                          ready for model training.
        """
        print("Generating time-series features (moving averages, momentum)...")
        self._generate_ts_features()

        print("Engineering the forward-looking target variable...")
        self._create_target_variable()

        print("Cleaning final dataset...")
        self._clean_dataframe()

        # Separate features and target
        y = self.data['target']
        X = self.data.drop(columns=['target'])

        return X, y

    def _generate_ts_features(self):
        """
        Calculates moving averages and rate-of-change for all feature columns.
        """
        feature_cols = [col for col in self.data.columns if col != 'Downgrade_Occurred']

        # Group by sector to calculate features independently for each sector
        grouped = self.data.groupby('Sector')

        for col in feature_cols:
            # Moving Averages (e.g., 6-month and 12-month)
            self.data[f'{col}_MA_2Q'] = grouped[col].transform(
                lambda x: x.rolling(window=2, min_periods=1).mean()
            )
            self.data[f'{col}_MA_4Q'] = grouped[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean()
            )

            # Rate of Change (Momentum)
            self.data[f'{col}_ROC_1Q'] = grouped[col].transform(
                lambda x: x.pct_change(periods=1, fill_method=None)
            )
            self.data[f'{col}_ROC_4Q'] = grouped[col].transform(
                lambda x: x.pct_change(periods=4, fill_method=None)
            )

    def _create_target_variable(self):
        """
        Creates the target variable for predicting a downgrade in the next 6-12 months.
        """
        # The goal is to predict if a downgrade will occur in the next 6 to 12 months,
        # which corresponds to a window of 2 to 4 quarters from the current time `t`.
        # We create a target variable that is 1 if a downgrade occurred at t+2, t+3, or t+4.
        g = self.data.groupby('Sector')['Downgrade_Occurred']

        # Shift the downgrade flag back in time to align with current features
        shifted_downgrades = pd.concat([
            g.transform(lambda x: x.shift(-2)),
            g.transform(lambda x: x.shift(-3)),
            g.transform(lambda x: x.shift(-4))
        ], axis=1)

        # The target is 1 if a downgrade occurs in *any* of the next 2, 3, or 4 quarters
        self.data['target'] = shifted_downgrades.max(axis=1)

    def _clean_dataframe(self):
        """
        Cleans the DataFrame by dropping unnecessary columns and handling missing values.
        """
        # Drop the original Downgrade_Occurred column as it's now encoded in the target
        self.data.drop(columns=['Downgrade_Occurred'], inplace=True)

        # Replace infinite values that can arise from pct_change (division by zero)
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with any NaN values. This handles:
        # 1. NaNs at the start of each series from moving average/ROC calculations.
        # 2. NaNs at the end of each series from shifting the target variable.
        self.data.dropna(inplace=True)

        # Ensure target is integer type
        self.data['target'] = self.data['target'].astype(int)


if __name__ == '__main__':
    # Example of how to use the FeatureEngineer
    print("Running FeatureEngineer standalone example...")

    # 1. Get data from DataHandler
    print("First, getting data from DataHandler...")
    dh = DataHandler()
    raw_data = dh.get_full_dataset()

    if not raw_data.empty:
        # 2. Engineer features
        print("\nNow, engineering features...")
        feature_engineer = FeatureEngineer(raw_data)
        X_featured, y_target = feature_engineer.create_features()

        print("\nShape of the final features (X):", X_featured.shape)
        print("Shape of the final target (y):", y_target.shape)

        print("\nSample of the final features dataset:")
        print(X_featured.head())

        print("\nTarget variable distribution:")
        print(y_target.value_counts())
    else:
        print("Could not retrieve data to run feature engineering example.")
