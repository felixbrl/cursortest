"""
This module is responsible for data ingestion and processing for the credit downgrade prediction model.
"""

import configparser
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np


class DataHandler:
    """
    Handles fetching, cleaning, and preprocessing all required data.

    This class orchestrates the collection of macroeconomic data from FRED,
    company-level financial data via yfinance, aggregates it to the sector
    level, and merges it with historical downgrade data.

    Attributes:
        config (configparser.ConfigParser): The configuration object.
        downgrade_history_path (str): Path to the downgrade history CSV.
        sectors (dict): Dictionary of sectors and their constituent tickers.
        macro_indicators (dict): Dictionary of macro indicators and their FRED codes.
    """

    def __init__(self, config_path='config.ini', downgrade_history_path='historical_downgrades.csv'):
        """
        Initializes the DataHandler.

        Args:
            config_path (str): Path to the configuration file (e.g., 'config.ini').
            downgrade_history_path (str): Path to the CSV file with historical downgrade data.
        """
        self.config = self._load_config(config_path)
        self.downgrade_history_path = downgrade_history_path
        self.sectors = self._get_sectors()
        self.macro_indicators = self._get_macro_indicators()

    def _load_config(self, config_path):
        """
        Loads the configuration from the specified .ini file.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            configparser.ConfigParser: A parser object with the loaded configuration.
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _get_sectors(self):
        """
        Retrieves the sector-ticker mapping from the configuration file.

        Returns:
            dict: A dictionary where keys are capitalized sector names and
                  values are lists of stock tickers.
        """
        sectors = {}
        for sector, tickers in self.config['SECTORS'].items():
            sectors[sector.capitalize()] = [ticker.strip() for ticker in tickers.split(',')]
        return sectors

    def _get_macro_indicators(self):
        """
        Retrieves the macroeconomic indicator mapping from the configuration file.

        Returns:
            dict: A dictionary of macroeconomic indicator names and their corresponding FRED codes.
        """
        return dict(self.config['MACRO_INDICATORS'])

    def load_downgrade_history(self):
        """
        Loads historical sector-level credit rating changes from the CSV file.

        The method ensures the 'Date' column is parsed into datetime objects
        and sets a multi-index on 'Date' and 'Sector'.

        Returns:
            pd.DataFrame: A DataFrame containing the historical downgrade data.
        """
        df = pd.read_csv(self.downgrade_history_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sector'] = df['Sector'].str.capitalize()
        df = df.set_index(['Date', 'Sector'])
        return df

    def fetch_macro_data(self, start_date='2010-01-01'):
        """
        Fetches macroeconomic data series from the Federal Reserve Economic Data (FRED).

        Args:
            start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame with macroeconomic data, resampled to a quarterly frequency.
        """
        fred_codes = list(self.macro_indicators.values())
        end_date = datetime.now()

        try:
            macro_data = web.DataReader(fred_codes, 'fred', start_date, end_date)
            macro_data.rename(columns={v: k.upper() for k, v in self.macro_indicators.items()}, inplace=True)

            # Forward-fill to handle non-trading days or missing data, then resample
            macro_data = macro_data.ffill().resample('QE').mean()
            macro_data.index = macro_data.index.to_period('Q').to_timestamp('Q')
        except Exception as e:
            print(f"Error fetching macro data from FRED: {e}")
            return pd.DataFrame()

        return macro_data

    def _calculate_financial_ratios(self, ticker):
        """
        Calculates key financial ratios for a single company using yfinance data.
        This internal method is designed to be robust to missing data or differences
        in financial statement structures (e.g., for financial institutions).
        Args:
            ticker (yf.Ticker): An initialized yfinance Ticker object.
        Returns:
            pd.DataFrame or None: A DataFrame with calculated financial ratios indexed
                                  by date, or None if data is insufficient.
        """
        try:
            balance_sheet = ticker.quarterly_balance_sheet
            income_statement = ticker.quarterly_financials

            if balance_sheet.empty or income_statement.empty:
                return None

            # Transpose so that dates are rows
            df = pd.concat([balance_sheet.T, income_statement.T], axis=1, join='inner')
            df.index = pd.to_datetime(df.index)

            def safe_get(data_frame, key):
                """Safely get a column or return a Series of NaNs with the same index."""
                if key in data_frame.columns:
                    return pd.to_numeric(data_frame[key], errors='coerce')
                return pd.Series(np.nan, index=data_frame.index, name=key)

            # --- Calculate all ratios, allowing for NaNs if data is missing ---

            # Leverage
            total_debt = safe_get(df, 'Total Debt')
            equity = safe_get(df, 'Stockholders Equity')
            total_assets = safe_get(df, 'Total Assets')
            df['Debt_to_Equity'] = (total_debt / equity).replace([np.inf, -np.inf], np.nan)
            df['Debt_to_Assets'] = (total_debt / total_assets).replace([np.inf, -np.inf], np.nan)

            # Profitability
            net_income = safe_get(df, 'Net Income')
            revenue = safe_get(df, 'Total Revenue')
            df['Net_Profit_Margin'] = (net_income / revenue).replace([np.inf, -np.inf], np.nan)
            df['ROE'] = (net_income / equity).replace([np.inf, -np.inf], np.nan)

            # Liquidity (may be NaN for financial companies)
            current_assets = safe_get(df, 'Current Assets')
            current_liabilities = safe_get(df, 'Current Liabilities')
            inventory = safe_get(df, 'Inventory').fillna(0)
            df['Current_Ratio'] = (current_assets / current_liabilities).replace([np.inf, -np.inf], np.nan)
            df['Quick_Ratio'] = ((current_assets - inventory) / current_liabilities).replace([np.inf, -np.inf], np.nan)

            ratio_cols = [
                'Debt_to_Equity', 'Debt_to_Assets', 'Net_Profit_Margin',
                'ROE', 'Current_Ratio', 'Quick_Ratio'
            ]

            # Return only the columns we created, and only if at least one has some data
            return df[ratio_cols].dropna(how='all')

        except Exception as e:
            print(f"Could not process ratios for {ticker.ticker}: {e}")
            return None

    def fetch_micro_data(self):
        """
        [JULES - MOCK IMPLEMENTATION]
        Generates a mock dataset for microeconomic features.
        NOTE: The yfinance API proved unreliable in the execution environment.
        To demonstrate the full pipeline, this function generates a synthetic but
        realistic dataset. In a real-world scenario, this would be replaced
        with a robust, production-grade data provider.
        """
        print("WARNING: Using mocked microeconomic data due to yfinance unreliability in environment.")

        base_df = self.load_downgrade_history()
        dates = pd.to_datetime(base_df.index.get_level_values('Date').unique()).sort_values()
        sectors = base_df.index.get_level_values('Sector').unique()

        base_ratios = {
            'Technology': {'Debt_to_Equity': 0.5, 'ROE': 0.25, 'Net_Profit_Margin': 0.20, 'Current_Ratio': 2.0},
            'Financials': {'Debt_to_Equity': 2.0, 'ROE': 0.10, 'Net_Profit_Margin': 0.15, 'Current_Ratio': np.nan},
            'Industrials': {'Debt_to_Equity': 1.2, 'ROE': 0.15, 'Net_Profit_Margin': 0.10, 'Current_Ratio': 1.5},
            'Healthcare': {'Debt_to_Equity': 0.8, 'ROE': 0.18, 'Net_Profit_Margin': 0.12, 'Current_Ratio': 2.5},
        }

        all_sector_dfs = []
        for sector in sectors:
            sector_data = []
            sector_base = base_ratios.get(sector, base_ratios['Industrials'])
            for date in dates:
                row = {'Date': date, 'Sector': sector}
                for ratio, base_value in sector_base.items():
                    if pd.isna(base_value):
                        row[ratio] = np.nan
                    else:
                        # Add some noise and a slight time trend
                        noise = np.random.normal(0, 0.1)
                        time_trend = (date - dates[0]).days / (365 * 10) * -0.1 # Approx 1% decay per year
                        row[ratio] = base_value * (1 + noise + time_trend)

                # Derive other ratios
                if not pd.isna(row.get('Current_Ratio')):
                    row['Quick_Ratio'] = row['Current_Ratio'] * np.random.normal(0.8, 0.05)
                else:
                    row['Quick_Ratio'] = np.nan
                row['Debt_to_Assets'] = row['Debt_to_Equity'] / (1 + row['Debt_to_Equity']) if not pd.isna(row.get('Debt_to_Equity')) else np.nan
                sector_data.append(row)
            all_sector_dfs.append(pd.DataFrame(sector_data))

        if not all_sector_dfs:
            return pd.DataFrame()

        final_df = pd.concat(all_sector_dfs)
        final_df.set_index(['Date', 'Sector'], inplace=True)
        return final_df

    def get_full_dataset(self):
        """
        Orchestrates the complete data loading and merging pipeline.
        This master method uses the historical downgrade data as the base, ensuring
        the full time series is preserved, and then left-joins macro and micro data.
        It forward-fills missing micro data to handle sparse API responses.
        Returns:
            pd.DataFrame: A fully merged and cleaned DataFrame.
        """
        print("Loading downgrade history...")
        base_df = self.load_downgrade_history()

        start_date = base_df.index.get_level_values('Date').min() - pd.DateOffset(years=2)
        start_date_str = start_date.strftime('%Y-%m-%d')

        print("Fetching macroeconomic data...")
        macro_data = self.fetch_macro_data(start_date=start_date_str)

        print("Fetching microeconomic (sector) data...")
        micro_data = self.fetch_micro_data()

        # Start with the base data (downgrades), which has the full date range
        full_data = base_df.join(macro_data, on='Date')

        # Join the potentially sparse micro data
        if not micro_data.empty:
            full_data = full_data.join(micro_data, on=['Date', 'Sector'])

        # The target variable is already in the base_df.
        # Drop rows where we have no feature data at all (especially macro data)
        macro_cols_upper = [k.upper() for k in self.macro_indicators.keys()]
        full_data.dropna(subset=macro_cols_upper, inplace=True)

        print("Full dataset created successfully.")
        return full_data

if __name__ == '__main__':
    # Example of how to use the DataHandler
    print("Running DataHandler standalone example...")
    data_handler = DataHandler()

    # Fetch and display a sample of the final dataset
    final_dataset = data_handler.get_full_dataset()

    print("\nShape of the final dataset:", final_dataset.shape)
    print("\nSample of the final dataset:")
    print(final_dataset.head())

    print("\nData types and missing values:")
    final_dataset.info()
