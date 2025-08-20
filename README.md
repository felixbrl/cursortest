# Sector Credit Rating Downgrade Prediction

## Project Overview

This project provides a modular and extensible Python pipeline to predict the probability of a credit rating downgrade for a specific economic sector within the next 6-12 months. It is designed with best practices for financial modeling, including time-series feature engineering, proper model evaluation, and a clear separation of concerns between data handling, feature engineering, and modeling.

The pipeline ingests macroeconomic data, sector-level financial data, and historical downgrade events to train a machine learning model (Logistic Regression or Random Forest). It is built to be configurable and can be easily adapted to use new data sources.

**Note on Current Data Sources:** Due to unreliability of the `yfinance` API in the development environment, the microeconomic (sector-level financial ratios) data source is currently a **mocked implementation** inside `data_handler.py`. This allows for a full demonstration of the pipeline's functionality. The module is designed to be easily updated to point to a production-grade data source (e.g., a CSV file or a different API) when available.

## Installation

To set up the environment and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Specification

The model requires a historical record of credit rating downgrades provided in a CSV file.

*   **File:** `historical_downgrades.csv`
*   **Format:** The CSV must contain the following columns:
    *   `Date`: The date of the observation (e.g., quarter-end) in `YYYY-MM-DD` format.
    *   `Sector`: The name of the economic sector (must match a sector defined in `config.ini`).
    *   `Downgrade_Occurred`: A binary flag, where `1` indicates a downgrade occurred in that quarter and `0` indicates no downgrade.

**Example:**
```csv
Date,Sector,Downgrade_Occurred
2022-12-31,Technology,1
2023-03-31,Technology,0
2023-03-31,Financials,1
```

## Configuration

All project settings are managed in the `config.ini` file.

*   **[API_KEYS]**: A placeholder section for API keys. While the primary micro-data source is currently mocked, this can be used for future integrations.
*   **[SECTORS]**: Define the economic sectors and their constituent company stock tickers. The script uses these tickers to fetch financial data (when not using the mock source).
    ```ini
    [SECTORS]
    Technology = AAPL,MSFT,GOOG
    ```
*   **[MACRO_INDICATORS]**: Map user-friendly names to their corresponding codes on the FRED (Federal Reserve Economic Data) database.
    ```ini
    [MACRO_INDICATORS]
    GDP = GDP
    CPI = CPIAUCSL
    ```
*   **[MODEL_PARAMETERS]**: Set parameters for the modeling process.
    *   `TEST_START_DATE`: The date that marks the beginning of the test set for time-series cross-validation. Data before this date is used for training, and data on or after this date is used for testing.

## How to Run

The pipeline is controlled via the `main.py` script, which provides two main commands.

### 1. Training a New Model

To run the full pipeline, which includes data ingestion, feature engineering, model training, evaluation, and saving the model, use the `train` command. The trained model will be saved as `downgrade_model.joblib`.

```bash
python main.py train
```
The script will print a detailed evaluation report, including an AUC-ROC score, confusion matrix, classification report, and the top 5 most important features.

### 2. Running a Prediction

To use the pre-trained model to predict the downgrade probability for a specific sector, use the `predict` command.

```bash
python main.py predict --sector <SectorName>
```

Replace `<SectorName>` with the name of a sector defined in your `config.ini` (e.g., `Technology`, `Financials`).

**Example:**
```bash
python main.py predict --sector Technology
```

**Output:**
```
--- Prediction Result ---
The probability of a credit rating downgrade for the Technology sector in the next 6-12 months is: 49.0%
```

### 3. Using the Web Interface

This project includes a simple web-based dashboard for interactive predictions.

1.  **Ensure dependencies are installed:** If you haven't already, install streamlit:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```
This will start the web server and provide a local URL to open the application in your browser. From there, you can select a sector from the dropdown menu and click the "Predict" button to see the result.

## Methodology

*   **Data Sources**:
    *   **Macroeconomic**: Data is sourced from the Federal Reserve Economic Data (FRED) via the `pandas-datareader` library. Key indicators include GDP, CPI, Federal Funds Rate, and Unemployment Rate.
    *   **Microeconomic (Sector-Aggregated)**: This data is **currently mocked** for reliability. The mock function generates realistic financial ratios (Leverage, Profitability, Liquidity) for each sector. The system is designed to easily swap this for a real data source.
    *   **Target Variable**: Historical downgrade events are loaded from `historical_downgrades.csv`.

*   **Feature Engineering**:
    *   The pipeline creates time-series features, including 2-quarter and 4-quarter moving averages and rates of change (momentum) for all indicators.
    *   To prevent lookahead bias, the target variable is engineered to be forward-looking. The features for a given quarter are used to predict whether a downgrade will occur in the following 6-to-12-month window.

*   **Modeling**:
    *   The data is split into training and testing sets based on a fixed date to respect the time-series nature of the data.
    *   Features are scaled using `StandardScaler`, which is fit only on the training data.
    *   The default model is a `RandomForestClassifier`, chosen for its ability to handle non-linear relationships and its robustness. The code can be easily switched to use `LogisticRegression`.
    *   The model is evaluated using metrics suitable for imbalanced datasets, including AUC-ROC.
