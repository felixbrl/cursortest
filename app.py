"""
Streamlit web application for the Credit Downgrade Prediction model.

This app provides an interactive user interface to get predictions.
"""
import configparser
import streamlit as st

# Import the core prediction logic
from predictor import get_prediction
from main import MODEL_PATH # Use the same constant for the model path

# --- Helper Functions ---

def get_available_sectors():
    """Reads the config file to get a list of available sectors."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return [s.capitalize() for s in config['SECTORS'].keys()]


# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Downgrade Prediction",
    page_icon="📉",
    layout="centered"
)


# --- UI Elements ---
st.title("📉 Credit Downgrade Prediction")
st.write(
    "This application uses a machine learning model to predict the probability "
    "of a credit rating downgrade for a given economic sector within the next 6-12 months."
)
st.info("To train a new model, run `python main.py train` from your terminal.", icon="ℹ️")
st.markdown("---")


# Create a container for the inputs
with st.container():
    st.subheader("Select a Sector for Prediction")

    available_sectors = get_available_sectors()
    selected_sector = st.selectbox(
        label="Choose a sector from the list below.",
        options=available_sectors,
        index=0  # Default to the first sector in the list
    )

    predict_button = st.button("Predict Downgrade Probability", type="primary", use_container_width=True)

# Placeholder for the result and status messages
result_placeholder = st.empty()


# --- Backend Logic ---
if predict_button:
    # Use a spinner to show that the process is running
    with st.spinner(f"Running prediction for {selected_sector}..."):
        # Call the refactored prediction logic
        probability = get_prediction(selected_sector, MODEL_PATH)

        if probability is not None:
            # Display the result in a metric card for visual emphasis
            result_placeholder.metric(
                label=f"Probability of Downgrade for {selected_sector}",
                value=f"{probability:.1%}",
                help="This is the model's predicted probability of a credit rating downgrade in the next 6-12 months."
            )
            # Add a success message below the metric
            st.success("Prediction complete.", icon="✅")
        else:
            # Display an error message if prediction failed
            result_placeholder.error(
                "Prediction failed. Please check the console logs. "
                "Ensure a model has been trained by running 'python main.py train' first."
            )
