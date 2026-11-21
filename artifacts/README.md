# Model artifacts

This directory is intentionally kept free of generated model files in the source repository.

To run the Streamlit app:

1. Execute `notebooks/household_energy_forecasting.ipynb` in Google Colab.
2. Download the generated `streamlit_artifacts.zip`.
3. Extract its `artifacts/` directory here.

Expected files include `best_model.joblib`, `hourly_history.csv`, `metadata.json`,
`test_metrics.csv`, `validation_metrics.csv`, and `test_predictions.csv`.

Do not commit private data, credentials, or untrusted `.joblib`/pickle files.
