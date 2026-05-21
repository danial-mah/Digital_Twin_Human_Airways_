"""Streamlit app for exploring trained human airways digital twin models.

The app loads a saved model, creates sliders for the model input parameters,
predicts an airway field, and visualizes the predicted values on the mesh.
"""

# This import enables modern type-hint behavior.
from __future__ import annotations

# Path provides clean filesystem path handling.
from pathlib import Path

# joblib loads trained model files saved by scripts/train_surrogates.py.
import joblib

# matplotlib creates the 3D scatter plot and histogram.
import matplotlib.pyplot as plt

# NumPy handles numeric arrays and statistics.
import numpy as np

# streamlit builds the interactive web app interface.
import streamlit as st


# PROJECT_ROOT points to the main project folder, one level above app/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MODELS_DIR is where trained .joblib models are saved.
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


# Cache the loaded model so Streamlit does not reload it on every slider change.
@st.cache_resource
def load_model(path: Path) -> dict:
    # Load and return the selected trained model dictionary.
    return joblib.load(path)


def available_models() -> list[Path]:
    # Find every trained digital twin model in outputs/models.
    return sorted(MODELS_DIR.glob("*_digital_twin.joblib"))


# Configure the browser tab title and use a wide page layout.
st.set_page_config(page_title="Airways Digital Twin", layout="wide")

# Show the main app title at the top of the page.
st.title("Human Airways Digital Twin")

# Get the list of trained models available on disk.
models = available_models()

# If no models exist, show a helpful message and stop the app.
if not models:
    st.info("No trained models found. Run `python scripts/train_surrogates.py --dataset both --selection glotis` first.")
    st.stop()

# Add a sidebar dropdown so the user can choose which trained model to load.
selected = st.sidebar.selectbox("Model", models, format_func=lambda path: path.stem.replace("_digital_twin", ""))

# Load the selected model from disk.
model = load_model(selected)

# Add a sidebar heading for the input controls.
st.sidebar.subheader("Input Parameters")

# Create an empty dictionary that will store the current slider values.
values = {}

# Loop through every input feature the model expects.
for column in model["feature_columns"]:
    # Read the minimum training value for this feature.
    min_value = float(model["feature_min"][column])

    # Read the maximum training value for this feature.
    max_value = float(model["feature_max"][column])

    # Use the median training value as the default slider position.
    default = float(model["feature_defaults"][column])

    # Create a slider and store the selected value.
    values[column] = st.sidebar.slider(column, min_value, max_value, default)

# Build a one-row NumPy input matrix in exactly the order the model expects.
x = np.array([[values[column] for column in model["feature_columns"]]], dtype=np.float64)

# Predict the PCA latent coordinates from the selected input values.
latent = model["surrogate"].predict(x)

# Reconstruct the full predicted field from the PCA latent coordinates.
prediction = model["pca"].inverse_transform(latent)[0].reshape(-1, model["components_per_node"])

# Load the mesh points saved inside the trained model.
points = model["points"]

# Create two columns for summary information and prediction statistics.
left, right = st.columns([1, 1])

# Fill the left column with model information.
with left:
    # Add a heading for the model summary.
    st.subheader("Model Summary")

    # Show which dataset this model represents.
    st.metric("Dataset", model["dataset"])

    # Show how many PCA/POD modes the model uses.
    st.metric("PCA modes", model["metrics"]["pca_components"])

    # Show the total variance captured by the PCA modes.
    st.metric("Explained variance", f"{model['metrics']['explained_variance_ratio_sum']:.3f}")

    # Show the mean absolute error measured during validation.
    st.metric("Validation MAE", f"{model['metrics']['field_mae']:.4g}")

# Fill the right column with statistics for the current prediction.
with right:
    # Add a heading for prediction statistics.
    st.subheader("Prediction Statistics")

    # For scalar fields use the first column; for vector fields use vector magnitude.
    scalar = prediction[:, 0] if prediction.shape[1] == 1 else np.linalg.norm(prediction, axis=1)

    # Use the stored physical unit if it exists; otherwise use an empty string.
    unit = model["unit"] or ""

    # Show the minimum predicted value.
    st.metric("Minimum", f"{float(np.min(scalar)):.4g} {unit}")

    # Show the average predicted value.
    st.metric("Mean", f"{float(np.mean(scalar)):.4g} {unit}")

    # Show the maximum predicted value.
    st.metric("Maximum", f"{float(np.max(scalar)):.4g} {unit}")

# Add a heading above the 3D preview plot.
st.subheader("Predicted Field Preview")

# Choose a stride so very large meshes are downsampled for faster plotting.
stride = max(1, len(points) // 30000)

# Keep every stride-th point for plotting.
sample_points = points[::stride]

# Keep the matching predicted values for those sampled points.
sample_values = scalar[::stride]

# Create a matplotlib figure for the 3D scatter plot.
fig = plt.figure(figsize=(9, 6))

# Add a 3D axis to the figure.
ax = fig.add_subplot(111, projection="3d")

# Draw sampled mesh points colored by predicted field value.
plot = ax.scatter(
    sample_points[:, 0],
    sample_points[:, 1],
    sample_points[:, 2],
    c=sample_values,
    s=1,
    cmap="viridis",
)

# Label the x axis.
ax.set_xlabel("X")

# Label the y axis.
ax.set_ylabel("Y")

# Label the z axis.
ax.set_zlabel("Z")

# Add a plot title using the field name from the model.
ax.set_title(f"{model['field_name']} preview")

# Add a colorbar showing what the colors mean.
fig.colorbar(plot, ax=ax, shrink=0.65, label=model["unit"] or model["field_name"])

# Render the matplotlib figure inside Streamlit.
st.pyplot(fig, clear_figure=True)

# Add a heading above the histogram.
st.subheader("Field Distribution")

# Create a separate matplotlib figure for the histogram.
hist_fig, hist_ax = plt.subplots(figsize=(9, 3))

# Plot a histogram of all predicted scalar values.
hist_ax.hist(scalar, bins=60, color="#2a9d8f", edgecolor="white")

# Label the histogram x axis with the unit or field name.
hist_ax.set_xlabel(model["unit"] or model["field_name"])

# Label the histogram y axis with node count.
hist_ax.set_ylabel("Node count")

# Add a light grid so the histogram is easier to read.
hist_ax.grid(True, alpha=0.2)

# Render the histogram inside Streamlit.
st.pyplot(hist_fig, clear_figure=True)
