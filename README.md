# Digital Twin Airways Project

This project builds a data-driven digital twin for human airways using geometry and pressure datasets.

The planned workflow uses Proper Orthogonal Decomposition / Principal Component Analysis (POD/PCA) to compress high-dimensional simulation or measurement snapshots, surrogate modeling to learn relationships between design/input parameters and reduced outputs, and Streamlit visualization to explore predictions interactively.

## Project Structure

```text
Digital_Twin_Airways_Project/
|-- data/
|   |-- geometry/
|   |   |-- Snapshots/
|   |   |-- doe.csv
|   |   |-- points.bin
|   |   |-- settings.json
|   |   `-- outputDefinition.json
|   `-- pressure/
|       |-- Snapshots/
|       |-- doe.csv
|       |-- points.bin
|       |-- settings.json
|       `-- outputDefinition.json
|-- outputs/
|   |-- geometry/
|   |-- pressure/
|   |-- models/
|   |-- figures/
|   `-- predictions/
|-- scripts/
|-- app/
|-- requirements.txt
`-- README.md
```

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train Models

Train geometry and pressure digital twin models:

```bash
python scripts/train_surrogates.py --dataset both
```

For faster experiments, train on a named airway region:

```bash
python scripts/train_surrogates.py --dataset both --selection glotis
```

Trained models are saved in `outputs/models/`, and PCA variance figures are saved in `outputs/figures/`.

## Predict

Predict from a row in the DOE table:

```bash
python scripts/predict.py --dataset pressure --doe-row 0
```

The predicted field is saved as a NumPy array in `outputs/predictions/`.

## Visualize

Launch the Streamlit app:

```bash
python -m streamlit run app/streamlit_app.py
```

## Current Implementation

The project now includes loaders for the airway binary files, POD/PCA compression, random forest surrogate modeling, prediction export, and a Streamlit visualization app.
