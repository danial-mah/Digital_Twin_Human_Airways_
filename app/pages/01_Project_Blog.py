"""Streamlit blog-style teaching page for the digital twin project.

This page is more than a plain text reader. It shows Markdown lessons from
docs/blog/ and adds visual material such as pipeline diagrams, dataset cards,
PCA plots, and a simple airway point-cloud projection when data is available.

Beginner example:
The Markdown files are the written lesson. This Streamlit page is the classroom
view: it adds pictures, charts, and summaries around the lesson so the project
is easier to understand.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# Path is used to locate the project root, blog files, data files, and figures.
from pathlib import Path

# matplotlib creates small teaching plots for the blog page.
import matplotlib.pyplot as plt

# NumPy reads binary point data and matrix shapes.
import numpy as np

# pandas reads CSV summary files when they exist.
import pandas as pd

# Streamlit builds the interactive blog page.
import streamlit as st


# PROJECT_ROOT points to the main project folder, two levels above this page file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BLOG_DIR points to the folder that stores the Markdown blog posts.
BLOG_DIR = PROJECT_ROOT / "docs" / "blog"

# DATA_ROOT points to the original geometry and pressure datasets.
DATA_ROOT = PROJECT_ROOT / "data"

# OUTPUT_ROOT points to generated matrices, figures, models, and predictions.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def list_blog_posts() -> list[Path]:
    """Return all Markdown blog files in sorted order."""

    # glob("*.md") finds every Markdown file in docs/blog/.
    # sorted(...) keeps the numbered posts in reading order.
    return sorted(BLOG_DIR.glob("*.md"))


def readable_title(path: Path) -> str:
    """Convert a filename into a clean title for the sidebar."""

    # path.stem removes the .md extension.
    # replace("_", " ") makes the filename easier to read.
    # title() capitalizes the words.
    return path.stem.replace("_", " ").title()


def file_size_mb(path: Path) -> float | None:
    """Return file size in megabytes, or None if the file is missing."""

    # If the file does not exist, there is no size to show.
    if not path.exists():
        return None

    # stat().st_size gives bytes; dividing converts bytes to megabytes.
    return path.stat().st_size / (1024 * 1024)


@st.cache_data
def matrix_shape(path_text: str) -> tuple[int, ...] | None:
    """Read the shape of a .npy matrix without loading all data into memory."""

    # Convert the string path back into a Path object.
    path = Path(path_text)

    # If the matrix does not exist yet, return None.
    if not path.exists():
        return None

    # mmap_mode="r" opens the array in read-only memory-mapped mode.
    # This lets us inspect shape without loading gigabytes into RAM.
    matrix = np.load(path, mmap_mode="r")

    # Return the matrix shape as a normal tuple.
    return tuple(matrix.shape)


@st.cache_data
def load_airway_points_sample(max_points: int = 25000) -> np.ndarray | None:
    """Load a small sample of airway points for a teaching visualization."""

    # Use pressure points because it shares the airway mesh location information.
    points_path = DATA_ROOT / "pressure" / "points.bin"

    # If the file is missing, the plot cannot be created.
    if not points_path.exists():
        return None

    # The dataset stores points.bin as float64 with one leading header value.
    raw_values = np.fromfile(points_path, dtype=np.float64)

    # Remove the leading header value and reshape remaining values into x, y, z points.
    clean_values = raw_values[1:]

    # If values do not form triples, return None instead of crashing.
    if clean_values.size % 3 != 0:
        return None

    # Reshape the flat vector into a point cloud with columns x, y, z.
    points = clean_values.reshape(-1, 3)

    # Downsample points so the blog page remains fast.
    stride = max(1, points.shape[0] // max_points)

    # Return a sampled point cloud as float32 for smaller memory use.
    return points[::stride].astype(np.float32)


def show_pipeline_diagram() -> None:
    """Show a visual graph of the full digital twin pipeline."""

    # graphviz_chart creates a simple flow diagram from text.
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#eef6ff", color="#6b7280"];
            data [label="Raw Data\\ngeometry + pressure"];
            matrix [label="Snapshot Matrices\\nrows = snapshots"];
            pca [label="POD/PCA\\nlarge fields -> 20 coefficients"];
            surrogate [label="Surrogate Models\\nDOE -> coefficients"];
            reconstruct [label="Reconstruction\\ncoefficients -> fields"];
            dashboard [label="Dashboard\\nplots + proxy metrics"];
            data -> matrix -> pca -> surrogate -> reconstruct -> dashboard;
        }
        """
    )


def show_dataset_cards() -> None:
    """Show small cards summarizing important project files."""

    # Paths to generated matrices.
    geometry_matrix = OUTPUT_ROOT / "geometry" / "geometry_snapshot_matrix.npy"
    pressure_matrix = OUTPUT_ROOT / "pressure" / "pressure_snapshot_matrix.npy"

    # Paths to DOE files.
    geometry_doe = DATA_ROOT / "geometry" / "doe.csv"
    pressure_doe = DATA_ROOT / "pressure" / "doe.csv"

    # Create four compact columns for important facts.
    col1, col2, col3, col4 = st.columns(4)

    # Show geometry matrix shape if it exists.
    col1.metric("Geometry Matrix", str(matrix_shape(str(geometry_matrix)) or "not built"))

    # Show pressure matrix shape if it exists.
    col2.metric("Pressure Matrix", str(matrix_shape(str(pressure_matrix)) or "not built"))

    # Show geometry DOE file size.
    geometry_size = file_size_mb(geometry_doe)
    col3.metric("Geometry DOE", "missing" if geometry_size is None else f"{geometry_size:.2f} MB")

    # Show pressure DOE file size.
    pressure_size = file_size_mb(pressure_doe)
    col4.metric("Pressure DOE", "missing" if pressure_size is None else f"{pressure_size:.2f} MB")


def show_airway_projection() -> None:
    """Show a simple 2D projection of the airway point cloud."""

    # Load a downsampled point cloud.
    points = load_airway_points_sample()

    # If points are unavailable, explain why the visual is missing.
    if points is None:
        st.info("Airway point-cloud preview is unavailable because points.bin could not be loaded.")
        return

    # Create a figure for an x-z projection.
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot x versus z so learners can see the airway shape as a point cloud.
    ax.scatter(points[:, 0], points[:, 2], s=1, color="#2a9d8f", alpha=0.7)

    # Label the axes.
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Z coordinate")

    # Add a teaching title.
    ax.set_title("Airway Point Cloud Preview")

    # Keep the geometry proportions realistic.
    ax.set_aspect("equal", adjustable="box")

    # Add a light grid for readability.
    ax.grid(True, alpha=0.2)

    # Keep labels inside the figure.
    fig.tight_layout()

    # Display the figure in Streamlit.
    st.pyplot(fig, clear_figure=True)


def show_pca_figures() -> None:
    """Show PCA explained-variance plots if script 04 has created them."""

    # Paths to PCA figures created by script 04.
    figure_paths = [
        OUTPUT_ROOT / "figures" / "geometry_explained_variance.png",
        OUTPUT_ROOT / "figures" / "pressure_explained_variance.png",
    ]

    # Filter to existing figures only.
    existing = [path for path in figure_paths if path.exists()]

    # If no PCA plots exist yet, show a helpful note.
    if not existing:
        st.info("PCA explained-variance figures are not available yet. Run script 04 to create them.")
        return

    # Create one column per existing figure.
    columns = st.columns(len(existing))

    # Display each PCA plot.
    for column, path in zip(columns, existing):
        column.image(str(path), caption=path.name, use_container_width=True)


def show_results_tables() -> None:
    """Show model-result or proxy-metric CSV files when available."""

    # Useful CSV files generated by later pipeline steps.
    csv_paths = [
        OUTPUT_ROOT / "geometry" / "geometry_surrogate_results.csv",
        OUTPUT_ROOT / "pressure" / "pressure_surrogate_results.csv",
        OUTPUT_ROOT / "predictions" / "prediction_error_row0.csv",
        OUTPUT_ROOT / "predictions" / "proxy_physics_row0.csv",
    ]

    # Keep only CSV files that exist.
    existing = [path for path in csv_paths if path.exists()]

    # If no result tables exist yet, do not show an empty section.
    if not existing:
        st.info("No result tables are available yet. They appear after running scripts 05, 06, or 08.")
        return

    # Add a tab for each available CSV file.
    tabs = st.tabs([path.stem for path in existing])

    # Load and display each CSV table.
    for tab, path in zip(tabs, existing):
        with tab:
            table = pd.read_csv(path)
            st.dataframe(table, use_container_width=True)


def show_visual_learning_panel(selected_post: Path) -> None:
    """Show visuals related to the selected blog topic."""

    # Use the filename to decide which visuals are most helpful.
    name = selected_post.name

    # The index page benefits from seeing the whole workflow.
    if name.startswith("00"):
        st.subheader("Pipeline Overview")
        show_pipeline_diagram()
        st.subheader("Current Project Artifacts")
        show_dataset_cards()

    # Dataset and binary pages benefit from seeing the point cloud.
    elif name.startswith("01") or name.startswith("02") or name.startswith("03"):
        st.subheader("Dataset Snapshot")
        show_dataset_cards()
        st.subheader("Airway Geometry Preview")
        show_airway_projection()

    # PCA page benefits from explained-variance figures.
    elif name.startswith("04"):
        st.subheader("PCA Explained Variance")
        show_pca_figures()

    # Surrogate page benefits from model comparison tables.
    elif name.startswith("05"):
        st.subheader("Model Result Tables")
        show_results_tables()

    # Prediction/dashboard page benefits from result tables and point-cloud preview.
    elif name.startswith("06"):
        st.subheader("Prediction And Proxy Outputs")
        show_results_tables()
        st.subheader("Airway Point Cloud Preview")
        show_airway_projection()


def main() -> None:
    """Render the blog page."""

    # Configure this page with a wide layout for comfortable reading.
    st.set_page_config(page_title="Project Blog", layout="wide")

    # Add a small responsive style block so the two-column lesson layout stacks on mobile.
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            h1 {
                font-size: 1.6rem !important;
            }
            h2, h3 {
                font-size: 1.15rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Show the page title.
    st.title("Project Learning Blog")

    # Explain what this page is for.
    st.caption("A visual guide to the human airways digital twin pipeline.")

    # Add a visible top navigation button back to the main dashboard.
    # This helps users move between the learning blog and the interactive app.
    st.page_link("airways_digital_twin_app.py", label="Back To Dashboard")

    # Find available blog posts.
    posts = list_blog_posts()

    # If no posts exist, show a helpful message instead of crashing.
    if not posts:
        st.error(f"No blog posts found in `{BLOG_DIR}`.")
        return

    # Create a sidebar selector so the reader can choose a topic.
    selected_post = st.sidebar.selectbox(
        "Choose a topic",
        posts,
        format_func=readable_title,
    )

    # Read the selected Markdown file as text.
    markdown_text = selected_post.read_text(encoding="utf-8")

    # Split page into lesson text and visual learning panel.
    lesson_col, visual_col = st.columns([1.15, 0.85])

    # Display the written blog lesson on the left.
    with lesson_col:
        st.markdown(markdown_text)

    # Display visuals and tables on the right.
    with visual_col:
        show_visual_learning_panel(selected_post)


# This makes main() run only when Streamlit opens this page.
if __name__ == "__main__":
    main()
