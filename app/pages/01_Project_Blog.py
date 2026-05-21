"""Streamlit blog-style teaching page for the digital twin project.

This page reads Markdown files from docs/blog/ and displays them in Streamlit.
It gives the project a small educational blog inside the app.

Beginner example:
The blog posts are normal Markdown files. This page simply lists them in a
dropdown menu and displays the selected file. It is like a tiny local website
inside Streamlit for explaining the project.

If you add a new file such as:

    docs/blog/07_new_topic.md

it will automatically appear in the dropdown the next time the page refreshes.
"""

# This line allows Python type hints to be handled in a modern, flexible way.
from __future__ import annotations

# Path is used to locate the project root and Markdown blog files.
from pathlib import Path

# Streamlit builds the interactive blog page.
import streamlit as st


# PROJECT_ROOT points to the main project folder, two levels above this page file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BLOG_DIR points to the folder that stores the Markdown blog posts.
BLOG_DIR = PROJECT_ROOT / "docs" / "blog"


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


def main() -> None:
    """Render the blog page."""

    # Configure this page with a wide layout for comfortable reading.
    st.set_page_config(page_title="Project Blog", layout="wide")

    # Show the page title.
    st.title("Project Learning Blog")

    # Explain what this page is for.
    st.caption("A guided explanation of the human airways digital twin pipeline.")

    # Add a visible top navigation button back to the main dashboard.
    # This helps users move between the learning blog and the interactive app.
    st.page_link("airways_digital_twin_app.py", label="Back To Dashboard", icon="⬅️")

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

    # Display the Markdown content in the Streamlit page.
    st.markdown(markdown_text)


# This makes main() run only when Streamlit opens this page.
if __name__ == "__main__":
    main()
