# Provides the web interface for the HierVision classifier.
# Handles image uploads, threshold configuration, and displays the hierarchical
# prediction steps (including Grad-CAM heatmaps and charts) column by column.

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from typing import List, Tuple

import config
from hiervision import pipeline


@st.cache_resource
def load_base_model() -> None:
    # Caches the heavy VGG16 backbone in Streamlit memory
    # to prevent reloading it during every UI interaction.
    pipeline.get_cached_backbone()


def plot_top3(top3: List[Tuple[str, float]]) -> Figure:
    # Plots a bar chart of the top 3 predictions using matplotlib.
    # This avoids external data-wrangling libraries like pandas.
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = [item[0] for item in top3]
    scores = [item[1] for item in top3]

    ax.bar(labels, scores, color="skyblue")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    return fig


def run_prediction(img: Image.Image, threshold: float) -> None:
    # Runs the inference pipeline and builds the step-by-step UI columns.
    with st.spinner("Analyzing image..."):
        result = pipeline.predict(img, threshold=threshold, explain_flag=True)

    if isinstance(result, str):
        st.error(result)
        return

    st.success(f"**Final Label:** {result.final_label} | **Status:** {result.status}")

    # Distribute the step explanations evenly across columns
    cols = st.columns(len(result.steps))

    for col, step in zip(cols, result.steps):
        with col:
            st.subheader(f"Node: '{step.node or 'root'}'")
            st.metric(
                label="Prediction",
                value=step.label,
                delta=f"{step.confidence:.1%} conf",
                delta_color="off"
            )

            if step.heatmap is not None:
                st.image(step.heatmap, caption=f"Grad-CAM: {step.label}", use_container_width=True)

            st.write("**Top 3 Probabilities**")
            fig = plot_top3(step.top3)
            st.pyplot(fig)


def main() -> None:
    st.title("HierVision: Hierarchical Image Classifier")

    # Friendly message if no models have been trained yet
    if pipeline.get_cached_head("") is None:
        st.info("\U0001f44b Welcome! It looks like no model has been trained yet. Please train the root node to get started.")
        return

    load_base_model()

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(config.THRESHOLD),
        help="Minimum confidence required to proceed to a subcategory."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            run_prediction(img, threshold)


if __name__ == "__main__":
    main()