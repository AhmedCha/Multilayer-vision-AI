# Provides the web interface for the HierVision classifier.
# Handles image uploads, threshold configuration, and displays the hierarchical
# prediction steps (Grad-CAM heatmaps and top-3 bars) column by column.

import streamlit as st
from PIL import Image, ImageOps

import config
from hiervision import pipeline


@st.cache_resource
def load_base_model() -> None:
    # Keeps the heavy VGG16 backbone in memory between Streamlit reruns.
    pipeline.get_cached_backbone()


def run_prediction(img: Image.Image, threshold: float) -> None:
    # Runs the cascade and keeps the result, and the threshold it used, in session state.
    with st.spinner("Analyzing image..."):
        try:
            result = pipeline.predict(img, threshold=threshold, with_heatmaps=True)
        except FileNotFoundError as e:
            st.error(str(e))
            return
    st.session_state["prediction_result"] = result
    st.session_state["result_threshold"] = threshold


def show_step(step: pipeline.Step) -> None:
    # Draws one column of the explanation: label, confidence, heatmap and top 3.
    st.subheader(f"Node: '{step.node or 'root'}'")
    st.metric(
        label="Prediction",
        value=step.label,
        delta=f"{step.confidence:.1%} conf",
        delta_color="off",
    )
    if not step.accepted:
        st.caption("\u26a0\ufe0f Below threshold, not used")
    if step.heatmap is not None:
        st.image(step.heatmap, caption=f"Grad-CAM: {step.label}", use_container_width=True)
    st.write("**Top 3 Probabilities**")
    for label, prob in step.top3:
        st.write(f"{label} ({prob:.1%})")
        st.progress(prob)


def show_result(result: pipeline.Result) -> None:
    # Shows the final label, then one column per step of the cascade.
    st.success(f"**Final Label:** {result.final_label} | **Status:** {result.status}")
    cols = st.columns(len(result.steps))
    for col, step in zip(cols, result.steps):
        with col:
            show_step(step)


def main() -> None:
    st.title("HierVision: Hierarchical Image Classifier")

    with st.sidebar:
        if st.button("Reload models", key="reload_button"):
            pipeline.reload_models()
            st.success("Models reloaded successfully!")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(config.THRESHOLD),
            help="Minimum confidence required to proceed to a subcategory.",
        )

    # Friendly message if no models have been trained yet.
    if pipeline.get_cached_head("") is None:
        st.info("\U0001f44b Welcome! No model has been trained yet. Train the root node to get started.")
        return

    load_base_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        return

    # Fix phone-photo rotation (EXIF) and make sure the image is RGB.
    img = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Forget the old result when a different file is uploaded.
    if st.session_state.get("last_file_id") != uploaded_file.file_id:
        st.session_state["last_file_id"] = uploaded_file.file_id
        st.session_state.pop("prediction_result", None)

    # ONE Predict button: it is True only on the run right after the click.
    if st.button("Predict", key="predict_button"):
        run_prediction(img, threshold)

    result = st.session_state.get("prediction_result")
    if result is None:
        return

    # The slider does not re-run the model, so tell the user when the result is out of date.
    used = st.session_state.get("result_threshold", threshold)
    if abs(used - threshold) > 1e-9:
        st.warning(f"This result used a threshold of {used:.2f}. Press Predict to apply {threshold:.2f}.")

    show_result(result)


if __name__ == "__main__":
    main()