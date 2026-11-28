# Main entry point and router for the HierVision web app.
# Uses Streamlit navigation to route between pages.

import streamlit as st

st.set_page_config(page_title="HierVision", layout="wide")

pages = {
    "App Pages": [
        st.Page("app_pages/predict.py", title="Predict", default=True),
        st.Page("app_pages/dataset.py", title="Dataset"),
        st.Page("app_pages/training.py", title="Training")
    ]
}

pg = st.navigation(pages)
pg.run()