"""
This Streamlit page is to prepare the data to the train process.
"""

import streamlit as st
import pandas as pd
import numpy as np
from network_architecture.classification.image import load_data


def main():
    st.title('Model Preparation')
    st.markdown("Apply data configurtion")
    st.sidebar.title("Image Classification Web App")
    st.sidebar.subheader("Image Data Configuration")

    file_path = st.sidebar.text_input("Image Data Folder Path:")
    st.sidebar.number_input("Set image size:", 28, 255)

    st.sidebar.multiselect("Data augmentation", ('Flip', 'Crop', 'Rotate'))

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom data set(Classification)")


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
