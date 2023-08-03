"""
This Streamlit page is to prepare the data to the train process.
"""

import streamlit as st

# from src.deepsea.network_architecture.classification.image.cats_vs_dogs.basic import (
#    train,
# )
from network_architecture.image.classification.cats_vs_dogs.feature_extraction_from_pretrained import (
    fast_feat_extract_without_augmentation,
    feat_extract_with_augmentation,
)


def main():
    st.title("Model Preparation")
    st.markdown("Apply data configurtion")
    st.sidebar.title("Image Classification Web App")
    st.sidebar.subheader("Image Data Configuration")

    file_path = st.sidebar.text_input("Image Data Folder Path:")
    st.sidebar.number_input("Set image size:", 28, 255)

    st.sidebar.multiselect("Data augmentation", ("Flip", "Crop", "Rotate"))

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom data set(Classification)")

    # clothes_classification.main()

    # this is for re-organizing cats and dogs images according to train, validation and test
    # img_class_organizer.main()

    # train.cats_vs_dogs()
    # train.evaluate()
    # fast_feat_extract_without_augmentation.run()
    feat_extract_with_augmentation.run()


if __name__ == "__main__":
    st.set_option("deprecation.showPyplotGlobalUse", False)
    main()
