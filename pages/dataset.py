
import streamlit as st
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt


def app():
    st.title('DATASET')

    st.markdown("Dataset consists of images with label : OK and NOT_OK")

    option = st.selectbox(
     'Which data do you want to view?',
     ('Train dataset sample', 'Test dataset sample','OK sample'))

    st.write('You picked:', option)

    sample_dict = {"OK sample":r"data/image/OK/OK (18).jpg",
    "Train dataset sample":r"data/image/train_with_aug.png",
    "Test dataset sample":r"data/image/test_without_aug_nok.png"}

    image = Image.open(sample_dict[option])
    st.image(image, caption='Sample Image', use_column_width="auto")

    #plt.figure()
    #plt.title("A")
    #plt.imshow(image)

    #fig, ax = plt.subplots()
    #ax.imshow(image)

    #st.pyplot(fig)

    

    # tunjukin foto foto OK dan NOT OK disini