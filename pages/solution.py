import streamlit as st
import pandas as pd
from PIL import Image


def app():
    st.title('SOLUTION')

    st.markdown("**Computer vision** is a field of artificial intelligence (AI) that enables computers \
        and systems to derive meaningful information from digital images, videos and \
            other visual inputs — and take actions or make recommendations based on that information. \
                If AI enables computers to think, computer vision enables them to see, observe and understand.")

    st.markdown("Types of Computer Vision tasks:")

    image_cv_tasks = Image.open(r"data/slide_img/computer vision tasks.jpeg")
    st.image(image_cv_tasks, caption='Computer Vision Tasks', use_column_width="auto")

    st.markdown("""---""")

    st.markdown("### Type of Computer Vision Used")

    st.markdown("We use **Classification** to predict image input data into two classes: \n - NOT_OK (Class 0) : \
        Rejected image, image not represent customer house \n - OK (Class 1): \
            Accepted image, image represent customer house view")

    st.markdown("Example use of neural network in classifying images into cat or dog:")

    st.markdown('<img src="https://miro.medium.com/max/1400/1*BIpRgx5FsEMhr1k2EqBKFg.gif" alt="drawing" width="800"/>',
    unsafe_allow_html=True)

    st.markdown("""---""")

    st.markdown("### Type of Neural Network Used")

    # kasitau secara singkat tentang CNN
    image_cnn_general = Image.open(r"data/slide_img/cnn-general.jpeg")
    st.image(image_cnn_general, caption='Convolutional Neural Network - Illustration', use_column_width="auto")

    st.markdown("A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, \
        assign importance (learnable weights and biases) to various aspects/objects in the image and \
            be able to differentiate one from the other. \
                [source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)")
    
