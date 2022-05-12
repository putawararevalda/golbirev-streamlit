import streamlit as st
import pandas as pd
from PIL import Image



def app():
    st.title('ABOUT')

    st.markdown("Automation to authenticate Indihome installation evidence")

    image_back = Image.open(r"data/slide_img/background.png")
    st.image(image_back, caption='Background', use_column_width="auto")

    st.markdown("We need a system that **automatically verify** incoming installation evidence by **classifying whether** that image is accepted **(OK)** or rejected **(NOT_OK)**")

    image_obj = Image.open(r"data/slide_img/objective.png")
    st.image(image_obj, caption='Objective', use_column_width="auto")
    

    