import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import time

import os

import requests

from tempfile import NamedTemporaryFile

fig = plt.figure()

#with open("custom.css") as f:
    #st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def app():
    st.title('Golbirev House/No House Classifier')

    st.markdown("Welcome to this simple web application that classifies houses.  \n Created by : **Golbirev Team**, TelkomAthon #3 2022. Go Golbirev!")
    st.markdown("You can predict image classes here or via our telegram bot:  \n  [🤖 telegram bot 🤖](https://t.me/golbirev_api_bot)")

    st.markdown("### UPLOAD IMAGE THAT YOU WANT TO CLASSIFY HERE")

    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    temp_file = NamedTemporaryFile(delete=False) #added

    class_btn = st.button("Classify")
    
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        temp_file.write(file_uploaded.getvalue()) #added

        
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
            result = "ERROR!"
        else:

            files = {'file': (os.path.basename(temp_file.name), open(temp_file.name, 'rb'), 'application/octet-stream')}

            r2 = requests.post("https://golbirev-01.herokuapp.com/predict", files=files)

            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                json_result = r2.json()
                time.sleep(1)
                st.success('Classified')
                
                #st.pyplot(fig)

                st.write("Model Output :")
                st.write(json_result["prediction_result"]["model_output"])

                col1, col2= st.columns(2)
                
                col1.metric("Predicted Class", json_result["prediction_result"]["predicted_class"])
                col2.metric("Confidence Percentage", "{} %".format(json_result["prediction_result"]["confidence_percentage"]))

                #show_all_json = st.checkbox('Show full API response')

                #if show_all_json:
                    #st.write("JSON reply from API https://golbirev-01.herokuapp.com/predict : ")
                    #st.write(json_result)


                


    return ""
    

if __name__ == "__main__":
    main()
