import streamlit as st
import pandas as pd
from PIL import Image
import json

def app():
    st.title('Model Architecture and Tuning')

    st.markdown("We use several CNN architecture and compare them later to find out which model gives the best performance. \
        \n Our own made CNN architecture is marked as `BASE MODEL`, we compare them to two of best CNN architecture : `Inception V3` and `VGG16`. \
        \n\n Hyperparameter tuning that we applied to our models are as follows: \n - number \
        of Convolutional and Pooling Layer : 1, 2 , or 3. \n - optimizer \
         function : ADAM or RMSProp ")
         
    st.markdown("\n \n The model\
         is ran in 50 epochs, model is saved with `keras.callbacks.ModelCheckpoint` when improvement in \
        `validation_loss` is reached. Model is saved in `.hdf5` format.")

    model_dict = {'Base model 1 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3_1CONV.xlsx',
            'Base model 1 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3_1CONV.xlsx',
            'Base model 2 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3.xlsx',
            'Base model 2 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3.xlsx',
            'Base model 3 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3_3CONV.xlsx',
            'Base model 3 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3_3CONV.xlsx',
            'Inception V3 model (50 epoch) ADAM':'golbirev_INCV3_RGB_with_tpu_50ep_ADAM_LR10e-3_DENSE_2.xlsx',
            'Inception V3 model (50 epoch) RMSPROP':'golbirev_INCV3_RGB_with_tpu_50ep_RMSPROP_LR10e-3_DENSE_2.xlsx',
            'VGG16 model (50 epoch) ADAM':'golbirev_VGG16_with_tpu_50ep_ADAM_LR10e-3.xlsx',
            'VGG16 model (50 epoch) RMSPROP':'golbirev_VGG16_with_tpu_50ep_RMSPROP_LR10e-3.xlsx'}

    st.markdown("""---""")

    option = st.selectbox(
     'Which model do you want to view?',
     (model_dict.keys()))

    st.write('You picked:', option)

    st.markdown("""---""")

  
    # Opening JSON file
    with open(r"data/model_info/"+option+".json", 'r') as openfile:
    
        # Reading from json file
        json_object = json.load(openfile)

    show_all_json = st.checkbox('Show model json')

    if show_all_json:
        st.write(json_object)

    st.markdown("""---""")

    image_arch = Image.open(r"data/model_info/"+option+".png")
    st.image(image_arch, caption='Model Architecture', use_column_width="auto")