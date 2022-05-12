import streamlit as st
import pandas as pd
import numpy as np

import plotly.figure_factory as ff
import plotly.express as px

def app():

    model_hist_dict = {'Base model 1 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3_1CONV.xlsx',
              'Base model 1 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3_1CONV.xlsx',
              'Base model 2 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3.xlsx',
              'Base model 2 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3.xlsx',
              'Base model 3 CONV (50 epoch) ADAM':'golbirev_vanilla_with_tpu_50ep_ADAM_LR10e-3_3CONV.xlsx',
              'Base model 3 CONV (50 epoch) RMSPROP':'golbirev_vanilla_with_tpu_50ep_RMS_LR10e-3_3CONV.xlsx',
              'Inception V3 model (50 epoch) ADAM':'golbirev_INCV3_RGB_with_tpu_50ep_ADAM_LR10e-3_DENSE_2.xlsx',
              'Inception V3 model (50 epoch) RMSPROP':'golbirev_INCV3_RGB_with_tpu_50ep_RMSPROP_LR10e-3_DENSE_2.xlsx',
              'VGG16 model (50 epoch) ADAM':'golbirev_VGG16_with_tpu_50ep_ADAM_LR10e-3.xlsx',
              'VGG16 model (50 epoch) RMSPROP':'golbirev_VGG16_with_tpu_50ep_RMSPROP_LR10e-3.xlsx'}

    st.title('MODEL AND TRAINING HISTORY')

    st.markdown("How our model get into its final form")
    st.markdown("""---""")

    option = st.selectbox(
     'Which model do you want to view?',
     (model_hist_dict.keys()))

    st.write('You picked:', option)

    history_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[option],
    #index_col=0,
    sheet_name = "Fit History")

    history_df = history_df.rename(columns={"Unnamed: 0":"Epoch"})

    st.dataframe(history_df)
    st.markdown("""---""")

    idx_min_vl = int(history_df[['val_loss']].idxmin().iloc[0])

    col1, col2, col3= st.columns(3)
    
    col1.metric("Epoch with lowest val_loss", int(history_df.at[idx_min_vl,"Epoch"]))
    col2.metric("Val_loss", "{:.6f}".format(float(history_df.at[idx_min_vl,"val_loss"])))
    col3.metric("Val_accuracy", "{:.2f} %".format(float(history_df.at[idx_min_vl,"val_accuracy"]*100)))

    st.markdown("""---""")
    
    st.markdown("### Train Loss vs Validation Loss")

    fig_val_loss = px.line(history_df, x="Epoch", y=["loss","val_loss"],markers=True)
    fig_val_loss.add_vline(x=idx_min_vl+1, line_width=2, line_dash='dash',annotation_text="best val_loss", 
              annotation_position="bottom right") #add line of lowest val_loss

    st.plotly_chart(fig_val_loss, use_container_width=True)

    st.markdown("### Train Acuracy vs Validation Accuracy")

    fig_val_acc = px.line(history_df, x="Epoch", y=["accuracy","val_accuracy"],markers=True)
    fig_val_acc.add_vline(x=idx_min_vl+1, line_width=2, line_dash='dash',annotation_text="best val_loss", 
              annotation_position="bottom right") #add line of lowest val_loss

    st.plotly_chart(fig_val_acc, use_container_width=True)
    