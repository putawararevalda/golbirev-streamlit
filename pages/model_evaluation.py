import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

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

    performance_df = pd.DataFrame(columns=["model_name","OK precision","accuracy","False OK","True NOT_OK","False NOT_OK","True OK"])

    for mhd_iter, mhd in enumerate(model_hist_dict.keys()):
        conf_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[mhd],
        #index_col=0,
        sheet_name = "Confusion Matrix",skiprows=2,usecols="C:D")

        class_report_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[mhd],
        #index_col=0,
        sheet_name = "Classification Report")

        acc_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[mhd],
        #index_col=0,
        sheet_name = "Accuracy for Test Data")
        

        a_no_p_no = conf_df.iloc[0,0]
        a_ok_p_no = conf_df.iloc[1,0]
        a_no_p_ok = conf_df.iloc[0,1]
        a_ok_p_ok = conf_df.iloc[1,1]

        prec_1 = class_report_df.at[0,"1"]
        acc_score = acc_df.at[0,"f1-score"]
    			

        listdummy = []

        listdummy.append(mhd)
        listdummy.append(prec_1)
        listdummy.append(acc_score)
        listdummy.append(a_ok_p_no)
        listdummy.append(a_no_p_no)
        listdummy.append(a_no_p_ok)
        listdummy.append(a_ok_p_ok)

        s = pd.Series(listdummy, index=performance_df.columns)
        performance_df = performance_df.append(s, ignore_index=True)

    

    st.title('MODEL Evaluation')

    st.markdown("How our model is evaluated")

    st.dataframe(performance_df)

    st.info("We choose Base model 2 CONV (50 epoch) ADAM (highest OK precision)")

    st.markdown("""---""")

    option = st.selectbox(
     'Which model do you want to view?',
     (model_hist_dict.keys()))

    st.write('You picked:', option)

    conf_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[option],
    #index_col=0,
    sheet_name = "Confusion Matrix",skiprows=2,usecols="C:D")

    a_no_p_no = conf_df.iloc[0,0]
    a_ok_p_no = conf_df.iloc[1,0]
    a_no_p_ok = conf_df.iloc[0,1]
    a_ok_p_ok = conf_df.iloc[1,1]

    cf_matrix = np.array([[a_no_p_no,a_no_p_ok]
    ,[a_ok_p_no,a_ok_p_ok]])

    st.markdown("""---""")
    st.markdown("### Confusion Matrix")
    st.markdown("Y Axis : Actual, X Axis : Model prediction")
    
    group_names = ["True NOT_OK","False OK","False NOT_OK","True OK"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    

    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    st.pyplot(fig)

    class_report_df = pd.read_excel(r"data/prod_history/"+model_hist_dict[option],
    #index_col=0,
    sheet_name = "Classification Report")
    st.dataframe(class_report_df)


    