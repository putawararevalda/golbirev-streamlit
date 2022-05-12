import streamlit as st
from PIL import Image

# Custom imports 
from multipage import MultiPage
from pages import home, about, solution, model, model_history, dataset, model_evaluation

# Create an instance of the app 
app_to_run = MultiPage()

# Title of the main page
#st.title("Data Storyteller Application")

# Add all your applications (pages) here
app_to_run.add_page("Home", home.app)
app_to_run.add_page("About", about.app)
app_to_run.add_page("Dataset", dataset.app)
app_to_run.add_page("Our Solution", solution.app)
app_to_run.add_page("Model Architecture", model.app)
app_to_run.add_page("Model and Training", model_history.app)
app_to_run.add_page("Model Evaluation", model_evaluation.app)

st.set_page_config(page_title='Golbirev API', page_icon = "🏡", initial_sidebar_state = 'auto')

# The main app
app_to_run.run()