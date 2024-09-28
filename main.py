import streamlit as st
import importlib

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #0c0c0c;
        color: #e0e0e0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ff4757;
        text-align: center;
    }
    .sidebar .button-card {
        background-color: #1abc9c; /* Change to your desired color */
        border-radius: 15px;
        padding: 20px;  /* Increased padding for larger buttons */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin: 10px; /* Adjusted margin for spacing */
        transition: transform 0.2s, background-color 0.3s;
        cursor: pointer;
        font-size: 22px; /* Increased font size */
        height: 100px; /* Set a fixed height for buttons */
        width: 100%; /* Make buttons take full width of column */
    }
    .sidebar .button-card:hover {
        transform: scale(1.05);
        background-color: #f39c12; /* Change to your desired hover color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_page(page_name):
    module = importlib.import_module(page_name)
    module.run()

def sidebar():
    st.sidebar.title("Model Deployment")
    st.sidebar.write("Choose the type of task:")
    
    button_options = [
        ("Tabular Regression", "tabular_regression"),
        ("Tabular Classification", "tabular_classification"),
        ("Image Classification", "image_classification"),
        ("Image Segmentation", "image_segmentation"),
        ("Text Classification", "text_classification"),
    ]
    
    for button_text, page_key in button_options:
        if st.sidebar.button(button_text, key=page_key):
            st.session_state.page = page_key

if 'page' not in st.session_state:
    st.session_state.page = 'homepage'

sidebar()

if st.session_state.page == 'homepage':
    st.title("Welcome to Model Deployment")
    st.write("Select a task from the sidebar.")
else:
    load_page(st.session_state.page)
