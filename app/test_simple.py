

import streamlit as st
import torch
import os
import sys

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="LIBRAS Test",
    page_icon="🖖",
    layout="wide"
)

def main():
    st.title("LIBRAS Application - Basic Test")
    st.write("This is a basic test to verify Streamlit is working properly.")
    
    st.subheader("System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test basic Streamlit functionality
    st.subheader("Basic Functionality Test")
    
    if st.button("Test Button"):
        st.success("Button clicked successfully!")
    
    name = st.text_input("Enter your name:")
    if name:
        st.write(f"Hello, {name}!")
    
    # Test importing custom modules
    st.subheader("Module Import Test")
    
    try:
        from config import MODEL_CONFIG
        st.success("✅ config.py imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import config.py: {e}")
    
    try:
        from models import ModelManager
        st.success("✅ models.py imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import models.py: {e}")
    
    try:
        from video_processing import MediaPipeProcessor
        st.success("✅ video_processing.py imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import video_processing.py: {e}")
    
    try:
        from interface import UIManager
        st.success("✅ interface.py imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import interface.py: {e}")

if __name__ == "__main__":
    main()
