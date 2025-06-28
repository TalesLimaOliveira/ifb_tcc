"""
LIBRAS to Portuguese Neural Translator - Clean Code Version

This module provides a complete Streamlit interface for translating
Brazilian Sign Language (LIBRAS) to Portuguese using a neural network.

Main Features:
- Real-time webcam translation
- Image upload processing
- Video upload processing  
- MediaPipe hand landmark detection
- CNN + RNN + GPT neural pipeline
- User feedback system for model improvement

Usage:
    streamlit run streamlit_app.py

Dependencies:
    - streamlit
    - opencv-python
    - mediapipe
    - torch
    - transformers
    - Pillow
    - numpy

Author: Neural LIBRAS Translation Team
Date: June 2025
"""

__version__ = "1.0.0"
__author__ = "Neural LIBRAS Translation Team"

# Application constants
APP_NAME = "LIBRAS para Português - Tradutor Neural"
APP_DESCRIPTION = "Sistema de tradução neural de sinais em LIBRAS para português brasileiro"

# Supported file types
SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png']
SUPPORTED_VIDEO_TYPES = ['mp4', 'avi', 'mov']

# Input types
INPUT_TYPES = {
    "camera": "Câmera/Webcam",
    "image": "Upload de Imagem", 
    "video": "Upload de Vídeo"
}
