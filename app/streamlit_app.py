"""
LIBRAS to Portuguese Neural Translator - Main Application

Sistema otimizado de tradução de LIBRAS para português usando redes neurais.

Usage:
    streamlit run streamlit_app.py

Author: Tales Lima Oliveira
Date: Julho 2025
Version: 2.0 (Otimizada)
"""

import streamlit as st
import torch
import os
import sys
import traceback
import warnings
from models import ModelManager
from video_processing import MediaPipeProcessor, LandmarkVisualizer, VideoProcessor, ImageProcessor
from interface import UIManager
from config import MODEL_CONFIG, MEDIAPIPE_CONFIG, TOKENIZER_CONFIG

# --- Environment and Warning-Related Configurations ---

def setup_environment():
    """Sets up environment variables and suppresses warnings for a cleaner output."""
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

    # Suppress various warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*use_column_width.*')

# Call setup function at the beginning
setup_environment()

        
# --- Page Configuration ---

st.set_page_config(
    page_title="LIBRAS para Português - Tradutor Neural",
    page_icon="🖖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add caching for better performance
@st.cache_resource
def load_model_manager():
    """Cache the model manager to avoid reloading"""
    return ModelManager(MODEL_CONFIG, TOKENIZER_MODEL, DEVICE)

@st.cache_resource  
def load_video_processors():
    """Cache video processors to avoid reinitializing MediaPipe"""
    mp_processor = MediaPipeProcessor(MEDIAPIPE_CONFIG)
    visualizer = LandmarkVisualizer(mp_processor.mp_hands, mp_processor.mp_drawing)
    video_processor = VideoProcessor(mp_processor, visualizer)
    image_processor = ImageProcessor(mp_processor, visualizer)
    return mp_processor, visualizer, video_processor, image_processor

# --- Global Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_MODEL = TOKENIZER_CONFIG["model_name"]

# --- Application Initialization ---

def initialize_application():
    """
    Initializes and returns all application components.
    Uses caching to avoid re-initialization and improve performance.
    """
    if 'initialized' in st.session_state and st.session_state.initialized:
        return st.session_state.model_manager, st.session_state.ui_manager

    try:
        with st.spinner("Inicializando aplicação..."):
            # Use cached model manager
            model_manager = load_model_manager()
            
            # Load tokenizer with progress
            st.text("Carregando tokenizer...")
            if not hasattr(model_manager, 'tokenizer') or model_manager.tokenizer is None:
                tokenizer = model_manager.load_tokenizer()
                if tokenizer is None:
                    st.error("Falha ao carregar o tokenizer.")
                    return None, None

            # Initialize model with progress
            st.text("Inicializando modelo...")
            if model_manager.model is None:
                model = model_manager.initialize_model(model_manager.tokenizer.vocab_size)
                if model is None:
                    st.error("Falha ao inicializar o modelo.")
                    return None, None

                # Load pretrained weights
                st.text("Carregando pesos do modelo...")
                success, loaded_path = model_manager.load_pretrained_weights()
                if success:
                    st.success(f"Modelo carregado: {os.path.basename(loaded_path)}")
                else:
                    st.warning("Nenhum modelo pré-treinado encontrado.")

            # Use cached processors
            st.text("Inicializando processadores de vídeo...")
            mp_processor, visualizer, video_processor, image_processor = load_video_processors()
            ui_manager = UIManager(model_manager, video_processor, image_processor)

            # Store in session state
            st.session_state.model_manager = model_manager
            st.session_state.ui_manager = ui_manager
            st.session_state.initialized = True

            st.success("Aplicação inicializada com sucesso!")
            return model_manager, ui_manager

    except Exception as e:
        st.error(f"Falha na inicialização da aplicação: {e}")
        with st.expander("Detalhes do erro"):
            st.code(traceback.format_exc())
        st.session_state.initialized = False
        return None, None

# --- Main Application Logic ---

def main():
    """
    Main application entry point.
    Initializes the app and runs the user interface.
    """
    st.title("Tradutor Neural de LIBRAS para Português")

    model_manager, ui_manager = initialize_application()

    if not st.session_state.get('initialized', False):
        st.error("A aplicação não pôde ser inicializada. Verifique os logs de erro acima.")
        # Display troubleshooting information
        with st.expander("Informações para Solução de Problemas"):
            st.markdown(f"""
            **Possíveis Causas:**
            1. **Dependências faltando:** Execute `pip install -r requirements.txt`
            2. **Arquivos de modelo ausentes:** Verifique se o modelo (`.pt` ou `.pth`) está na pasta `models/`.
            3. **Problemas com CUDA:** Se encontrar erros de GPU, tente desativar a GPU para teste.
            
            **Ambiente Atual:**
            - Versão do Python: `{sys.version}`
            - Versão do PyTorch: `{torch.__version__}`
            - CUDA disponível: `{torch.cuda.is_available()}`
            - Dispositivo em uso: `{DEVICE}`
            """)
        return

    try:
        ui_manager.run()
    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução: {e}")
        with st.expander("Detalhes do Erro"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()