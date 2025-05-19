import streamlit as st
import cv2
from app.camera import Camera
from app.gesture_processing import GestureProcessor
from app.neural_network import NeuralNetwork
from app.utils import load_label_encoder
from model.src.preprocessing import preprocess_landmarks

MODEL_PATH = "../saved_models/model_libras.h5"
ENCODER_PATH = "../saved_models/label_encoder.pkl"

@st.cache_resource
def load_model_and_encoder():
    model = NeuralNetwork.load(MODEL_PATH)
    encoder = load_label_encoder(ENCODER_PATH)
    return model, encoder

def main():
    st.set_page_config(page_title="Reconhecimento de Gestos em Libras", layout="centered")
    st.title("Tradutor LIBRAS em Tempo Real")

    neural_network, label_encoder = load_model_and_encoder()
    gesture_processor = GestureProcessor()

    if 'run' not in st.session_state:
        st.session_state['run'] = False

    start = st.button("Iniciar Webcam")
    stop = st.button("Parar Webcam")

    if start:
        st.session_state['run'] = True
    if stop:
        st.session_state['run'] = False

    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    if st.session_state['run']:
        camera = Camera()
        while st.session_state['run']:
            frame = camera.get_frame()
            if frame is None:
                st.warning("Não foi possível acessar a webcam.")
                break
            frame_landmarks, landmarks_df = gesture_processor.process_frame(frame)
            gesture_text = ""
            if landmarks_df is not None:
                features = preprocess_landmarks(landmarks_df)
                pred = neural_network.predict([features])[0]
                gesture_text = label_encoder.inverse_transform([pred])[0]
                result_placeholder.markdown(f"**Reconhecido:** {gesture_text}")
            frame_placeholder.image(cv2.cvtColor(frame_landmarks, cv2.COLOR_BGR2RGB), channels="RGB")
    else:
        frame_placeholder.empty()
        result_placeholder.empty()

if __name__ == "__main__":
    main()