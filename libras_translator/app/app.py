import streamlit as st
import cv2
from camera import Camera
from gesture_processing import GestureProcessor
from neural_network import NeuralNetwork
from utils import load_label_encoder

st.set_page_config(page_title="Reconhecimento de Gestos em Libras", layout="centered")
st.title("Reconhecimento de Gestos em Libras com MediaPipe e Rede Neural")

MODEL_PATH = "../model_libras.h5"
ENCODER_PATH = "../label_encoder.pkl"

@st.cache_resource
def load_model_and_encoder():
    nn = NeuralNetwork(MODEL_PATH)
    encoder = load_label_encoder(ENCODER_PATH)
    return nn, encoder

neural_network, label_encoder = load_model_and_encoder()
gesture_processor = GestureProcessor()

run = st.button("Iniciar Webcam")
frame_window = st.empty()
output_text = st.empty()

if run:
    camera = Camera()
    while True:
        frame = camera.get_frame()
        if frame is None:
            st.warning("Não foi possível acessar a webcam.")
            break
        frame_landmarks, landmarks = gesture_processor.process_frame(frame)
        gesture_text = ""
        if landmarks is not None:
            pred = neural_network.predict(landmarks)
            gesture_text = label_encoder.inverse_transform([pred])[0]
            output_text.markdown(f"**Reconhecido:** {gesture_text}")
        frame_window.image(cv2.cvtColor(frame_landmarks, cv2.COLOR_BGR2RGB), channels="RGB")
        if st.button("Parar"):
            break
    camera.release()

if __name__ == "__main__":
    main()