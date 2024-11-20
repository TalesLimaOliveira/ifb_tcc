import cv2  # Biblioteca para captura e manipulação de vídeo
import mediapipe as mp  # Biblioteca para detecção de mãos e outros recursos de visão computacional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Caminho para o modelo de reconhecimento de gestos
model_path = './gesture_recognizer.task'  # Caminho relativo ao diretório atual

# Definição das opções e classes do MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Função de callback para imprimir os resultados
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

# Função para desenhar os landmarks na imagem
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10  # MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv2.LINE_AA)  # FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS

    return annotated_image

# Configura as opções do HandLandmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# Captura de vídeo da câmera padrão (geralmente a webcam)
cap = cv2.VideoCapture(0)

# Inicializa o HandLandmarker com as opções configuradas
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():  # Loop enquanto a captura de vídeo está aberta
        success, image = cap.read()  # Lê um frame da câmera
        if not success:  # Se a leitura falhar
            print("Ignoring empty camera frame.")  # Ignora o frame vazio
            continue  # Continua para o próximo frame

        # Converte a imagem de BGR (padrão do OpenCV) para RGB (padrão do MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Cria um objeto Image do MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Processa a imagem para detectar mãos
        landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Desenha os landmarks na imagem
        annotated_image = draw_landmarks_on_image(image_rgb, landmarker)

        # Exibe a imagem com as detecções (se houver)
        cv2.imshow('Hand Detection', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(5) & 0xFF == 27:  # Pressione 'ESC' para sair
            break

cap.release()
cv2.destroyAllWindows()