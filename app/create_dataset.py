import os
import cv2
import pandas as pd
import mediapipe as mp
from datetime import datetime
import uuid

# Configs
SAVE_DIR = "dataset/train"
LANDMARK_DIR = os.path.join(SAVE_DIR, "landmarks")
METADATA_FILE = os.path.join(SAVE_DIR, "supplemental_metadata.csv")
os.makedirs(LANDMARK_DIR, exist_ok=True)

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Inicializa metadata.csv
if not os.path.exists(METADATA_FILE):
    df_meta = pd.DataFrame(columns=["path", "file_id", "phrase"])
else:
    df_meta = pd.read_csv(METADATA_FILE)

def extract_landmark_frame(hand_landmarks):
    # Extrai os 21 pontos da mão: (x, y, z) para cada ponto
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def record_sequence(label, seconds=3, fps=15):
    cap = cv2.VideoCapture(0)
    file_id = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    path = f"landmarks/{file_id}.parquet"
    full_path = os.path.join(LANDMARK_DIR, f"{file_id}.parquet")

    print(f"[INFO] Gravando gesto: {label}")
    print("[INFO] Pressione 'q' para parar manualmente.")

    sequence = []
    total_frames = int(seconds * fps)
    count = 0

    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = extract_landmark_frame(results.multi_hand_landmarks[0])
            sequence.append(landmarks)
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Converte e salva como Parquet
    df = pd.DataFrame(sequence)
    df.to_parquet(full_path, index=False)

    # Atualiza metadata
    global df_meta
    df_meta = df_meta.append({
        "path": path,
        "file_id": file_id,
        "phrase": label
    }, ignore_index=True)

    df_meta.to_csv(METADATA_FILE, index=False)
    print(f"[✔] Salvo: {full_path}")
    print(f"[✔] Metadata atualizada.")

if __name__ == "__main__":
    while True:
        label = input("Digite a frase ou palavra (ex: bom dia, tudo bem): ").strip()
        if not label:
            break
        record_sequence(label)
