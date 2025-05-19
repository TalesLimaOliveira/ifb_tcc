import os
import argparse
import cv2
import mediapipe as mp
import pandas as pd

def extract_hand_landmarks_from_video(video_path):
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    landmarks_list = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        row = {'frame': frame_index}
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    row[f'hand{hand_idx}_x_{lm_idx}'] = lm.x
                    row[f'hand{hand_idx}_y_{lm_idx}'] = lm.y
                    row[f'hand{hand_idx}_z_{lm_idx}'] = lm.z
        landmarks_list.append(row)
        frame_index += 1

    cap.release()
    hands.close()
    return pd.DataFrame(landmarks_list)

def process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(input_dir, filename)
            print(f'Processando {video_path}...')
            df_landmarks = extract_hand_landmarks_from_video(video_path)
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.parquet')
            df_landmarks.to_parquet(output_path)
            print(f'Salvo: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extrai landmarks das mãos de vídeos .mp4 usando MediaPipe.')
    parser.add_argument('--input_dir', required=True, help='Diretório com vídeos .mp4')
    parser.add_argument('--output_dir', required=True, help='Diretório para salvar arquivos .parquet')
    args = parser.parse_args()
    process_videos(args.input_dir, args.output_dir)