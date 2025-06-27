import os
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Parâmetros
BASE_URL = "https://libras.cin.ufpe.br/sign/"
VIDEO_FOLDER = "../dataset/raw_data/vlibrasil/"
OUTPUT_CSV = "../labels_vlisbrasil_final.csv"
INPUT_CSV = "../labels_vlisbrasil_sings.csv"

# Cria a pasta se não existir
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Lê os sinais
df_signs = pd.read_csv(INPUT_CSV)

# Lista de registros para o CSV final
final_data = []

# Scraping e download
for _, row in df_signs.iterrows():
    sign_id = str(row["sign_id"]).strip()
    phrase = str(row["phrase"]).strip()
    url = urljoin(BASE_URL, sign_id)

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[ERRO] Não foi possível acessar {url}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        participants = soup.find_all("h2", class_="page-section-heading text-success mb-0")
        videos = soup.find_all("video")

        for i, (participant_tag, video_tag) in enumerate(zip(participants, videos)):
            participant = participant_tag.text.strip()
            source_tag = video_tag.find("source")
            if not source_tag:
                continue
            video_url = source_tag["src"]
            video_filename = video_url.split("/")[-1]
            local_filename = os.path.join(VIDEO_FOLDER, video_filename)

            # Baixar o vídeo se não existir
            if not os.path.exists(local_filename):
                print(f"[↓] Baixando: {video_url}")
                video_resp = requests.get(video_url, stream=True)
                with open(local_filename, "wb") as f:
                    for chunk in video_resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Salvar entrada
            final_data.append({
                "video_path": local_filename,
                "file_name": video_filename,
                "participant": participant,
                "sing_id": sign_id,
                "phrase": phrase
            })

    except Exception as e:
        print(f"[ERRO] Ao processar {url}: {e}")

# Salvar o CSV final
df_output = pd.DataFrame(final_data)
df_output.to_csv(OUTPUT_CSV, index=False)
print(f"\nCSV final salvo como: {OUTPUT_CSV}")
