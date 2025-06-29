import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configurações iniciais
base_url = "https://libras.cin.ufpe.br/sign/"
participant = "vlisbrasil"
input_csv = "vlisbrasil_sing.csv"
output_csv = "saida_videos.csv"
output_rows = []

# Criar pasta de destino se não existir
os.makedirs("videos", exist_ok=True)

# Carregar o CSV de entrada
df = pd.read_csv(input_csv)

# Função para limpar o nome da pasta
def format_folder_name(sign_id, phrase):
    phrase_clean = re.sub(r'[^\w\s]', '', phrase).strip().replace(" ", "_")
    return f"{sign_id}_{phrase_clean}"

# Função para baixar vídeos da página
def baixar_videos(sign_id, phrase):
    url = f"{base_url}{sign_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Erro ao acessar {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    video_tags = soup.find_all("video")

    folder_name = format_folder_name(sign_id, phrase)
    folder_path = os.path.join("videos", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Encontrar URLs de vídeos
    for idx, video_tag in enumerate(video_tags):
        source = video_tag.find("source")
        if source and source.get("src", "").endswith(".mp4"):
            video_url = source["src"]
            video_name = os.path.basename(video_url)
            video_path = os.path.join(folder_path, video_name)

            # Baixar vídeo
            try:
                video_resp = requests.get(video_url, stream=True)
                with open(video_path, "wb") as f:
                    for chunk in video_resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Adicionar ao CSV de saída
                output_rows.append({
                    "video_path": os.path.relpath(video_path),
                    "participant": participant,
                    "sign_id": sign_id,
                    "phrase": phrase
                })
            except Exception as e:
                print(f"Erro ao baixar vídeo {video_url}: {e}")

# Loop principal com barra de progresso
for _, row in tqdm(df.iterrows(), total=len(df), desc="Baixando vídeos"):
    sign_id = row["sign_id"]
    phrase = row["phrase"]
    baixar_videos(sign_id, phrase)

# Salvar CSV de saída
saida_df = pd.DataFrame(output_rows)
saida_df.to_csv(output_csv, index=False)
print(f"\nDownload concluído. Arquivo de saída: {output_csv}")