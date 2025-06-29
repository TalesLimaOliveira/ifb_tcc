import os
import re
import sys
import signal
import requests
import pandas as pd
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configurações principais
BASE_URL = "https://libras.cin.ufpe.br/sign/"
PARTICIPANT = "vlisbrasil"
INPUT_CSV = "../labels_vlisbrasil_sings.csv"
OUTPUT_CSV = "../saida_videos.csv"
VIDEOS_DIR = "../dataset/raw_data/vlibrasil"
MAX_IDS_POR_EXECUCAO = 1364  # quatidade de IDs a serem processados por execução

# Variável para controle de interrupção
interromper = False

def signal_handler(sig, frame):
    global interromper
    print("\nInterrupção solicitada. Finalizando o processo com segurança...")
    interromper = True


signal.signal(signal.SIGINT, signal_handler)


def formatar_nome_pasta(sign_id: str, phrase: str) -> str:
    """Formata o nome da pasta com ID e frase."""
    frase_limpa = re.sub(r'[^\w\s]', '', phrase).strip().replace(" ", "_")
    return f"{sign_id}_{frase_limpa}"


def carregar_ids_baixados(path: str) -> set:
    """Carrega os IDs já presentes no CSV de saída."""
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"sign_id": str})
        return set(df["sign_id"])
    return set()


def obter_urls_videos(html: str) -> List[str]:
    """Extrai URLs de vídeos .mp4 da página HTML."""
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all("video")
    return [source["src"] for video in tags if (source := video.find("source")) and source["src"].endswith(".mp4")]


def baixar_video(url: str, destino: str) -> None:
    """Realiza o download do vídeo para o caminho de destino."""
    resposta = requests.get(url, stream=True)
    resposta.raise_for_status()
    with open(destino, "wb") as f:
        for bloco in resposta.iter_content(chunk_size=8192):
            f.write(bloco)


def baixar_videos_sign(sign_id: str, phrase: str) -> List[Dict]:
    """Baixa os vídeos associados a um ID e retorna os metadados para o CSV."""
    url = f"{BASE_URL}{sign_id}"
    try:
        resposta = requests.get(url, timeout=10)
        resposta.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERRO] Falha ao acessar {url}: {e}")
        return []

    videos = obter_urls_videos(resposta.text)
    if not videos:
        print(f"[AVISO] Nenhum vídeo encontrado para o ID {sign_id}")
        return []

    pasta = formatar_nome_pasta(sign_id, phrase)
    caminho_pasta = os.path.join(VIDEOS_DIR, pasta)
    os.makedirs(caminho_pasta, exist_ok=True)

    metadados = []
    for video_url in videos:
        nome_arquivo = os.path.basename(video_url)
        destino = os.path.join(caminho_pasta, nome_arquivo)

        try:
            baixar_video(video_url, destino)
            metadados.append({
                "video_path": os.path.relpath(destino),
                "participant": PARTICIPANT,
                "sign_id": sign_id,
                "phrase": phrase
            })
        except Exception as e:
            print(f"[ERRO] Falha ao baixar {video_url}: {e}")
    return metadados


def salvar_progresso(saida_csv: str, dados: List[Dict]) -> None:
    """Salva o progresso em disco no arquivo de saída."""
    if not dados:
        return
    df_novo = pd.DataFrame(dados)
    if os.path.exists(saida_csv):
        df_existente = pd.read_csv(saida_csv)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo
    df_final.to_csv(saida_csv, index=False)


def main():
    print("Iniciando download de vídeos VLISBRASIL...")
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    df_entrada = pd.read_csv(INPUT_CSV, dtype={"sign_id": str})
    ids_ja_baixados = carregar_ids_baixados(OUTPUT_CSV)

    df_pendentes = df_entrada[~df_entrada["sign_id"].isin(ids_ja_baixados)]
    if df_pendentes.empty:
        print("Todos os vídeos já foram baixados.")
        return

    df_lote = df_pendentes.head(MAX_IDS_POR_EXECUCAO)
    progresso: List[Dict] = []

    for _, row in tqdm(df_lote.iterrows(), total=len(df_lote), desc="Processando IDs"):
        if interromper:
            break
        sign_id = row["sign_id"]
        phrase = row["phrase"]
        resultados = baixar_videos_sign(sign_id, phrase)
        progresso.extend(resultados)

    salvar_progresso(OUTPUT_CSV, progresso)
    print("Processo finalizado. Progresso salvo com sucesso.")


if __name__ == "__main__":
    main()
