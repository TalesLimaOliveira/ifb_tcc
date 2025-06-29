import requests
from bs4 import BeautifulSoup
import csv
import time

# Intervalo de IDs
start_id = 2001
end_id = 5000

# URL base
base_url = "https://libras.cin.ufpe.br/sign/"

# Nome do arquivo CSV
output_file = "labels_metadata.csv"

# Abrindo arquivo CSV para escrita
with open(output_file, mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["sign_id", "phrase"])  # Cabeçalho

    for sign_id in range(start_id, end_id + 1):
        url = f"{base_url}{sign_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                h2 = soup.find("h2", class_="page-section-heading text-center text-uppercase text-secondary mb-0")
                if h2:
                    phrase = h2.get_text(strip=True)
                    writer.writerow([sign_id, phrase])
                    print(f"[{sign_id}] - '{phrase}'")
                else:
                    print(f"[{sign_id}] - h2 não encontrado")
            else:
                print(f"[{sign_id}] - Página não encontrada (status {response.status_code})")
        except Exception as e:
            print(f"[{sign_id}] - Erro: {e}")

        time.sleep(0.2)  # Intervalo entre requisições para evitar sobrecarga do servidor
