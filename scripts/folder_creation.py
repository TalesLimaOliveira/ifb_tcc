import os
import csv

# Caminho do arquivo CSV
csv_path = '../labels_vlisbrasil_sings.csv'

# Pasta onde as novas pastas serão criadas
output_dir = '../dataset/raw_data/vlibrasil'
os.makedirs(output_dir, exist_ok=True)

def format_phrase(phrase):
    # Substitui espaços por underscores e remove quebras de linha
    return phrase.strip().replace(" ", "_")

# Lê o CSV e cria as pastas
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sign_id = row['sign_id']
        phrase = format_phrase(row['phrase'])
        folder_name = f"{sign_id}_{phrase}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Criada: {folder_path}")