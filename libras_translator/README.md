# 🌟 Trabalho de Conclusão de Curso - Instituto Federal de Brasília 🌟

## RECONHECIMENTO E TRADUÇÃO DE FRASES EM LIBRAS UTILIZANDO REDES NEURAIS

<p align="center">
    <img src="https://img.shields.io/badge/Language-Python-blue?logo=python" alt="Language">
    <img src="https://img.shields.io/badge/Language-Julia-purple?logo=julia" alt="Language">
    <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>


## 👨‍🏫 Orientador
- **Raimundo Vasconcelos**

## 👨‍🎓 Aluno
- **[Tales Oliveira](https://github.com/TalesLimaOliveira)**

---

## 📚 Descrição
Este projeto tem como objetivo desenvolver uma rede neural profunda que utiliza técnicas de visão computacional para o reconhecimento e tradução de gestos e frases completas em LIBRAS (Língua Brasileira de Sinais). Diferentemente de soluções existentes que se concentram na tradução de sinais ou letras isoladas, esta proposta busca criar um sistema que compreenda e traduza frases de forma contínua, e em tempo real, levando em consideração o contexto dos sinais previamente realizados.

# LIBRAS Sign Recognition

Reconhecimento de sinais em LIBRAS a partir de vídeos usando MediaPipe e TensorFlow.

## Estrutura

- `data/raw_videos/`: vídeos originais em .mp4
- `data/landmarks/`: arquivos .parquet com landmarks extraídos
- `data/train.csv`: metadados para treinamento
- `src/`: scripts Python para extração, preparação e modelagem
- `notebooks/`: notebooks para experimentação e treinamento

## Como usar
0. Crie um ambiente virtual:

   ```
   py -3.10 -m venv venv
   venv\Scripts\activate
   ```

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Extraia os landmarks dos vídeos:
   ```
   python src/extract_landmarks.py --input_dir data/raw_videos --output_dir data/landmarks
   ```

3. Prepare o arquivo `train.csv` com os metadados.

4. Treine o modelo usando o notebook em `notebooks/train_model.ipynb`.