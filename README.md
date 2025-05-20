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

Reconhecimento e tradução de sinais em LIBRAS (Língua Brasileira de Sinais) usando visão computacional e redes neurais profundas.

## Objetivo

Desenvolver um sistema capaz de reconhecer e traduzir gestos e frases completas em LIBRAS a partir de vídeos, utilizando MediaPipe para extração de landmarks e TensorFlow para classificação, com tradução em tempo real via webcam.

## Estrutura do Projeto

- `data/raw_videos/`: vídeos originais em .mp4
- `data/landmarks/`: arquivos .parquet com landmarks extraídos
- `data/train.csv`: metadados para treinamento
- `model/src/`: scripts Python para extração, preparação e modelagem
- `notebooks/`: notebooks para experimentação e treinamento
- `saved_models/`: modelos treinados e encoders salvos
- `app/`: aplicação Streamlit para tradução em tempo real

## Como treinar o modelo

1. **Crie um ambiente virtual:**
   ```
   py -3.10 -m venv venv
   venv\Scripts\activate
   ```

2. **Instale as dependências:**
   ```
   pip install -r requirements.txt
   ```

3. **Extraia os landmarks dos vídeos:**
   ```
   python model/src/extract_landmarks.py --input_dir data/raw_videos --output_dir data/landmarks
   ```

4. **Prepare o arquivo `train.csv` com os metadados.**

5. **Treine o modelo:**
   - Abra e execute o notebook `notebooks/train_model.ipynb`.
   - O modelo treinado (`model_libras.h5`) e o encoder (`label_encoder.pkl`) serão salvos em `saved_models/`.

## Como usar a tradução em tempo real

1. **Execute o app Streamlit:**
   ```
   streamlit run app/app.py
   ```

2. **Clique em "Iniciar Webcam" para começar a tradução dos sinais capturados pela câmera.**

O modelo treinado será carregado automaticamente de `saved_models/`.

---

## Observações

- Certifique-se de que os arquivos `model_libras.h5` e `label_encoder.pkl` estejam presentes em `saved_models/` antes de rodar o app.
- O pré-processamento dos dados deve ser idêntico no treinamento e na aplicação em tempo real para garantir bons resultados.