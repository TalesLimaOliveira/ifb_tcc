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
Este projeto tem como objetivo desenvolver uma rede neural profunda que utiliza técnicas de visão computacional para o reconhecimento e tradução de gestos e frases completas em LIBRAS (Língua Brasileira de Sinais). O sistema é capaz de compreender e traduzir frases de forma contínua, levando em consideração o contexto dos sinais realizados.

# LIBRAS Sign Recognition

Reconhecimento e tradução de sinais em LIBRAS (Língua Brasileira de Sinais) usando visão computacional e redes neurais profundas.

## Objetivo

Desenvolver um sistema capaz de reconhecer e traduzir gestos e frases completas em LIBRAS a partir de vídeos, utilizando MediaPipe para extração de landmarks e PyTorch para classificação, com tradução em tempo real via webcam.

## Estrutura do Projeto

- `dataset/raw_data/`: vídeos ou imagens originais
- `dataset/processed/landmarks/`: arquivos `.npy` com landmarks extraídos (shape: seq_len, 21, 3)
- `dataset/processed/supplemental_metadata.csv`: metadados para treinamento
- `app/create_dataset.py`: script para extração dos landmarks a partir de webcam
- `notebooks/libras2pt_cnn_rnn_gpt.ipynb`: notebook principal de treinamento e avaliação
- `models/`: modelos treinados

## Como processar os dados brutos e criar landmarks

1. **Certifique-se de que as dependências estão instaladas:**
   ```
   pip install -r requirements.txt
   ```

2. **Execute o script de coleta de landmarks:**
   ```
   cd app
   python create_dataset.py
   ```
   - Siga as instruções do terminal para gravar gestos/frases via webcam.
   - Cada gravação gera um arquivo `.npy` em `dataset/processed/landmarks/` e atualiza o arquivo `supplemental_metadata.csv`.
   - O arquivo `.npy` tem shape `(seq_len, 21, 3)` (sequência de frames, 21 pontos, 3 coordenadas).

## Como treinar e avaliar o modelo no notebook

1. **Abra o notebook principal:**
   - `notebooks/libras2pt_cnn_rnn_gpt.ipynb`

2. **Execute as células na ordem:**
   - O notebook irá:
     - Carregar os dados e metadados
     - Tokenizar as frases usando o tokenizador BERTimbau (WordPiece para português)
     - Montar o DataLoader com padding dinâmico
     - Definir e inicializar a arquitetura CNN + LSTM
     - Treinar o modelo e salvar o melhor checkpoint
     - Avaliar a acurácia token a token

3. **Salvamento:**
   - O modelo treinado será salvo em `models/`.

## Como usar a tradução em tempo real

1. **Execute o app Streamlit (opcional):**
   ```
   streamlit run app/streamlit_app.py
   ```
   - O modelo treinado será carregado automaticamente de `models/`.

---

## Observações

- Certifique-se de que os arquivos `.npy` e `supplemental_metadata.csv` estejam presentes em `dataset/processed/` antes de rodar o notebook.
- O pré-processamento dos dados deve ser idêntico no treinamento e na aplicação em tempo real para garantir bons resultados.
- O notebook utiliza tokenização robusta para português (BERTimbau) e arquitetura moderna (CNN + LSTM) para tradução de LIBRAS para português natural.