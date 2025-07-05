# 🧠 Classificador de Sentimentos com Machine Learning

Um aplicativo de desktop para análise de sentimento, capaz de detectar se um texto é **positivo** ou **negativo**, construído com **Python**, a biblioteca **scikit-learn** para Machine Learning e **Tkinter** para a interface gráfica 💻✨

Este projeto é uma excelente oportunidade para aprofundar conhecimentos em:

* **Machine Learning (ML):** Treinamento e avaliação de modelos de classificação.
* **Processamento de Linguagem Natural (PLN):** Pré-processamento de texto, vetorização (TF-IDF) e remoção de Stop Words.
* **Engenharia de Dados:** Carregamento, limpeza e balanceamento de datasets.
* **Interface Gráfica (GUI):** Criação de uma aplicação interativa com Tkinter.
* **Boas Práticas de Código:** Estrutura modular e tratamento de erros.

-----

## ✨ Funcionalidades

* **Classificação de Sentimentos:** Detecta a polaridade emocional de um texto como `positivo` ou `negativo`.
* **Treinamento com Dataset Real:** O modelo é treinado usando um dataset de resenhas em português (Olist), garantindo maior precisão.
* **Pré-processamento de Texto:**
    * Converte texto para minúsculas.
    * Remove pontuação e caracteres especiais.
    * **Remoção de Stop Words:** Elimina palavras comuns (ex: "de", "a", "o") que não contribuem para o sentimento, usando a biblioteca NLTK.
* **Balanceamento de Classes:** Implementa downsampling para equilibrar a proporção de amostras positivas e negativas no dataset de treinamento, melhorando a performance do modelo em prever sentimentos negativos.
* **Interface Gráfica Intuitiva:**
    * Campo de texto para o usuário digitar a frase a ser classificada.
    * Exibe o sentimento previsto (`POSITIVO` ou `NEGATIVO`).
    * Mostra as probabilidades de o texto ser positivo ou negativo.
    * Design limpo e moderno com Tkinter.
* **Avaliação do Modelo:** Imprime métricas de desempenho (Acurácia, Precisão, Recall, F1-Score) no console para análise.

-----

## 🛠️ Tecnologias Utilizadas

* **Python**: A linguagem de programação principal.
* **Pandas**: Biblioteca para manipulação e análise de dados (DataFrames).
* **scikit-learn**: Biblioteca essencial de Machine Learning, utilizada para:
    * `TfidfVectorizer`: Conversão de texto em vetores numéricos (extração de características).
    * `MultinomialNB`: Algoritmo de classificação (Naive Bayes).
    * `train_test_split`: Divisão de dados em conjuntos de treino e teste.
    * `accuracy_score`, `classification_report`: Avaliação do modelo.
* **NLTK (Natural Language Toolkit)**: Biblioteca para Processamento de Linguagem Natural, usada para:
    * `stopwords`: Acesso a listas de stop words em português.
* **Tkinter**: Biblioteca padrão do Python para a criação da interface gráfica de usuário (GUI).
* **Módulo `re`**: Módulo padrão do Python para expressões regulares (limpeza de texto).

-----

## 🚀 Como executar localmente

Para rodar este projeto em sua máquina, siga os passos abaixo:

1.  **Clone este repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
    ```
    (Lembre-se de substituir `SEU-USUARIO/SEU-REPOSITORIO` pelo caminho real do seu repositório no GitHub).

2.  **Acesse a pasta do projeto:**
    ```bash
    cd nome-do-repositorio # ou meu_classificador_sentimentos
    ```

3.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # No Windows (Command Prompt):
    venv\Scripts\activate.bat
    # No macOS/Linux:
    source venv/bin/activate
    ```

4.  **Instale as dependências:**
    ```bash
    pip install pandas scikit-learn nltk
    ```

5.  **Baixe os dados de Stop Words do NLTK:**
    * Abra o interpretador Python no seu terminal (com o ambiente virtual ativo):
        ```bash
        python
        ```
    * Dentro do interpretador, execute:
        ```python
        import nltk
        nltk.download('stopwords')
        exit() # Para sair do interpretador
        ```

6.  **Obtenha o Dataset:**
    * Baixe o arquivo `olist.csv` (ou o dataset de sua preferência com colunas de texto e sentimento) e coloque-o na **mesma pasta** do script `sentiment_classifier_app.py`.
    * **Verifique e ajuste** as variáveis `DATASET_FILE`, `TEXT_COLUMN`, `SENTIMENT_COLUMN` no arquivo `sentiment_classifier_app.py` para corresponderem ao seu dataset.

7.  **Execute o aplicativo:**
    ```bash
    python sentiment_classifier_app.py
    ```

O aplicativo será iniciado, o modelo será treinado (o que pode levar alguns segundos dependendo do tamanho do dataset) e a janela da GUI aparecerá, pronta para classificar textos!
