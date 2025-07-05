import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import os
import sys
import nltk
from nltk.corpus import stopwords 

try:
    STOP_WORDS_PORTUGUESE = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS_PORTUGUESE = set(stopwords.words('portuguese'))
print(f"Número de Stop Words em Português carregadas: {len(STOP_WORDS_PORTUGUESE)}\n")


DATASET_FILE = 'olist.csv'
TEXT_COLUMN = 'review_text'
SENTIMENT_COLUMN = 'rating'

df = None
if os.path.exists(DATASET_FILE):
    try:
        df = pd.read_csv(DATASET_FILE, encoding='utf-8')
        print(f"Dataset '{DATASET_FILE}' carregado com sucesso (UTF-8).")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(DATASET_FILE, encoding='latin1')
            print(f"Dataset '{DATASET_FILE}' carregado com sucesso (Latin-1).")
        except Exception as e:
            print(f"Erro ao carregar o dataset com UTF-8 ou Latin-1: {e}")
            print("Por favor, verifique o nome do arquivo e a codificação.")
    except Exception as e:
        print(f"Erro inesperado ao carregar o dataset: {e}")
        print("Por favor, verifique o nome do arquivo.")
else:
    print(f"Erro: O arquivo '{DATASET_FILE}' não foi encontrado na mesma pasta do script.")
    print("Por favor, baixe o dataset 'olist.csv' e coloque-o na pasta correta.")
    sys.exit(1)

if df is None or TEXT_COLUMN not in df.columns or SENTIMENT_COLUMN not in df.columns:
    print(f"\nFalha ao preparar o DataFrame.")
    print(f"Verifique se as colunas '{TEXT_COLUMN}' e '{SENTIMENT_COLUMN}' existem no seu CSV.")
    sys.exit(1)

df.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN], inplace=True)
df = df.rename(columns={TEXT_COLUMN: 'text', SENTIMENT_COLUMN: 'sentiment'})

sentiment_mapping = {
    1: 'negative',
    2: 'negative',
    3: 'neutral',
    4: 'positive',
    5: 'positive'
}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)
df = df[df['sentiment'].isin(['positive', 'negative'])]

print(f"Dataset carregado e processado. Total de {len(df)} amostras.")
print(f"Primeiras 5 linhas do dataset após pré-processamento:\n{df.head()}\n")
print(f"Distribuição dos sentimentos:\n{df['sentiment'].value_counts()}\n")


def preprocess_text(text):
    """
    Limpa o texto para análise:
    - Converte para minúsculas
    - Remove caracteres especiais, números e pontuação
    - Remove espaços extras
    - REMOVE STOP WORDS EM PORTUGUÊS (NOVO)
    """
    if not isinstance(text, str):
        return "" 
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split() 
    filtered_words = [word for word in words if word not in STOP_WORDS_PORTUGUESE] 
    text = " ".join(filtered_words) 
    
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

if not df.empty:
    print("--- Exemplo de Pré-processamento (com Stop Words Removidas) ---")
    print(f"Original: '{df['text'].iloc[0]}'")
    print(f"Processado: '{df['cleaned_text'].iloc[0]}'\n")
else:
    print("DataFrame vazio após pré-processamento, pulando exemplo.\n")


vectorizer = TfidfVectorizer(max_features=5000) 

if not df.empty:
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    print("--- Exemplo de Vetorização (TF-IDF) ---")
    print(f"Texto: '{df['cleaned_text'].iloc[0]}'")
    print(f"Vetor (parte): {X[0].toarray()[0][:10]}...\n") 

    if len(df) < 2:
        print("Dataset muito pequeno para dividir em treino e teste. Mínimo de 2 amostras necessárias.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("--- Divisão de Dados ---")
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras\n")

    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("--- Treinamento do Modelo Concluído ---\n")

    y_pred = model.predict(X_test)

    print("--- Avaliação do Modelo ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

    def predict_sentiment(text_to_predict):
        """
        Preprocessa e vetoriza um novo texto, e então usa o modelo treinado para prever o sentimento.
        """
        cleaned_text = preprocess_text(text_to_predict)
        vectorized_text = vectorizer.transform([cleaned_text]) 
        
        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)[0]

        sentiment_label = prediction[0]
        positive_prob = probability[list(model.classes_).index('positive')] if 'positive' in model.classes_ else 0.0
        negative_prob = probability[list(model.classes_).index('negative')] if 'negative' in model.classes_ else 0.0

        print(f"\n--- Previsão para Novo Texto ---")
        print(f"Texto: '{text_to_predict}'")
        print(f"Sentimento Previsto: {sentiment_label.upper()}")
        print(f"Probabilidade (Positivo): {positive_prob:.2f}")
        print(f"Probabilidade (Negativo): {negative_prob:.2f}")
        print("-" * 30)

    print("\n--- Testando Previsões com Textos de Exemplo ---")
    predict_sentiment("Adorei o novo filme, muito divertido!")
    predict_sentiment("Que péssimo serviço, estou furioso.")
    predict_sentiment("O clima hoje está neutro, sem emoções.")
    predict_sentiment("O produto é bom, mas a entrega demorou.")
    predict_sentiment("Simplesmente sensacional, recomendo a todos!")

else:
    print("Não foi possível treinar o modelo devido a um DataFrame vazio ou erro no carregamento.")
    print("Por favor, corrija o caminho do arquivo, nomes das colunas ou problemas de codificação.")
    sys.exit(1)
