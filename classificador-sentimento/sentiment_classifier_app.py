import tkinter as tk
from tkinter import messagebox
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

vectorizer = None
model = None

DATASET_FILE = 'olist.csv'
TEXT_COLUMN = 'review_text'
SENTIMENT_COLUMN = 'rating'

def load_and_prepare_data():
    """
    Carrega o dataset, pré-processa o texto, mapeia os sentimentos e balanceia as classes.
    Retorna o DataFrame processado e balanceado.
    """
    df = None
    if os.path.exists(DATASET_FILE):
        try:
            df = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATASET_FILE, encoding='latin1')
        except Exception as e:
            messagebox.showerror("Erro de Carregamento", f"Erro inesperado ao carregar o dataset: {e}")
            return None
    else:
        messagebox.showerror("Erro de Arquivo", f"Erro: O arquivo '{DATASET_FILE}' não foi encontrado na mesma pasta do script.")
        return None

    if df is None or TEXT_COLUMN not in df.columns or SENTIMENT_COLUMN not in df.columns:
        messagebox.showerror("Erro de Colunas", f"Falha ao preparar o DataFrame. Verifique se as colunas '{TEXT_COLUMN}' e '{SENTIMENT_COLUMN}' existem no seu CSV.")
        return None

    df.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN], inplace=True)
    df = df.rename(columns={TEXT_COLUMN: 'text', SENTIMENT_COLUMN: 'sentiment'})

    sentiment_mapping = {
        1: 'negative', 2: 'negative',
        3: 'neutral',
        4: 'positive', 5: 'positive'
    }
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    df = df[df['sentiment'].isin(['positive', 'negative'])] 

    print(f"Dataset carregado e processado. Total de {len(df)} amostras antes do balanceamento.")
    print(f"Distribuição dos sentimentos antes do balanceamento:\n{df['sentiment'].value_counts()}\n")

    df_positive = df[df['sentiment'] == 'positive']
    df_negative = df[df['sentiment'] == 'negative']

    if len(df_positive) > len(df_negative):
        df_positive_downsampled = df_positive.sample(n=len(df_negative), random_state=42)
        df_balanced = pd.concat([df_positive_downsampled, df_negative])
    else: 
        df_balanced = df 

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset balanceado. Total de {len(df_balanced)} amostras.")
    print(f"Distribuição dos sentimentos após balanceamento:\n{df_balanced['sentiment'].value_counts()}\n")

    return df_balanced 

def preprocess_text(text):
    """
    Limpa o texto para análise:
    - Converte para minúsculas
    - Remove caracteres especiais, números e pontuação
    - Remove espaços extras
    - Remove stop words em português
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

def train_sentiment_model():
    """
    Carrega os dados, pré-processa, vetoriza e treina o modelo de classificação.
    """
    global vectorizer, model

    df = load_and_prepare_data() 
    if df is None or df.empty:
        messagebox.showerror("Erro de Treinamento", "Não foi possível carregar ou processar o dataset para treinar o modelo.")
        return False

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    if df.empty:
        messagebox.showerror("Erro de Treinamento", "DataFrame vazio após pré-processamento. Não há dados para treinar o modelo.")
        return False

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    if len(df) < 2:
        messagebox.showerror("Erro de Treinamento", "Dataset muito pequeno para dividir em treino e teste. Mínimo de 2 amostras necessárias.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("--- Avaliação do Modelo ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
    
    messagebox.showinfo("Treinamento Concluído", "Modelo de Classificação de Sentimentos treinado com sucesso!")
    return True

def predict_sentiment_gui():
    """
    Pega o texto da caixa de entrada da GUI, preprocessa, vetoriza e prevê o sentimento.
    Exibe o resultado na GUI.
    """
    if model is None or vectorizer is None:
        messagebox.showwarning("Modelo Não Treinado", "O modelo de classificação ainda não foi treinado. Por favor, reinicie o aplicativo ou verifique o carregamento do dataset.")
        return

    text_to_predict = text_input.get("1.0", tk.END).strip()
    if not text_to_predict or text_to_predict == "Digite seu texto aqui...": 
        messagebox.showwarning("Entrada Vazia", "Por favor, digite um texto para classificar.")
        return

    cleaned_text = preprocess_text(text_to_predict)
    
    if not cleaned_text:
        sentiment_label.config(text="Sentimento: Indefinido")
        positive_prob_label.config(text="Prob. Positivo: N/A")
        negative_prob_label.config(text="Prob. Negativo: N/A")
        messagebox.showwarning("Texto Inválido", "O texto digitado não contém palavras significativas após o pré-processamento.")
        return

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)
    probability = model.predict_proba(vectorized_text)[0]

    sentiment_result = prediction[0]
    
    positive_prob = probability[list(model.classes_).index('positive')] if 'positive' in model.classes_ else 0.0
    negative_prob = probability[list(model.classes_).index('negative')] if 'negative' in model.classes_ else 0.0

    sentiment_label.config(text=f"Sentimento: {sentiment_result.upper()}")
    positive_prob_label.config(text=f"Prob. Positivo: {positive_prob:.2f}")
    negative_prob_label.config(text=f"Prob. Negativo: {negative_prob:.2f}")


def set_text_input_placeholder():
    text_input.delete("1.0", tk.END)
    text_input.insert("1.0", "Digite seu texto aqui...")
    text_input.config(fg=COLOR_TEXT_SECONDARY)

def on_text_input_focus_in(event):
    if text_input.get("1.0", tk.END).strip() == "Digite seu texto aqui...":
        text_input.delete("1.0", tk.END)
        text_input.config(fg=COLOR_TEXT_PRIMARY)

def on_text_input_focus_out(event):
    if not text_input.get("1.0", tk.END).strip():
        set_text_input_placeholder()


root = tk.Tk()
root.title("Classificador de Sentimentos")
root.geometry("600x550")
root.resizable(False, False)

COLOR_BG_LIGHT = "#E0F7FA"
COLOR_BG_CARD = "#FFFFFF"
COLOR_SHADOW = "#B0BEC5"
COLOR_TEXT_PRIMARY = "#263238"
COLOR_TEXT_SECONDARY = "#78909C"
COLOR_ACCENT_PRIMARY = "#00BCD4"
COLOR_ACCENT_HOVER = "#00838F"

FONT_FAMILY = "Verdana"
FONT_TITLE = (FONT_FAMILY, 24, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 14, "bold")
FONT_NORMAL = (FONT_FAMILY, 11)
FONT_RESULT = (FONT_FAMILY, 18, "bold")
FONT_PROB = (FONT_FAMILY, 10)

root.configure(bg=COLOR_BG_LIGHT)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, bg=COLOR_BG_LIGHT, padx=25, pady=25)
main_frame.grid(row=0, column=0, sticky="nsew")

main_frame.grid_rowconfigure(0, weight=0)
main_frame.grid_rowconfigure(1, weight=1)
main_frame.grid_rowconfigure(2, weight=0)
main_frame.grid_rowconfigure(3, weight=1)
main_frame.grid_columnconfigure(0, weight=1) 

app_title = tk.Label(main_frame, text="Classificador de Sentimentos", font=FONT_TITLE, bg=COLOR_BG_LIGHT, fg=COLOR_TEXT_PRIMARY)
app_title.grid(row=0, column=0, pady=(0, 20))

def create_shadow_card(parent_frame, grid_row, grid_column, pady_val):
    shadow_container = tk.Frame(parent_frame, bg=COLOR_SHADOW)
    shadow_container.grid(row=grid_row, column=grid_column, pady=pady_val, sticky="ew", padx=10) 

    content_frame = tk.Frame(shadow_container, bg=COLOR_BG_CARD, padx=20, pady=20, bd=1, relief="flat", highlightbackground=COLOR_SHADOW, highlightthickness=1)
    content_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    content_frame.grid_columnconfigure(0, weight=1)
    return content_frame

text_input_card_frame = create_shadow_card(main_frame, 1, 0, (0, 20))

input_label = tk.Label(text_input_card_frame, text="Digite seu texto aqui:", font=FONT_SUBTITLE, bg=COLOR_BG_CARD, fg=COLOR_TEXT_PRIMARY)
input_label.pack(pady=(0, 10), anchor="w")

text_input = tk.Text(text_input_card_frame, height=8, font=FONT_NORMAL, bd=1, relief="solid", bg=COLOR_BG_CARD, fg=COLOR_TEXT_PRIMARY, wrap="word", insertbackground=COLOR_TEXT_PRIMARY)
text_input.pack(fill="both", expand=True, pady=(0, 10))

set_text_input_placeholder() 

text_input.bind("<FocusIn>", on_text_input_focus_in)
text_input.bind("<FocusOut>", on_text_input_focus_out)


classify_button = tk.Button(main_frame, text="Classificar Sentimento", command=predict_sentiment_gui,
                            font=FONT_SUBTITLE, bg=COLOR_ACCENT_PRIMARY, fg="white",
                            activebackground=COLOR_ACCENT_HOVER, activeforeground="white",
                            bd=0, relief="flat", padx=20, pady=10, cursor="hand2")
classify_button.grid(row=2, column=0, pady=(0, 20))

results_card_frame = create_shadow_card(main_frame, 3, 0, (0, 0))

result_title_label = tk.Label(results_card_frame, text="Resultado da Análise", font=FONT_SUBTITLE, bg=COLOR_BG_CARD, fg=COLOR_TEXT_PRIMARY)
result_title_label.pack(pady=(0, 10), anchor="w")

sentiment_label = tk.Label(results_card_frame, text="Sentimento: ---", font=FONT_RESULT, bg=COLOR_BG_CARD, fg=COLOR_TEXT_PRIMARY)
sentiment_label.pack(pady=(5, 5))

positive_prob_label = tk.Label(results_card_frame, text="Prob. Positivo: ---", font=FONT_PROB, bg=COLOR_BG_CARD, fg=COLOR_TEXT_SECONDARY)
positive_prob_label.pack(pady=(2, 2))

negative_prob_label = tk.Label(results_card_frame, text="Prob. Negativo: ---", font=FONT_PROB, bg=COLOR_BG_CARD, fg=COLOR_TEXT_SECONDARY)
negative_prob_label.pack(pady=(2, 5))

if train_sentiment_model():
    print("Aplicativo pronto para classificação.")
else:
    messagebox.showerror("Erro de Inicialização", "O aplicativo não pôde ser inicializado devido a um erro no treinamento do modelo. Verifique o console para mais detalhes.")
    root.destroy()

root.mainloop()
