# ============================================
# 1. Montaggio di Google Drive e importazione delle librerie
# ============================================
from google.colab import drive
import pandas as pd
import numpy as np
import re
import string
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm

tqdm = tqdm.pandas()


# Monta Google Drive
drive.mount('/content/drive')

# Scarica le stopwords (se non sono già presenti)
nltk.download('stopwords')

# ============================================
# 2. Caricamento del Dataset
# ============================================
# Specifica il percorso del file CSV su Google Drive
csv_path = "/content/drive/MyDrive/US ELECTION/dataset con autori/telegram/dataset.csv"
embedding_csv_path = "/content/dataset_telegram_with_embedding.csv"

# Carica il dataset (modifica l'encoding se necessario)
df = pd.read_csv(csv_path)

# Conversione della colonna 'create_time' in formato datetime
# Assicurati che il formato corrisponda a quello presente nel file CSV
df['create_time'] = pd.to_datetime(df['create_time'].str.strip(), dayfirst=True, format="%Y-%m-%d  %H:%M:%S")

# Verifica che le colonne necessarie siano presenti
assert set(['create_time', 'text', 'author']).issubset(df.columns), "Errore: colonne mancanti nel CSV!"

# Visualizza le prime righe del dataset
print("Prime righe del dataset:")
print(df.head())

# ============================================
# 3. Preprocessing e Calcolo degli Embedding per Messaggio
# ============================================

stop_words = set(stopwords.words('english'))

# Funzione di pulizia del testo
def clean_text(text):
    # Converti in minuscolo
    text = text.lower()
    # Rimuovi URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Rimuovi punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Rimuovi numeri
    text = re.sub(r'\d+', '', text)
    # Rimuovi spazi extra
    text = re.sub(r'\s+', ' ', text).strip()
    # Rimuovi stopwords (in inglese)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Pulisce i nomi degli autori: rimuove spazi indesiderati
df['author'] = df['author'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

# Applica la funzione di pulizia del testo a ciascun messaggio
df['clean_text'] = df['text'].progress_apply(clean_text)

# Filtra i messaggi con meno di 4 parole nel testo pulito (opzionale)
df = df[df['clean_text'].str.split().str.len() > 4]

# Inizializza il modello SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Calcola gli embedding per ogni messaggio utilizzando il modello
df['embedding'] = df['clean_text'].progress_apply(lambda x: model.encode(x, convert_to_tensor=False))

df.to_csv(embedding_csv_path, index=False)
print("Nuovo file CSV con la colonna 'embedding' salvato in:", embedding_csv_path)

# (Opzionale) Aggregazione globale degli embedding per autore per visualizzazione
global_grouped = df.groupby('author').agg({
    'embedding': lambda embeddings: np.mean(np.vstack(embeddings), axis=0)
}).reset_index()

print("\nEsempi di embedding aggregati per autore (global grouping):")
for index, row in global_grouped.head().iterrows():
    print("Autore:", row['author'])
    print("Embedding (shape {}):".format(row['embedding'].shape), row['embedding'])
    print("-" * 50)
