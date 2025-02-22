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

# ============================================
# 2. Caricamento del Dataset e casting dei tipi
# ============================================
# Specifica il percorso del file CSV su Google Drive
csv_path = ""

# Carica il dataset
df = pd.read_csv(csv_path)

# Conversione della colonna 'create_time' in formato datetime
df['create_time'] = pd.to_datetime(df['create_time'].str.strip(), dayfirst=True, format="%Y-%m-%d  %H:%M:%S")

# Verifica che le colonne necessarie siano presenti
assert set(['create_time', 'text', 'author']).issubset(df.columns), "Errore: colonne mancanti nel CSV!"

# Visualizza le prime righe del dataset
print("Prime righe del dataset:")
print(df.head())

# Converte la colonna degli embedding da stringa a array NumPy
df['embedding'] = df['embedding'].progress_apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))


# ============================================
# 3. Creazione della Matrice di similarità
# ============================================

# Estrazione del numero di settimane
df['date'] = df['create_time'].dt.isocalendar().week

# Get unique authors once
print(f"Autori prima del filtering: {len(df['author'].unique())}")

# Contiamo quante righe ha ogni autore
conteggio_autori = df['author'].value_counts()

# Filtriamo per tenere solo gli autori con almeno 20 righe
autori_frequenti = conteggio_autori[conteggio_autori >= 20].index

# Filtriamo il DataFrame originale
df = df[df['author'].isin(autori_frequenti)]

authors = df['author'].unique()
print(f"Autori dopo il filtering: {len(authors)}")
author_to_idx = {author: idx for idx, author in enumerate(authors)}
n_authors = len(authors)

# Inizializzazione matrice
final_similarity_matrix = np.zeros((n_authors, n_authors))
processed_windows = 0

# Pre-compute weeks
weeks = sorted(df['date'].unique())

for week in weeks:
    print(f"Process week: {week}")
    weekly_df = df[df['date'] == week]

    # Compute mean embeddings
    weekly_grouped = weekly_df.groupby('author').agg({
        'embedding': lambda x: np.mean(np.vstack(x), axis=0)
    }).reset_index()

    if len(weekly_grouped) < 2:
        continue

    # Compute similarities
    embedding_matrix = np.vstack(weekly_grouped['embedding'].values)
    weekly_matrix = cosine_similarity(embedding_matrix)

    # Update accumulated similarity using vectorized operations
    for i, author_i in enumerate(weekly_grouped['author']):
        for j, author_j in enumerate(weekly_grouped['author']):
            final_similarity_matrix[author_to_idx[author_i], author_to_idx[author_j]] += weekly_matrix[i,j]

    processed_windows += 1

final_similarity_matrix /= processed_windows
print(f"\nNumero di settimane analizzate (con almeno 2 autori): {processed_windows}")


# ============================================
# 4. Creazione della Text Similarity Network (TSN)
# ============================================
# Definisci la soglia di similarità per considerare due autori "coordinati"

SIMILARITY_THRESHOLD = 0.8

# Crea il grafo TSN utilizzando NetworkX
G = nx.Graph()

# Aggiungi i nodi (autori) basandoti sul mapping globale
for author in authors:
    G.add_node(author)

# Aggiungi gli archi per coppie di autori con similarità superiore alla soglia
for i in tqdm(range(0, len(authors)), desc="Aggiungengo archi ai nodi"):
    for j in range(i+1, len(authors)):
        if final_similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            G.add_edge(authors[i], authors[j], weight=final_similarity_matrix[i, j])

print("\nDettagli della rete TSN:")
print("Numero di nodi (autori):", G.number_of_nodes())
print("Numero di archi (connessioni):", G.number_of_edges())

# ============================================
# 5. Calcolo della Centralità degli Autovalori per Individuare Utenti Coordinati
# ============================================

try:
    eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence as e:
    print("⚠️ Convergenza non raggiunta, utilizzo metodo numpy:", e)
    eigen_centrality = nx.eigenvector_centrality_numpy(G)

# Ordina gli autori per centralità (dal più alto al più basso)
sorted_authors = sorted(eigen_centrality.items(), key=lambda x: x[1], reverse=True)

# Identifica i top 0.5% degli autori come coordinati
top_percent = 0.005
n_top = max(1, int(top_percent * len(sorted_authors)))
threshold_centrality = sorted_authors[n_top - 1][1]
coordinated_authors = [author for author, score in eigen_centrality.items() if score >= threshold_centrality]

print(f"\nNumero di autori coordinati (top {top_percent*100}%): {len(coordinated_authors)}")
print("Alcuni autori coordinati:")
for author in coordinated_authors[:10]:
    print(f"{author} -> Centralità: {eigen_centrality[author]:.4f}")

    
# ============================================
# 6. Salvataggio dei Risultati
# ============================================

# 6.1 Salva la lista degli autori coordinati in un file CSV (nella cartella principale di MyDrive)
coordinated_df = pd.DataFrame({"author": coordinated_authors})
coordinated_csv_path = "/content/coordinated_users.csv"
coordinated_df.to_csv(coordinated_csv_path, index=False)
print(f"\nLista degli autori coordinati salvata in: {coordinated_csv_path}")

# 6.2 Salva la rete TSN in formato GraphML (per analisi in Gephi o altri strumenti)
graphml_path = "/content/text_similarity_network.graphml"
nx.write_graphml(G, graphml_path)
print(f"Rete TSN salvata in formato GraphML in: {graphml_path}")

# 6.3 Salva la visualizzazione della rete come immagine PNG (senza etichette degli utenti)
plt.figure(figsize=(12, 12))

# Calcola il layout della rete
pos = nx.spring_layout(G, k=0.15)

# Definisce i colori dei nodi: rosso per gli autori coordinati, blu per gli altri
node_colors = ['red' if node in coordinated_authors else 'blue' for node in G.nodes()]

# Disegna i nodi e gli archi della rete
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Text Similarity Network (TSN) - Autori Coordinati evidenziati in rosso")
plt.axis("off")

# Salva l'immagine in alta risoluzione su Google Drive
image_path = "/content/tsn_network.png"
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Immagine della rete salvata in: {image_path}")