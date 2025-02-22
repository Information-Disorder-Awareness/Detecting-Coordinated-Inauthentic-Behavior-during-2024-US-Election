import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

social = ""
social_file = ""

from google.colab import drive

drive.mount('/content/drive')

# Esempio di utilizzo
file_graphml = "/content/drive/MyDrive/US ELECTION/dataset con autori/"+social_file+"/TSN/text_similarity_network.graphml"  # Sostituisci con il tuo file
file_csv = "/content/drive/MyDrive/US ELECTION/dataset con autori/"+social_file+"/TSN/coordinated_users.csv"  # Sostituisci con il tuo file
output_png = "/home/grafo_filtrato.png"

def visualizza_grafo_coordinati(file_graphml, file_csv, output_png, offset_y=0.0):
    # Carica il grafo dal file GraphML
    G = nx.read_graphml(file_graphml)

    # Carica il file CSV con i nodi coordinati
    coordinated_df = pd.read_csv(file_csv)
    # Assumiamo che la colonna che identifica i nodi sia 'author'
    coordinated_nodes = set(coordinated_df['author'])


    # Filtra il grafo per tenere solo i nodi coordinati
    filtered_G = G.subgraph(coordinated_nodes).copy()

    # Imposta lo stile della figura
    plt.figure(figsize=(18, 12))
    sns.set_style("white")

    # Usa un layout per una buona distribuzione dei nodi
    pos = nx.spring_layout(filtered_G, k=1.2, seed=42)

    # Disegna i nodi (tutti sono coordinati)
    nx.draw_networkx_nodes(filtered_G, pos,
                           node_color='red',
                           alpha=0.85,
                           node_size=80,
                           edgecolors='black')

    # Disegna gli archi
    nx.draw_networkx_edges(filtered_G, pos,
                           alpha=0.7,
                           edge_color='gray',
                           width=0.8)

    # Se le etichette appaiono troppo in alto, possiamo applicare un offset verticale.
    # Creiamo un nuovo dizionario per le posizioni delle etichette
    pos_labels = {node: (x, y - offset_y) for node, (x, y) in pos.items()}

    # Disegna le etichette dei nodi (nomi)
    nx.draw_networkx_labels(filtered_G, pos_labels,
                            font_size=9,
                            font_color='black',
                            verticalalignment='center',  # Allinea verticalmente al centro
                            horizontalalignment='center')  # Allinea orizzontalmente al centro

    # Rimuove gli assi e imposta il titolo
    plt.axis('off')
    plt.title("Coordinated "+ social +" users")

    # Salva il grafico in un file PNG
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()

    # Informazioni aggiuntive stampate a terminale
    total_nodes = len(G.nodes())
    print(f"Totale nodi nel grafo originale: {total_nodes}")

    coordinated_in_graph = len([n for n in G.nodes() if n in coordinated_nodes])
    print(f"Nodi coordinati presenti nel grafo originale: {coordinated_in_graph}")

    print(f"Il grafo filtrato (solo nodi coordinati) ha {filtered_G.number_of_nodes()} nodi.")


# Se il nome del nodo appare troppo in alto, prova ad aumentare offset_y
visualizza_grafo_coordinati(file_graphml, file_csv, output_png, offset_y=-0.05)
