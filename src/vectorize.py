import os
import json
import faiss
import numpy as np
import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Découpage en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def get_embeddings(texts: list) -> np.ndarray:
    """
    Transforme une liste de textes en vecteurs avec Mistral Embed.

    Args:
        texts (list): Liste de textes à vectoriser.

    Returns:
        np.ndarray: Matrice de vecteurs de forme (n_texts, 1024).
    """
    embeddings = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"   → Vectorisation batch {i//batch_size + 1}/{(len(texts)//batch_size) + 1}")

        response = client.embeddings.create(
            model="mistral-embed",
            inputs=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings, dtype="float32")


def build_faiss_index(input_path: str = "data/raw/events_clean.json",
                      index_path: str = "data/faiss_index",
                      metadata_path: str = "data/metadata.json"):
    """
    Construit la base vectorielle FAISS depuis les données nettoyées.

    Pipeline interne :
        1. Chargement des données nettoyées (events_clean.json)
        2. Découpage en chunks (RecursiveCharacterTextSplitter, size=500, overlap=50)
        3. Vectorisation par batches de 10 avec mistral-embed
        4. Construction de l'index FAISS (IndexFlatL2, dimension=1024)
        5. Sauvegarde de l'index et des métadonnées

    Args:
        input_path (str): Chemin vers les données nettoyées.
        index_path (str): Chemin de sauvegarde de l'index FAISS.
        metadata_path (str): Chemin de sauvegarde des métadonnées JSON.

    Returns:
        None — fichiers sauvegardés dans data/
    """
    print(" Chargement des données nettoyées...")
    df = pd.read_json(input_path)
    print(f"   → {len(df)} événements")

    # Découpage en chunks
    print(" Découpage des textes en chunks...")
    all_chunks = []
    all_metadata = []

    for _, row in df.iterrows():
        text = row["text_for_embedding"]
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "title_fr": row["title_fr"],
                "description_fr": row["description_fr"],
                "firstdate_begin": str(row["firstdate_begin"]),
                "location_city": row["location_city"],
                "keywords_fr": str(row["keywords_fr"]),
                "canonicalurl": row["canonicalurl"]
            })

    print(f"   → {len(all_chunks)} chunks créés depuis {len(df)} événements")

    # Vectorisation
    print(" Vectorisation avec Mistral...")
    embeddings = get_embeddings(all_chunks)
    print(f"   → Dimension des vecteurs : {embeddings.shape}")

    # Construction FAISS
    print("  Construction de la base FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"   → {index.ntotal} vecteurs indexés")

    # Sauvegarde
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f" Index FAISS sauvegardé : {index_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f" Métadonnées sauvegardées : {metadata_path}")


if __name__ == "__main__":
    build_faiss_index()
    print("\n Base vectorielle construite avec succès !")