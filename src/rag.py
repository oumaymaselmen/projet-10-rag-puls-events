import os
import json
import faiss
import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def load_faiss_index(index_path: str = "data/faiss_index",
                     metadata_path: str = "data/metadata.json"):
    """Charge l'index FAISS et les métadonnées."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def get_query_embedding(query: str) -> np.ndarray:
    """Transforme la question en vecteur."""
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[query]
    )
    return np.array([response.data[0].embedding], dtype="float32")


def search_events(query: str, index, metadata, top_k: int = 5) -> list:
    """Cherche les événements les plus pertinents."""
    query_vector = get_query_embedding(query)
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            event = metadata[idx].copy()
            event["score"] = float(distances[0][i])
            results.append(event)
    return results


def generate_response(query: str, events: list) -> str:
    """Génère une réponse avec Mistral basée sur les événements trouvés."""
    
    context = ""
    for i, event in enumerate(events, 1):
        context += f"""
Événement {i} :
- Titre : {event.get('title_fr', 'N/A')}
- Description : {event.get('description_fr', 'N/A')[:300]}
- Ville : {event.get('location_city', 'N/A')}
- Date : {event.get('firstdate_begin', 'N/A')}
- Lien : {event.get('canonicalurl', 'N/A')}
"""

    prompt = f"""Tu es un assistant culturel pour Puls-Events spécialisé dans les événements en Île-de-France.
    
Basé sur les événements suivants, réponds à la question de l'utilisateur de manière chaleureuse et utile.
Si les événements ne correspondent pas exactement, propose les plus proches.

Événements disponibles :
{context}

Question : {query}

Réponds en français de manière concise et engageante."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def ask(query: str) -> str:
    """Fonction principale du RAG."""
    index, metadata = load_faiss_index()
    events = search_events(query, index, metadata)
    response = generate_response(query, events)
    return response


if __name__ == "__main__":
    question = "Quels concerts de jazz y a-t-il à Paris ce mois-ci ?"
    print(f" Question : {question}\n")
    print(" Réponse :")
    print(ask(question))