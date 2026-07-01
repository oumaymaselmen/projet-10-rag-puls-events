import json
import sys
import os
import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag import ask

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


def get_embedding(text: str) -> np.ndarray:
    """Transforme un texte en vecteur."""
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcule la similarité entre deux vecteurs."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def evaluate():
    with open("tests/qa_dataset.json", "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    print(" Évaluation du système RAG")
    print("=" * 50)

    resultats = []
    score_total = 0

    for qa in qa_dataset:
        question = qa["question"]
        reponse_attendue = qa["reponse_attendue"]

        print(f"\n Question {qa['id']} : {question}")

        # Réponse du chatbot
        reponse_chatbot = ask(question)
        print(f" Réponse chatbot : {reponse_chatbot[:200]}...")
        print(f" Réponse attendue : {reponse_attendue}")

        # Calcul similarité cosinus
        vecteur_attendu = get_embedding(reponse_attendue)
        vecteur_chatbot = get_embedding(reponse_chatbot)
        score = cosine_similarity(vecteur_attendu, vecteur_chatbot) * 100

        print(f" Score similarité : {score:.1f}%")
        score_total += score

        resultats.append({
            "id": qa["id"],
            "question": question,
            "reponse_attendue": reponse_attendue,
            "reponse_chatbot": reponse_chatbot,
            "score": round(score, 1)
        })

    score_moyen = score_total / len(qa_dataset)
    print("\n" + "=" * 50)
    print(f" Score moyen du système RAG : {score_moyen:.1f}%")

    with open("tests/resultats_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    print(" Résultats sauvegardés : tests/resultats_evaluation.json")


if __name__ == "__main__":
    evaluate()