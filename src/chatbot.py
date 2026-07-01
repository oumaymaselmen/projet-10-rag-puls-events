import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import ask


def run_chatbot():
    """
    Lance le chatbot interactif Puls-Events en ligne de commande.

    Boucle infinie de questions/réponses jusqu'à la saisie de 'quitter'.
    Utilise le pipeline RAG (rag.py) pour générer les réponses en français
    basées sur les événements culturels indexés dans FAISS.

    Returns:
        None
    """
    print("=" * 50)
    print(" Bienvenue sur Puls-Events Chatbot !")
    print("Je vous aide à trouver des événements culturels en Île-de-France.")
    print("Tapez 'quitter' pour arrêter.")
    print("=" * 50)

    while True:
        print()
        question = input(" Votre question : ").strip()

        if not question:
            continue

        if question.lower() in ["quitter", "exit", "quit"]:
            print(" À bientôt !")
            break

        print("\n Réponse en cours...\n")
        try:
            response = ask(question)
            print(f" {response}")
        except Exception as e:
            print(f" Erreur : {e}")


if __name__ == "__main__":
    run_chatbot()