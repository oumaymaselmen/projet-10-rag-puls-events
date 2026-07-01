# Puls-Events RAG — POC Chatbot Culturel

Proof of Concept d'un système de recommandation d'événements culturels
basé sur une architecture RAG (Retrieval-Augmented Generation).

## Objectifs

- Récupérer des événements culturels en Île-de-France via OpenAgenda
- Indexer ces événements dans une base vectorielle FAISS
- Répondre en langage naturel via le modèle Mistral AI
- Valider la qualité avec un score de similarité cosinus de 89.6%

## Technologies utilisées

- **LangChain** : orchestration du pipeline RAG
- **Mistral AI** : embeddings (mistral-embed) et génération (mistral-small)
- **FAISS** : base de données vectorielle
- **OpenDataSoft** : source de données OpenAgenda
- **Pandas** : nettoyage des données
- **pytest** : tests unitaires
- **Python 3.10+**

## Installation

### 1. Cloner le projet
git clone <url_du_repo>
cd "projet 11 Concevez et déployez un système RAG"

### 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

### 3. Installer les dépendances
pip install -r requirements.txt

### 4. Configurer les variables d'environnement
Crée un fichier `.env` à la racine du projet :
OPENAGENDA_API_KEY=ta_cle_openagenda
MISTRAL_API_KEY=ta_cle_mistral

### 5. Vérifier l'installation
python test_imports.py

## Lancement

### Reconstruire la base vectorielle
python src/fetch_data.py
python src/preprocess.py
python src/vectorize.py

### Lancer les tests unitaires
python -m pytest tests/test_data.py -v

### Lancer le chatbot
python src/chatbot.py

### Évaluer la qualité des réponses
python tests/evaluate.py

## Structure du projet
projet 11/
├── src/
│   ├── fetch_data.py         # Récupération données OpenDataSoft
│   ├── preprocess.py         # Nettoyage et préparation
│   ├── vectorize.py          # Vectorisation + index FAISS
│   ├── rag.py                # Pipeline RAG LangChain + Mistral
│   └── chatbot.py            # Interface chatbot CLI
├── tests/
│   ├── test_data.py          # 5 tests unitaires pytest
│   ├── qa_dataset.json       # Jeu de données Q/R annoté
│   ├── evaluate.py           # Évaluation similarité cosinus
│   └── resultats_evaluation.json
├── data/
│   ├── raw/
│   │   ├── events_idf.json   # Données brutes
│   │   └── events_clean.json # Données nettoyées
│   ├── faiss_index           # Index vectoriel FAISS
│   └── metadata.json         # Métadonnées des événements
├── .env                      # Variables d'environnement (non versionné)
├── .gitignore
├── requirements.txt
├── test_imports.py
└── README.md

## Périmètre géographique

Île-de-France — événements des 12 derniers mois.

## Résultats du POC

- **488** événements récupérés
- **485** événements après nettoyage
- **495** chunks vectorisés (1024 dimensions)
- **5/5** tests unitaires passés
- **89.6%** score de qualité moyen (similarité cosinus)