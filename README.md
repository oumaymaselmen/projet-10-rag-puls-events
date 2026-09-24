# Projet 10 - Système RAG Puls-Events (LangChain, Mistral, FAISS)

> Formation Data Engineer - OpenClassrooms | Outils : Python, LangChain, Mistral AI, FAISS

## Objectif

Concevoir et déployer un système de question-réponse basé sur la technique RAG 
(Retrieval-Augmented Generation) pour Puls-Events, une plateforme d'événements 
en Île-de-France. Le chatbot répond aux questions des utilisateurs en s'appuyant 
sur une base de connaissances vectorisée.

## Architecture RAG

1. Collecte des données : récupération des événements via API (fetch_data.py)
2. Preprocessing : nettoyage et structuration des données (preprocess.py)
3. Vectorisation : création des embeddings et indexation FAISS (vectorize.py)
4. RAG : retrieval et génération de réponses avec Mistral (rag.py)
5. Chatbot : interface de conversation (chatbot.py)

## Compétences travaillées

- Implémentation d'un pipeline RAG complet
- Vectorisation de documents avec FAISS
- Intégration de LLM (Mistral AI) via LangChain
- Évaluation de la qualité des réponses (score cosine similarity 89.6%)

## Contenu du repo

- src/ : scripts Python du pipeline RAG
- data/ : index FAISS, metadata et données JSON des événements
- tests/ : scripts d'évaluation et datasets de test

## Installation

pip install -r requirements.txt

Créer un fichier .env avec votre clé API Mistral :
MISTRAL_API_KEY=votre_cle_ici

## Résultats

Chatbot RAG évalué à 89,6 % de similarité cosinus — base du projet MVP Puls-Events

---
Formation Data Engineer - OpenClassrooms
