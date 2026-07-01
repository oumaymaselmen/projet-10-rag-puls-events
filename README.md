# Projet 10 - Systeme RAG Puls-Events (LangChain, Mistral, FAISS)

> Formation Data Engineer - OpenClassrooms | Outils : Python, LangChain, Mistral AI, FAISS

## Objectif

Concevoir et deployer un systeme de question-reponse base sur la technique RAG (Retrieval-Augmented Generation) pour Puls-Events, une plateforme d'evenements en Ile-de-France. Le chatbot repond aux questions des utilisateurs en s'appuyant sur une base de connaissances vectorisee.

## Architecture RAG

1. Collecte des donnees : recuperation des evenements via API (fetch_data.py)
2. Preprocessing : nettoyage et structuration des donnees (preprocess.py)
3. Vectorisation : creation des embeddings et indexation FAISS (vectorize.py)
4. RAG : retrieval et generation de reponses avec Mistral (rag.py)
5. Chatbot : interface de conversation (chatbot.py)

## Competences travaillees

- Implementation d'un pipeline RAG complet
- Vectorisation de documents avec FAISS
- Integration de LLM (Mistral AI) via LangChain
- Evaluation de la qualite des reponses (score cosine similarity 89.6%)

## Contenu du repo

- src/ : scripts Python du pipeline RAG
- data/ : index FAISS, metadata et donnees JSON des evenements
- tests/ : scripts d'evaluation et datasets de test

## Resultats

Score de similarite cosine : 89.6% sur le dataset d'evaluation

## Installation

pip install -r requirements.txt

Creer un fichier .env avec votre cle API Mistral :
MISTRAL_API_KEY=votre_cle_ici

---
Formation Data Engineer - OpenClassrooms
