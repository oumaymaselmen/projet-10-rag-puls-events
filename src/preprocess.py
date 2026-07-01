import pandas as pd
import json
import os
from datetime import datetime, timedelta


def preprocess_events(input_path: str = "data/raw/events_idf.json",
                      output_path: str = "data/raw/events_clean.json") -> pd.DataFrame:
    """
    Nettoie et prépare les événements pour la vectorisation.

    Opérations effectuées :
        1. Conversion et validation des dates
        2. Filtrage strict : événements des 12 derniers mois uniquement
        3. Suppression des doublons (title_fr + firstdate_begin)
        4. Suppression des valeurs manquantes (title_fr, description_fr)
        5. Création de la colonne 'text_for_embedding'

    Args:
        input_path (str): Chemin vers les données brutes. Défaut: data/raw/events_idf.json
        output_path (str): Chemin de sauvegarde. Défaut: data/raw/events_clean.json

    Returns:
        pd.DataFrame: DataFrame nettoyé prêt pour la vectorisation.
    """
    print(" Chargement des données brutes...")
    df = pd.read_json(input_path)
    print(f"   → {len(df)} événements chargés")

    # Conversion dates
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], unit="ms", errors="coerce")
    df["firstdate_end"] = pd.to_datetime(df["firstdate_end"], unit="ms", errors="coerce")

    # Filtre strict : événements de moins d'un an uniquement
    date_limit = datetime.now() - timedelta(days=365)
    df = df[df["firstdate_begin"] >= date_limit]
    print(f"   → {len(df)} après filtre date (moins d'un an)")

    # Suppression doublons
    df = df.drop_duplicates(subset=["title_fr", "firstdate_begin"])
    print(f"   → {len(df)} après suppression des doublons")

    # Suppression lignes sans description
    df = df.dropna(subset=["title_fr", "description_fr"])
    print(f"   → {len(df)} après suppression des valeurs manquantes")

    # Nettoyage texte
    df["title_fr"] = df["title_fr"].str.strip()
    df["description_fr"] = df["description_fr"].str.strip()
    df["location_city"] = df["location_city"].fillna("Ville inconnue")
    df["keywords_fr"] = df["keywords_fr"].fillna("")

    # Création colonne texte pour vectorisation
    df["text_for_embedding"] = (
        "Titre : " + df["title_fr"] + "\n" +
        "Description : " + df["description_fr"] + "\n" +
        "Ville : " + df["location_city"] + "\n" +
        "Date : " + df["firstdate_begin"].astype(str) + "\n" +
        "Mots-clés : " + df["keywords_fr"].astype(str)
    )

    # Sauvegarde
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f" Données nettoyées sauvegardées : {output_path} ({len(df)} événements)")

    return df


if __name__ == "__main__":
    df = preprocess_events()
    print(f"\n Aperçu du texte pour vectorisation :")
    print(df["text_for_embedding"].iloc[0])