import pytest
import pandas as pd
import json
from datetime import datetime, timedelta

# Chemins des fichiers
CLEAN_DATA_PATH = "data/raw/events_clean.json"
METADATA_PATH = "data/metadata.json"
REGION = "Île-de-France"
DATE_LIMIT = datetime.now() - timedelta(days=365)


def load_clean_data():
    return pd.read_json(CLEAN_DATA_PATH)


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


#  Test 1 : Le fichier de données existe et n'est pas vide
def test_data_not_empty():
    df = load_clean_data()
    assert len(df) > 0, "Le fichier de données est vide !"
    print(f" {len(df)} événements trouvés")


#  Test 2 : Tous les événements sont en Île-de-France
def test_region_is_idf():
    df = load_clean_data()
    assert "location_region" in df.columns, "Colonne location_region manquante"
    non_idf = df[df["location_region"] != REGION]
    assert len(non_idf) == 0, f"{len(non_idf)} événements hors Île-de-France !"
    print(f" Tous les événements sont en {REGION}")


#  Test 3 : Tous les événements sont dans la dernière année
def test_dates_within_one_year():
    df = load_clean_data()
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], unit="ms", errors="coerce")
    old_events = df[df["firstdate_begin"] < DATE_LIMIT]
    assert len(old_events) == 0, f"{len(old_events)} événements de plus d'un an !"
    print(f" Toutes les dates sont dans la dernière année")


#  Test 4 : Pas de titres manquants
def test_no_missing_titles():
    df = load_clean_data()
    missing = df["title_fr"].isna().sum()
    assert missing == 0, f"{missing} titres manquants !"
    print(" Aucun titre manquant")


#  Test 5 : Les métadonnées FAISS contiennent au moins autant que les données
def test_metadata_matches_data():
    df = load_clean_data()
    metadata = load_metadata()
    assert len(metadata) >= len(df), \
        f"Métadonnées ({len(metadata)}) < données ({len(df)})"
    print(f" Métadonnées cohérentes : {len(metadata)} chunks pour {len(df)} événements")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])