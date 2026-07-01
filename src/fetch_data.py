import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

def fetch_events(max_records: int = 500) -> pd.DataFrame:
    """
    Récupère les événements culturels depuis l'API OpenDataSoft.

    Args:
        max_records (int): Nombre maximum d'événements à récupérer. Défaut: 500

    Returns:
        pd.DataFrame: DataFrame des événements Île-de-France des 12 derniers mois.

    Raises:
        requests.exceptions.RequestException: Si l'API est inaccessible.
    """
    date_limit = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    all_records = []
    offset = 0
    limit = 100

    print(f"📡 Récupération des événements Île-de-France depuis {date_limit}...")

    while len(all_records) < max_records:
        params = {
            "where": f"location_region=\"Île-de-France\" AND firstdate_begin>\"{date_limit}\"",
            "select": "title_fr,description_fr,firstdate_begin,firstdate_end,location_name,location_city,location_region,keywords_fr,canonicalurl",
            "limit": limit,
            "offset": offset
        }

        response = requests.get(BASE_URL, params=params, timeout=30)

        if response.status_code != 200:
            print(f" Erreur : {response.status_code} - {response.text[:200]}")
            break

        data = response.json()
        results = data.get("results", [])
        total = data.get("total_count", 0)

        if not results:
            print(" Toutes les données récupérées.")
            break

        all_records.extend(results)
        offset += limit
        print(f"   → {len(all_records)} / {total} événements récupérés")

        if len(results) < limit:
            break

    if not all_records:
        print(" Aucun événement trouvé.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["title_fr", "firstdate_begin"])
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], errors="coerce")
    df["firstdate_end"] = pd.to_datetime(df["firstdate_end"], errors="coerce")

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/events_idf.json"
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f" Sauvegardé : {output_path} ({len(df)} événements)")

    return df


if __name__ == "__main__":
    df = fetch_events()
    if not df.empty:
        print(f"\n Aperçu :")
        print(df[["title_fr", "firstdate_begin", "location_city"]].head(5).to_string())