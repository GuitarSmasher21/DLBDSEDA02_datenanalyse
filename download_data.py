# Dieses Skript ermöglicht den Download des unstrukturieten Datensatzes vom Berlin Ordnugnsamt
# Das Skript muss nicht erneut ausgeführt werden, die Daten sind bereits in GitHub abgelegt

import requests
import json
import os
import sys

# URL des öffentlichen Datensatzes "Ordnungsamt Online" von Berlin
DATA_URL = "https://ordnungsamt.berlin.de/frontend.webservice.opendata/api/meldungen"

# Der Dateiname, den das Hauptskript (main.py) erwartet
OUTPUT_FILENAME = "temp_unstructured_data.json"

def download_data():
    print(f"Starte Daten-Download")
    try:
        import requests 
    except ImportError:
        print("FEHLER: Das 'requests'-Modul ist nicht installiert. Bitte 'pip install requests' ausführen.")
        sys.exit(1)
    try:
        # 1. Anfrage an URL senden
        response = requests.get(DATA_URL)
        response.raise_for_status() # Auf HTTP-Fehler prüfen
        # 2. Daten als JSON verarbeiten
        data = response.json()
        # 3. Daten lokal speichern
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"ERFOLG! Daten wurden in '{OUTPUT_FILENAME}' gespeichert.")

    except requests.exceptions.RequestException as e:
        print(f"KRITISCHER FEHLER beim Download (Internet- oder Serverproblem): {e}")
    except json.JSONDecodeError:
        print("KRITISCHER FEHLER: Die heruntergeladene Datei ist kein gültiges JSON.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

download_data()