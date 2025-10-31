## scripts/schritt_2_daten_laden.py
import pandas as pd # DataFrame
import json # JSON Datei laden

def run(datei_pfad, record_path):
    # Lädt und normalisiert die JSON-Daten der Berliner Ordnungsamt-Datei in einen DataFrame.

    try:
        with open(datei_pfad, 'r', encoding='utf-8') as f:  # Dateipfad in main.py angegeben. Datei sollte lokal aufgerufen werden.
            data = json.load(f)
        df = pd.json_normalize(data, record_path=record_path) # # "records_path" angeben, damit Spalten richtig eingelesen werden.
        print("Daten erfolgreich geladen. Erste Zeilen des DataFrames:")
        print(df.head(5)) # Ausgabe erste 5 Zeilen von DataFrame für Überprüfung.
        print(f"\nGesamtzahl der Beschwerden (Zeilen): {len(df)}") 
        return df # Genutzt für späteren Abgleich der Korrektheit
    except FileNotFoundError:
        print("FEHLER: 'daten.json' nicht gefunden. Bitte Dateipfad überprüfen.") 
        return pd.DataFrame() # Prüfung + Fehlermeldung auf falschem Dateipfad
    except Exception as e:
        print(f"FEHLER beim Laden oder Parsen der Daten: {e}") 
        return pd.DataFrame() # Prüfung auf Prozessing-Fehler