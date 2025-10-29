import pandas as pd
import json
import nltk


try:
    ## JSON-Datei einlesen
    with open('/Users/michi/Documents/GitHub/DLBDSEDA02_datenanalyse/unstructured_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## Daten in Pandas DataFrame laden
    df = pd.json_normalize(data)
    
    print("Daten erfolgreich geladen. Erste Zeilen des DataFrames:") ## Ausgabe DataFrame für Überprüfung
    print(df.head())

except FileNotFoundError:
    print("FEHLER: 'daten.json' nicht gefunden. Bitte Dateipfad überprüfen.") ## Prüfung + Fehlermeldung auf falschem Dateipfad
except Exception as e:
    print(f"FEHLER beim Laden oder Parsen der Daten: {e}") ## Prüfung auf Prozessing-Fehler

'''
# 2. Textdaten extrahieren
# Sie müssen die Spalte identifizieren, die die unstrukturierten Beschwerdetexte enthält.
# Angenommen, diese Spalte heißt 'beschwerdetext' oder 'thema'.
TEXT_COLUMN = 'thema' # Passen Sie dies an Ihren tatsächlichen Datensatz an 

if TEXT_COLUMN in df.columns:
    texts = df[TEXT_COLUMN].astype(str)
    print(f"\nTextdaten aus Spalte '{TEXT_COLUMN}' extrahiert (Anzahl: {len(texts)})")
    print("Beispiel für einen Text:", texts.iloc[0])
else:
    print(f"\nFEHLER: Spalte '{TEXT_COLUMN}' nicht im DataFrame gefunden. Verfügbare Spalten: {df.columns.tolist()}")

# Lade die benötigten NLTK-Ressourcen, falls noch nicht geschehen
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Ab hier haben Sie die zu analysierenden Texte in der Variable 'texts'
'''