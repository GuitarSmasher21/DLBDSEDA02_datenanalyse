import pandas as pd
import json
import nltk


try:
    ## JSON-Datei einlesen
    with open('/Users/michi/Documents/GitHub/DLBDSEDA02_datenanalyse/unstructured_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## Daten in Pandas DataFrame laden
    df = pd.json_normalize(data, record_path=['index']) ## "records_path" angeben, damit Spalten richtig eingelesen werden
    
    print("Daten erfolgreich geladen. Erste Zeilen des DataFrames:") ## Ausgabe DataFrame für Überprüfung
    print(df.head(2)) ## Gibt die ersten zwei Zeilen des DataFrames aus
    print(f"\nGesamtzahl der Beschwerden (Zeilen): {len(df)}") ## Genutzt für späteren Abgleich

except FileNotFoundError:
    print("FEHLER: 'daten.json' nicht gefunden. Bitte Dateipfad überprüfen.") ## Prüfung + Fehlermeldung auf falschem Dateipfad
except Exception as e:
    print(f"FEHLER beim Laden oder Parsen der Daten: {e}") ## Prüfung auf Prozessing-Fehler


## Spalte identifizieren, die die unstrukturierten Beschwerdetexte enthält.
TEXT_COLUMN = 'sachverhalt' ##  Im Fall Berlin wird 'Sachverhalt' gewählt.

if TEXT_COLUMN in df.columns:
    text_list = df[TEXT_COLUMN].astype(str) ## Sachverhalte in Variable einlesen
    print(f"\nTextdaten aus Spalte '{TEXT_COLUMN}' extrahiert (Anzahl: {len(text_list)})") ## Abgleich mit vorheriger Gesamtzahl. WICHTIG: muss übereinstimmen
    print("Beispiel für einen Text:", text_list.iloc[0]) ## Gibt ein Beispiel Sachverhalt aus
else:
    print(f"\nFEHLER: Spalte '{TEXT_COLUMN}' nicht im DataFrame gefunden. Verfügbare Spalten: {df.columns.tolist()}") ## Ausgabe, sollte Sachverhalt nicht als Spalte gefunden werden

'''
# Lade die benötigten NLTK-Ressourcen, falls noch nicht geschehen
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Ab hier haben Sie die zu analysierenden Texte in der Variable 'text_list'
'''