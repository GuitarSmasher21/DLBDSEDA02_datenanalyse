## Python Module laden
import pandas as pd ## DataFrame
import json ## JSON Datei laden
import nltk ## Tokenisierung
import re ## Regular Expressions (Muster in Zeichenketten)
from nltk.corpus import stopwords ## Liste für Stoppwörter aus NLTK
from HanTa import HanoverTagger as ht ## Lemmartisierung über HanTa

try: ## Versuch HanTa zu initialisieren, falls nicht Fallback auf WordNetLemmatizer
    tagger_de = ht.HanoverTagger('morphmodel_ger.pgz') ## HanTa Modell muss geladen werden
    print("HanTa Lemmatizer erfolgreich initialisiert mit Pfad.")
except:
    print("FEHLER: HanTa nicht gefunden oder falsche Initialisierung. Nutze Fallback-Lemmatizer.")
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

## Python Script
## 1. Schritt
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
    print(f"FEHLER beim Laden oder Parsen der Daten: {e}") ## Prüfung auf Prozessing-Fehlecd s
## 2. Schritt
## Spalte identifizieren, die die unstrukturierten Beschwerdetexte enthält.
TEXT_COLUMN = 'sachverhalt' ##  Im Fall Berlin wird 'Sachverhalt' gewählt.

if TEXT_COLUMN in df.columns:
    text_list = df[TEXT_COLUMN].astype(str) ## Sachverhalte in Variable einlesen
    print(f"\nTextdaten aus Spalte '{TEXT_COLUMN}' extrahiert (Anzahl: {len(text_list)})") ## Abgleich mit vorheriger Gesamtzahl. WICHTIG: muss übereinstimmen
    print("Beispiel für einen Text:", text_list.iloc[0]) ## Gibt ein Beispiel Sachverhalt aus
else:
    print(f"\nFEHLER: Spalte '{TEXT_COLUMN}' nicht im DataFrame gefunden. Verfügbare Spalten: {df.columns.tolist()}") ## Ausgabe, sollte Sachverhalt nicht als Spalte gefunden werden

try:
    nltk.download('popular') ## Beliebte NLTK Datensätze installieren (basierend auf Dokumentation)
except:
    print("NLTK Ressourcen konnt nicht geladen werden.") ## Fehlermeldung für NLTK ausgeben

# Ab hier haben Sie die zu analysierenden Texte in der Variable 'text_list'