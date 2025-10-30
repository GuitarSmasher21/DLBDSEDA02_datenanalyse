## Python Module laden
import pandas as pd ## DataFrame
import json ## JSON Datei laden
import nltk ## Tokenisierung
import re ## Regular Expressions (Muster in Zeichenketten)
import numpy as np ## Vektorisierung
import os ## Schnittstelle für OS-Integration
from nltk.corpus import stopwords ## Liste für Stoppwörter aus NLTK
from nltk.tokenize import word_tokenize
from HanTa import HanoverTagger as ht ## Lemmartisierung über HanTa
from sklearn.feature_extraction.text import CountVectorizer ## Modul für Bag-of-Words
from sklearn.feature_extraction.text import TfidfVectorizer ## Modul für Tf-idf
import scipy.sparse # Benötigt für die Speicherung der Sparse Matrix

## -- 1. SETUP UND INITIALISIERUNG --

try: ## Versuch HanTa zu initialisieren, falls nicht Fallback auf WordNetLemmatizer
    tagger_de = ht.HanoverTagger('morphmodel_ger.pgz') ## HanTa Modell muss geladen werden
    print("HanTa Lemmatizer erfolgreich initialisiert.")
    hanta_initialisiert = True
except:
    print("FEHLER: HanTa nicht gefunden oder falsche Initialisierung. Nutze Fallback-Lemmatizer.")
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    hanta_initialisiert = False

try:
    nltk.download('popular', quiet=True) ## Beliebte NLTK Datensätze installieren (basierend auf Dokumentation)
    de_stoppwoerter = set(stopwords.words('german'))
except:
    print("NLTK Ressourcen konnt nicht geladen werden oder Stoppwörter nicht verfügbar.") ## Fehlermeldung für NLTK ausgeben
    de_stoppwoerter = set() ## Stoppwörter im Zweifel leer setzen

## -- 2. DATEN LADEN UND EXTRAHIERUNG --

try:
    ## JSON-Datei einlesen
    datei_pfad = '/Users/michi/Documents/GitHub/DLBDSEDA02_datenanalyse/unstructured_data.json'
    with open(datei_pfad, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ## Daten in Pandas DataFrame laden
    df = pd.json_normalize(data, record_path=['index']) ## "records_path" angeben, damit Spalten richtig eingelesen werden
    
    print("Daten erfolgreich geladen. Erste Zeilen des DataFrames:") ## Ausgabe DataFrame für Überprüfung
    print(df.head(5)) ## Gibt die ersten zwei Zeilen des DataFrames aus
    print(f"\nGesamtzahl der Beschwerden (Zeilen): {len(df)}") ## Genutzt für späteren Abgleich

except FileNotFoundError:
    print("FEHLER: 'daten.json' nicht gefunden. Bitte Dateipfad überprüfen.") ## Prüfung + Fehlermeldung auf falschem Dateipfad
except Exception as e:
    print(f"FEHLER beim Laden oder Parsen der Daten: {e}") ## Prüfung auf Prozessing-Fehler


## -- 3. TEXT-VORVERARBEITUNG --
## Spalte identifizieren, die die unstrukturierten Beschwerdetexte enthält.
text_spalte = 'betreff'
verarbeitete_spalte = 'verarbeiteter_text'

if not df.empty and text_spalte in df.columns:
    print(f"\nINFO: Starte Vorverarbeitung der Spalte '{text_spalte}'...")
    
    text_verarbeitung = df[text_spalte].astype(str).tolist()
    text_verarbeitung_fertig = []
    
    ## Fortschrittszähler
    gesamt_texte = len(text_verarbeitung)

    for i, text in enumerate(text_verarbeitung):
        ## 1. Kleinschreibung und Umlaute/Scharfes S bereinigen
        text.lower()
        #text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        #text = text.replace('ß', 'ss')
        
        ## 2. Entfernen von Satzzeichen, Zahlen und anderen Nicht-Buchstaben
        #text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = re.sub(r'[^a-zA-Zäöüß\s]', ' ', text)

        ## 3. Tokenisierung
        tokens = word_tokenize(text, 'german')

        ## 4. Stoppwörter entfernen und Filtern kurzer Tokens (> 2 Buchstaben)
        tokens = [word for word in tokens if word not in de_stoppwoerter and len(word) > 2]
        
        ## 5. Liste für endgültige Lemmata
        final_tokens = []

        if hanta_initialisiert: 
            ## HanTa: Aufruf NUR EINMAL pro Text. Lemma ist Index 1.
            try:
                tagged_tokens = tagger_de.tag_sent(tokens)
                final_tokens = [lemma_tag[1] for lemma_tag in tagged_tokens]
            except Exception as e:
                ## Falls HanTa bei einem bestimmten Text fehlschlägt, Tokens beibehalten und Fehler ausgeben
                print(f"WARNUNG: HanTa Fehler bei Text {i}: {e}")
                final_tokens = [word.lower() for word in tokens if word not in de_stoppwoerter and len(word) > 2]

        elif 'lemmatizer' in globals() and not hanta_initialisiert:
            ## Fallback Lemmatisierung
            final_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
        text_verarbeitung_fertig.append(final_tokens)

        ## Fortschrittsanzeige alle 1000 Texte
        if (i + 1) % 1000 == 0:
            print(f"INFO: {i + 1}/{gesamt_texte} Texte verarbeitet...")

    ## Die bereinigten Tokens zurück in den DataFrame speichern
    df[verarbeitete_spalte] = text_verarbeitung_fertig

## -- 4. FEATURE-EXTRAKTION / VEKTORISIERUNG (BAG-OF-WORDS UND TF-IDF) --

## 4.1 Vorbereitung der Daten
df['vektor_eingabe'] = df[verarbeitete_spalte].str.join(' ') ## Einzelwörter wieder in String zusammenfassen

## --- 4.2 BAG-OF-WORDS (BoW) Vektorisierung ---
print("\nINFO: Starte Bag-of-Words Vektorisierung...")

## CountVectorizer für BoW initialisieren
## Wörter, die in weniger als 5 und in mehr als 80% der Dokumente vorkommen filtern und Bigramme einbeziehen

bow_vector = CountVectorizer(
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)

## Daten anpassen und in Dokument-Begriff-Matrix überführen
bow_matrix= bow_vector.fit_transform(df['vektor_eingabe'])

## --- Ergebnis-Überprüfung BoW ---
## Prüfung Begriffe, die für Matrix-Bildung genutzt wurden 
## Beispielhafte Feature-Namen
bow_feature = bow_vector.get_feature_names_out()
print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
print(bow_feature[:10])

## Ausgabe der gebildeten BoW Matrix
print(f"INFO: Bag-of-Words (x_matrix) Feature-Matrix erstellt.")
print(f"INFO: Dimensionen der BoW-Matrix: {bow_matrix.shape[0]} Dokumente x {bow_matrix.shape[1]} Features.")

# --- 4.3 TF-IDF Vektorisierung ---
print("\nINFO: Starte TF-IDF Vektorisierung...")

# TfidfVectorizer initialisieren (gleiche Parameter wie BoW)
tfidf_vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)

# Daten fitten und transformieren
tfidf_vector = tfidf_vectorizer.fit_transform(df['vektor_eingabe'])

# --- Ergebnis-Überprüfung TF-IDF ---
## Prüfung Begriffe, die für Matrix-Bildung genutzt wurden 
## Beispielhafte Feature-Namen
tfidf_features = tfidf_vectorizer.get_feature_names_out()
print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
print(tfidf_features[:10])

## Ausgabe der gebildeten Tf-iDF Matrix
print(f"INFO: TF-IDF (tfidf_vector) Feature-Matrix erstellt.")
print(f"INFO: Dimensionen der TF-IDF-Matrix: {tfidf_vector.shape[0]} Dokumente x {tfidf_vector.shape[1]} Features.")


# 4.4 Optional: Speichern der Matrizen für weitere Schritte
# speicher_pfad_bow = 'bow_matrix.npz'
# scipy.sparse.save_npz(speicher_pfad_bow, X_bow)
# speicher_pfad_tfidf = 'tfidf_matrix.npz'
# scipy.sparse.save_npz(speicher_pfad_tfidf, tfidf_vector)
# print(f"\nINFO: Feature-Matrizen gespeichert: {speicher_pfad_bow} und {speicher_pfad_tfidf}")
