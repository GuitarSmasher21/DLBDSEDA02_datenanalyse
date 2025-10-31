## Hauptskript zur Steuerung des Datenanalyse-Workflows

# Importiert die einzelnen Schritte aus dem 'scripts'-Ordner
from scripts import schritt_1_ressourcen_initialisieren
from scripts import schritt_2_daten_laden
from scripts import schritt_3_text_vorverarbeitung
from scripts import schritt_4_vektorisierung
from scripts import schritt_5_themenmodilierung
from scripts import schritt_6_visualisierung

import pandas as pd

def main():

    # --- 1. KONFIGURATION & INITIALISIERUNG ---
    # Konstanten und Pfade
    DATEI_PFAD = '/Users/michi/Documents/GitHub/DLBDSEDA02_datenanalyse/unstructured_data.json'
    JSON_INDEX = 'index'
    JSON_SPALTE = 'betreff'
    VERARBEITETE_SPALTE = 'verarbeiteter_text'
    OPT_K = 10 

    # NLP-Ressourcen initialisieren
    tagger_de, hanta_initialisiert, de_stoppwoerter = schritt_1_ressourcen_initialisieren.run()

    # --- 2. DATEN LADEN ---
    df = schritt_2_daten_laden.run(DATEI_PFAD, JSON_INDEX)

    if df.empty:
        print("FEHLER: Daten konnten nicht geladen werden. Workflow wird abgebrochen.")
        return

    # --- 3. TEXT-VORVERARBEITUNG ---
    df = schritt_3_text_vorverarbeitung.run(
        df, 
        JSON_SPALTE, 
        VERARBEITETE_SPALTE, 
        tagger_de, 
        hanta_initialisiert, 
        de_stoppwoerter
    )

    # --- 4. VEKTORISIERUNG & VERGLEICH ---
    bow_matrix, bow_feature, tfidf_vector, tfidf_features = schritt_4_vektorisierung.run(
        df, 'vektor_eingabe'
    )

    # --- 5. THEMENMODELLIERUNG ---
    (id2word, corpus, token_liste, 
     final_lda_model, nmf_model) = schritt_5_themenmodilierung.run(
        df, VERARBEITETE_SPALTE, OPT_K, 
        tfidf_vector, tfidf_features
    )

    # --- 6. FINALE VISUALISIERUNG ---
    if final_lda_model:
        schritt_6_visualisierung.run(final_lda_model, topic_id=4)

    print("\n--- Workflow erfolgreich abgeschlossen. ---")

if __name__ == "__main__": # Ablauf nur über "main.py" ausgeführt
    main()