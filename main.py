# --- Moduldefinition ---
# Module für Schritt 1
import nltk # Tokenisierung
from nltk.corpus import stopwords # Liste für Stoppwörter aus NLTK
from HanTa import HanoverTagger as ht # Lemmartisierung über HanTa
from nltk.stem import WordNetLemmatizer # Alternative Lemmatisierung

# Module für Schritt 2

import pandas as pd # DataFrame
import json # JSON Datei laden

# Module für Schritt 3
import re # Regular Expressions (Muster in Zeichenketten)
from nltk.tokenize import word_tokenize # Modul für Tokenisierung

# Module für Schritt 4
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Module für Bag-of-Words + Tf-idf
import numpy as np # Vektorisierung
# Module für Schritt 5
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.decomposition import NMF
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models

# Module für Schritt 6
import matplotlib.pyplot as plt # Modul Visualisierung Coherence Score

# --- Funktionsdefinition ---
# Funktionsdefinition Schritt 1 - Initialisierung
def schritt_1_ressourcen_initialisieren():
    # HanTa-Tagger initialisieren und NLTK-Ressourcen laden
    try: 
        tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')  # HanTa-Modell laden
        print("HanTa Lemmatizer erfolgreich initialisiert.")
        hanta_initialisiert = True
    except:
        print("FEHLER: HanTa nicht gefunden oder falsche Initialisierung. Nutze Fallback-Lemmatizer.")
        lemmatizer = WordNetLemmatizer()
        tagger_de = lemmatizer # Alternative Lemmatisierung
        hanta_initialisiert = False
    try:
        nltk.download('popular', quiet=True) # Beliebte NLTK Datensätze installieren (basierend auf Dokumentation)
        de_stoppwoerter = set(stopwords.words('german')) # Deutsche Stoppwörter "füllen"
    except:
        print("NLTK Ressourcen konnt nicht geladen werden oder Stoppwörter nicht verfügbar.") # Fehlermeldung für NLTK ausgeben
        de_stoppwoerter = set() # Stoppwörter im Zweifel leer setzen

    return tagger_de, hanta_initialisiert, de_stoppwoerter # Ergebnisse an "main.py" zurückgeben

# Funktionsdefinition Schritt 2 - Daten laden
def schritt_2_daten_laden(datei_pfad, record_path):
    # Lädt und normalisiert die JSON-Daten der Berliner Ordnungsamt-Datei in einen DataFrame
    try:
        with open(datei_pfad, 'r', encoding='utf-8') as f:  # Dateipfad in main.py angegeben. Datei sollte lokal aufgerufen werden
            data = json.load(f)
        df = pd.json_normalize(data, record_path=record_path) # # "records_path" angeben, damit Spalten richtig eingelesen werden
        print("Daten erfolgreich geladen. Erste Zeilen des DataFrames:")
        print(df.head(5)) # Ausgabe erste 5 Zeilen von DataFrame für Überprüfung
        print(f"\nGesamtzahl der Beschwerden (Zeilen): {len(df)}") 
        return df # Genutzt für späteren Abgleich der Korrektheit
    except FileNotFoundError:
        print("FEHLER: 'daten.json' nicht gefunden. Bitte Dateipfad überprüfen.") 
        return pd.DataFrame() # Prüfung + Fehlermeldung auf falschem Dateipfad
    except Exception as e:
        print(f"FEHLER beim Laden oder Parsen der Daten: {e}") 
        return pd.DataFrame() # Prüfung auf Prozessing-Fehler

# Funktionsdefintion Schritt 3 - Text-Vorverarbeitung
def schritt_3_text_vorverarbeitung(df, text_spalte, verarbeitete_spalte, tagger_de, hanta_initialisiert, de_stoppwoerter):
    # Gesamte Text-Vorverarbeitungen (Regex, Tokenisierung, Lemmatisierung) ausgeführt
    if not (text_spalte in df.columns):
        print(f"FEHLER: Textspalte '{text_spalte}' nicht im DataFrame gefunden.")
        return df # Ausgabe falscher Initial-Spalte
    print(f"\nINFO: Starte Vorverarbeitung der Spalte '{text_spalte}'...") # Spalte in "main.py" definiert. In unserem Fall muss das "Beschwerden" sein
    text_verarbeitung = df[text_spalte].astype(str).tolist() # Umwandlung des DataFrames in eine Liste
    text_verarbeitung_fertig = [] # Platzhalter für finale Version
    gesamt_texte = len(text_verarbeitung) # Zähler für Verarbeitungs-Fortschritt
    for i, text in enumerate(text_verarbeitung): 
        text = text.lower() # Kleinschreibung erzwingen
        text = re.sub(r'[^a-zA-Zäöüß\s]', ' ', text) # Entfernen von allen Zeichen, außer Klein-/Großbuchstaben und Sonderzeichen
        tokens = word_tokenize(text, 'german') # Tokenisierung mit deutschem Wortschatz
        tokens = [word for word in tokens if word not in de_stoppwoerter and len(word) > 2] # Token prüfen und Wörter die kleiner als 2 Zeichen sind raus nehmen  
        final_tokens = [] # Platzhalter für finale Version.
        if hanta_initialisiert:
            try:
                tagged_tokens = tagger_de.tag_sent(tokens) # Token in HanTa laden
                final_tokens = [lemma_tag[1] for lemma_tag in tagged_tokens] # HanTa für Index = 1 (das eigentliche Wort) durchlaufen
            except Exception as e:
                print(f"WARNUNG: HanTa Fehler bei Text {i}: {e}")  # Fehlerausgabe
                final_tokens = [word.lower() for word in tokens if word not in de_stoppwoerter and len(word) > 2] # Weitere Behandlung ohne HanTa
        elif not hanta_initialisiert:
            try:
                final_tokens = [tagger_de.lemmatize(word) for word in tokens] # Alternative Lemmartizer genutzt (anstelle HanTa)
            except Exception as e:
                 print(f"WARNUNG: Fallback-Lemmatizer fehlgeschlagen: {e}") # Fehlerausgabe für alternativen Lemmartizer
                 final_tokens = tokens # Rückgabe ohne Tokenisierung + Lemmartisierung
        text_verarbeitung_fertig.append(final_tokens) # Rückgabe finale Version
        if (i + 1) % 1000 == 0: 
            print(f"INFO: {i + 1}/{gesamt_texte} Zeilen verarbeitet...") # Fortschritts-Zähler
    df[verarbeitete_spalte] = text_verarbeitung_fertig # Fertige Daten in DataFrame zurück schreiben
    df['vektor_eingabe'] = df[verarbeitete_spalte].str.join(' ') # Einzelwörter wieder in einen String zusammenfassen für weitere Verarbeitung
    return df # Rückgabe an "main.py"

# Funktionsdefintion Schritt 4 - Vektorisierung
def schritt_4_vektorisierung(df, vektor_spalte):
    # Auführung von BoW- und Tf-idf-Vektorisierung. Zudem ein Abgleich zwischen beiden Varianten
    text_data = df[vektor_spalte]
    # --- 4.2 BoW ---
    print("\nINFO: Starte Bag-of-Words Vektorisierung...")
    bow_vector = CountVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2)) # CountVectorizer für BoW initialisieren für Wörter, die in weniger als 5 und in mehr als 80% der Dokumente vorkommen und Bigramme sind
    bow_matrix = bow_vector.fit_transform(text_data) # Daten anpassen und in Dokument-Begriff-Matrix überführen
    bow_feature = bow_vector.get_feature_names_out() 
    print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
    print(bow_feature[:10]) # Überprüfung der Begriffe, die für Matrix-Bildung genutzt wurden. Gibt erste 10 Themen aus
    print(f"INFO: Bag-of-Words Feature-Matrix erstellt.")
    print(f"INFO: Dimensionen der BoW-Matrix: {bow_matrix.shape[0]} Dokumente x {bow_matrix.shape[1]} Features.")
    # --- 4.3 Tf-idf ---
    print("\nINFO: Starte TF-IDF Vektorisierung...")
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2)) # TfidfVectorizer initialisieren (gleiche Parameter wie BoW)
    tfidf_vector = tfidf_vectorizer.fit_transform(text_data)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
    print(tfidf_features[:10]) # Überprüfung der Begriffe, die für Matrix-Bildung genutzt wurden. Gibt erste 10 Themen aus
    print(f"INFO: TF-IDF Feature-Matrix erstellt.")
    print(f"INFO: Dimensionen der TF-IDF-Matrix: {tfidf_vector.shape[0]} Dokumente x {tfidf_vector.shape[1]} Features.")    
    # --- 4.4 Vergleich ---
    vergleich_vektoren(bow_matrix, bow_feature, tfidf_vector, tfidf_features) # Funktions-Aufruf
    return bow_matrix, bow_feature, tfidf_vector, tfidf_features # Rückgabe an "main.py"

# Funktionsdefintion Schritt 4 - Hilfsfunktion zur Ausgabe
def vergleich_vektoren(bow_matrix, bow_feature, tfidf_vector, tfidf_features): 
    bow_shape = bow_matrix.shape
    tfidf_shape = tfidf_vector.shape
    print(f"INFO: Dimensionen: BoW {bow_shape} | Tf-idf {tfidf_shape}") # Definition allgemeiner Metriken (sollten gleich sein, da gleiche Parameter verwendet wurden)
    print(f"INFO: Vokabulargröße: {len(bow_feature)} Features") # Ausgabe Vokabulargröße
    # Ermittlung der Top 10 Features für BoW (basierend auf absoluter Häufigkeit)
    bow_sum = np.sum(bow_matrix, axis=0)
    bow_counts = [(bow_feature[i], bow_sum[0, i]) for i in range(bow_matrix.shape[1])]
    bow_counts.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 BoW (häufigste Vorkommen):") # Ausgabe der gebildeten Matrix für Vergleich
    for word, count in bow_counts[:10]:
         print(f"  {word}: {int(count)}")
    # Ermittlung der Top 10 Features für Tf-idf (basierend auf der gewichteten Wichtigkeit)
    tfidf_sum = np.sum(tfidf_vector, axis=0)
    tfidf_scores = [(tfidf_features[i], tfidf_sum[0, i]) for i in range(tfidf_vector.shape[1])]
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 Tf-idf (höchste Wichtigkeit/Spezifität):")
    for word, score in tfidf_scores[:10]:
         print(f"  {word}: {score:.4f}")

# Funktionsdefintion Schritt 5 - Themenmodellierung
def schritt_5_themenmodellierung(df, verarbeitete_spalte, opt_k, tfidf_vector, tfidf_features):
    # Ausführung Themenmodellierung (Gensim-Vorbereitung, LDA vs. NMF und pyLDAvis)
    # Wörterbuch und Korpus in Gensim anlegen
    id2word, corpus, token_liste = vorbereitung_gensim(df, verarbeitete_spalte)
    final_lda_model = None
    if id2word and corpus: # Prüfung wenn Wörterbuch und Korpus gesetzt.
        # LDA Modellierung
        final_lda_model = lda_model_training(corpus, id2word, token_liste, opt_k)
        # pyLDAvis Visualisierung
        if final_lda_model:
            lda_visualisieren(final_lda_model, corpus, id2word)
    # NMF Modellierung
    nmf_model = nmf_model_training(tfidf_vector, tfidf_features, opt_k)
    return id2word, corpus, token_liste, final_lda_model, nmf_model # Rückgabe an "main.py"
# Funktionsdefintion Schritt 5.1 - Vorbereitung für Themenmodellierung
def vorbereitung_gensim(df, verarbeitete_spalte):
    if not df.empty and verarbeitete_spalte in df.columns:
        token_liste = df[verarbeitete_spalte].tolist() # Daten extrahieren: Gensim benötigt die Liste von Tokens (Liste von Listen)
        if not any(token_liste):
            print("FEHLER: Die Liste der verarbeiteten Texte ('token_liste') ist leer.")
            return None, None, None
        try:
            id2word = Dictionary(token_liste) # Erstellen des Wörterbuchs (Dictionary). Jedes Wort hat eine ID
            id2word.filter_extremes(no_below=5, no_above=0.8) # Wörterbuchs filtern. Entfernt Wörter, die in weniger als 5 Dokumenten (no_below=5) oder in mehr als 80% der Dokumente (no_above=0.8) vorkommen
            corpus = [id2word.doc2bow(text) for text in token_liste] # Erstellt Korpus (Bag-of-Words-Darstellung) und konvertiert die Texte in das Gensim-Format (Liste von Tupeln (Wort-ID, Zähler))
            print(f"INFO: Dictionary erstellt: {len(id2word)} einzigartige Token nach Filterung.") # Ausgabe zur Überprüfung
            print(f"INFO: Gensim Corpus erstellt: {len(corpus)} Dokumente.") # Ausgabe zur Überprüfung
            print("INFO: Schritt erfolgreich abgeschlossen.") # Ausgabe für Fortschritt
            return id2word, corpus, token_liste
        except Exception as e:
            print(f"KRITISCHER FEHLER in Gensim-Vorbereitung: {e}")
            return None, None, None
    else:
        print("ABBRUCH: DataFrame ist leer oder Spalte 'verarbeiteter_text' fehlt.")
        return None, None, None
# Funktionsdefintion Schritt 5.2 - Training LDA-Datenmodell
def lda_model_training(corpus, id2word, token_liste, opt_k):
    print(f"\n LDA Modellierung (Finales Modell mit K={opt_k})") 
    
    # Prüfen ob die Variablen existieren
    if corpus and id2word and token_liste:
        try:
            # 1. Modell trainieren
            final_lda_model = LdaModel(
                corpus=corpus, 
                id2word=id2word, 
                num_topics=opt_k, 
                random_state=42, 
                passes=15, 
                alpha='auto'
            ) 
            
            # 2. Themen ausgeben
            print(f"INFO: Top 10 Wörter pro Thema (LDA):")
            lda_topics = final_lda_model.print_topics(num_words=10)
            for idx, topic in lda_topics:
                print(f"  Thema #{idx}: {topic}") 
            
            # 3. Coherence Score Berechnung (Nutzt den globalen Import 'CoherenceModel')
            print("\nINFO: Berechne Coherence Score (Qualitätsprüfung)...")
            try:
                coherence_model_lda = CoherenceModel(
                    model=final_lda_model, 
                    texts=token_liste, 
                    dictionary=id2word, 
                    coherence='c_v'
                )
                coherence_lda = coherence_model_lda.get_coherence()
                print(f"ERGEBNIS: Der Coherence Score (C_v) für K={opt_k} beträgt: {coherence_lda:.4f}")
            except Exception as e:
                print(f"WARNUNG: Coherence konnte nicht berechnet werden: {e}")

            return final_lda_model

        except Exception as e:
            print(f"FEHLER: LDA-Training fehlgeschlagen: {e}")
    else:
        print("FEHLER: Gensim-Variablen fehlen. Abschnitt 5.1 prüfen.")
    
    return None
# Funktionsdefintion Schritt 5.3 - Vorbereitung Visualisierung
def lda_visualisieren(model, corpus, id2word, filepath='lda_visualisierung.html'):
    try: # Daten für pyLDAvis vorbereiten
        prepared_data = pyLDAvis.gensim_models.prepare(
            model, 
            corpus, 
            id2word, 
            mds='mmds'  # mmds = Standard-Methode zur Dimensionsreduktion
        )
        pyLDAvis.save_html(prepared_data, filepath) # Visualisierung als HTML-Datei speichern
        print(f"INFO: Interaktive Visualisierung erfolgreich gespeichert: {filepath}")
        print("INFO: Öffnen Sie diese HTML-Datei in Ihrem Browser, um die Themen zu explorieren.")
    except Exception as e:
        print(f"FEHLER: pyLDAvis-Visualisierung fehlgeschlagen: {e}")
        print("Stellen Sie sicher, dass die Bibliothek 'pyLDAvis' korrekt installiert ist (pip install pyLDAvis).")
# Funktionsdefintion Schritt 5.4 - Training NMF-Datenmodell
def nmf_model_training(tfidf_vector, tfidf_features, opt_k):
    print(f"\n NMF Modellierung (Finales Modell mit K={opt_k})") # Ausgabe der gesetzen Themen (manuell festgelegt)
    if 'tfidf_vector' in locals() and 'tfidf_features' in locals(): # NMF benötigt die Scikit-learn Tf-idf-Matrix (tfidf_vector)
        try:
            nmf_model = NMF(n_components=opt_k, random_state=42, init='nndsvda', max_iter=1000)
            nmf_model.fit(tfidf_vector)
            # Funktionsdefinition zur Ausgabe der Top-Wörter für NMF
            def display_topics(model, feature_names, no_top_words): 
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
                    topics.append(f"  Thema #{topic_idx}: {top_words}") 
                return topics
            # Funktionsaufruf zur Ausgabe der Top-Wörter für NMF
            nmf_topics = display_topics(nmf_model, tfidf_features, 10)
            print(f"INFO: Top 10 Wörter pro Thema (NMF):") # Ausgabe der 10 Themen
            for topic in nmf_topics:
                print(topic)
            return nmf_model
        except Exception as e:
            print(f"FEHLER: NMF-Training fehlgeschlagen: {e}")
    else:
        print("FEHLER: Tf-idf-Matrizen fehlen. Schritt 4 prüfen.")
    return None

# Funktionsdefintion Schritt 6 - Finale Visualisierung
def schritt_6_finale_visualisierung(model, topic_id=4, filepath='lda_themen_diagramm.png'):
    # Erstellung und Speicherung eines Matplotlib-Balkendiagramm
    print(f"\n Finale Visualisierung (Matplotlib)")
    if model: # Prüfen, ob das finale LDA-Modell existiert
        try:
            topic_terms = model.show_topic(topic_id, topn=10) # Wahl eines interessanten Themas zum Visualisieren
            words = [term[0] for term in topic_terms]  # Daten für den Matplotlib-Plot vorbereiten
            probabilities = [term[1] for term in topic_terms] 
            # Plot erstellen (horizontales Balkendiagramm)
            plt.figure(figsize=(10, 7))
            plt.barh(words, probabilities, color='steelblue') 
            plt.xlabel("Wichtigkeit (Wahrscheinlichkeit) im Thema")
            plt.ylabel("Top-Wörter")
            plt.title(f"Matplotlib: Detaillierte Ansicht für LDA Thema #{topic_id}")
            plt.gca().invert_yaxis() # Wichtigste Wort oben anzeigen
            plt.tight_layout()
            plt.savefig(filepath)
            print(f"INFO: Matplotlib-Visualisierung '{filepath}' gespeichert.")
        except Exception as e:
            print(f"FEHLER bei Matplotlib-Visualisierung von Thema #{topic_id}: {e}")
    else:
        print("INFO: Finales LDA-Modell nicht gefunden, überspringe Matplotlib-Balkendiagramm.")

# Hauptfunktion
def main():
    # --- 1. KONFIGURATION & INITIALISIERUNG ---
    # Konstanten und Pfade
    DATEI_PFAD = '/Users/michi/VisualStudio/Data_Analysis/DLBDSEDA02_datenanalyse/unstructured_data.json'
    JSON_INDEX = 'index'
    JSON_SPALTE = 'betreff'
    VERARBEITETE_SPALTE = 'verarbeiteter_text'
    OPT_K = 10 

    # NLP-Ressourcen initialisieren
    tagger_de, hanta_initialisiert, de_stoppwoerter = schritt_1_ressourcen_initialisieren()

    # --- 2. DATEN LADEN ---
    df = schritt_2_daten_laden(DATEI_PFAD, JSON_INDEX)
    if df.empty:
        print("FEHLER: Daten konnten nicht geladen werden. Workflow wird abgebrochen.")
        return

    # --- 3. TEXT-VORVERARBEITUNG ---
    df = schritt_3_text_vorverarbeitung(
        df, 
        JSON_SPALTE, 
        VERARBEITETE_SPALTE, 
        tagger_de, 
        hanta_initialisiert, 
        de_stoppwoerter
    )

    # --- 4. VEKTORISIERUNG & VERGLEICH ---
    bow_matrix, bow_feature, tfidf_vector, tfidf_features = schritt_4_vektorisierung(
        df, 'vektor_eingabe'
    )

    # --- 5. THEMENMODELLIERUNG ---
    (id2word, corpus, token_liste, 
     final_lda_model, nmf_model) = schritt_5_themenmodellierung(
        df, VERARBEITETE_SPALTE, OPT_K, 
        tfidf_vector, tfidf_features
    )

    # --- 6. FINALE VISUALISIERUNG ---
    if final_lda_model:
        schritt_6_finale_visualisierung(final_lda_model, topic_id=4)
    print("\n--- Workflow erfolgreich abgeschlossen. ---")

if __name__ == "__main__": # Ablauf nur über "main.py" ausgeführt
    main()