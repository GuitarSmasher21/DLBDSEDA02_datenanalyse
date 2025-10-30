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
import scipy.sparse ## Benötigt für die Speicherung der Sparse Matrix
from gensim.models.ldamodel import LdaModel ## Themen-Extraktion 1. Technik (LDA)
from gensim.models import CoherenceModel ## Modul für Coherence Model
from gensim.corpora.dictionary import Dictionary ## Modul um Wörterbuch in LDA anzulegen
import gensim.corpora as corpora
from sklearn.decomposition import NMF ## Themen-Extraktion 2. Technik (NMF)
import matplotlib.pyplot as plt ## Modul Visualisierung Coherence Score
import pyLDAvis
import pyLDAvis.gensim_models
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) ## Unterdrückt Gensim-Warnungen

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

        ## 1. Entfernen von allen Zeichen, außer Klein-/Großbuchstaben und Sonderzeichen
        text = re.sub(r'[^a-zA-Zäöüß\s]', ' ', text)

        ## 2. Tokenisierung
        tokens = word_tokenize(text, 'german')

        ## 3. Stoppwörter entfernen und Filtern kurzer Tokens (> 2 Buchstaben)
        tokens = [word for word in tokens if word not in de_stoppwoerter and len(word) > 2]
        
        ## 4. Liste für endgültige Lemmata
        final_tokens = []

        if hanta_initialisiert: 
            ## HanTa-Aufruf nur einmalig pro Text.
            try:
                tagged_tokens = tagger_de.tag_sent(tokens)
                final_tokens = [lemma_tag[1] for lemma_tag in tagged_tokens]
            except Exception as e:
                ## Falls HanTa bei einem bestimmten Text fehlschlägt, Tokens beibehalten und Fehler ausgeben
                print(f"WARNUNG: HanTa Fehler bei Text {i}: {e}")
                final_tokens = [word.lower() for word in tokens if word not in de_stoppwoerter and len(word) > 2]

        elif 'lemmatizer' in globals() and not hanta_initialisiert:
            ## Fallback auf andere Lemmatisierung
            final_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
        text_verarbeitung_fertig.append(final_tokens) ## Finale Liste speichern

        ## Fortschrittsanzeige alle 1000 Texte
        if (i + 1) % 1000 == 0:
            print(f"INFO: {i + 1}/{gesamt_texte} Zeilen verarbeitet...")

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
    ngram_range=(1, 1)
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


# --- 4.4 Vergleich der Vektorisierungs-Ergebnisse ---

if 'bow_matrix' in locals() and 'tfidf_vector' in locals():
    
    print("\n--- 4.4 Vergleich der Vektorisierungs-Ergebnisse ---")
    
    # Allgemeine Metriken (sollten gleich sein, da gleiche Parameter verwendet wurden)
    bow_shape = bow_matrix.shape
    tfidf_shape = tfidf_vector.shape
    print(f"INFO: Dimensionen: BoW {bow_shape} | Tf-idf {tfidf_shape}")
    print(f"INFO: Vokabulargröße: {len(bow_feature)} Features")
   
    # Ermitteln der Top 10 Features für BoW (basierend auf absoluter Häufigkeit)
    bow_sum = np.sum(bow_matrix, axis=0)
    bow_counts = [(bow_feature[i], bow_sum[0, i]) for i in range(bow_matrix.shape[1])]
    bow_counts.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 BoW (häufigste Vorkommen):")
    for word, count in bow_counts[:10]:
         print(f"  {word}: {int(count)}")

    # Ermitteln der Top 10 Features für Tf-idf (basierend auf der gewichteten Wichtigkeit)
    tfidf_sum = np.sum(tfidf_vector, axis=0)
    tfidf_scores = [(tfidf_features[i], tfidf_sum[0, i]) for i in range(tfidf_vector.shape[1])]
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 Tf-idf (höchste Wichtigkeit/Spezifität):")
    for word, score in tfidf_scores[:10]:
         print(f"  {word}: {score:.4f}")

else:
    print("WARNUNG: Vektorisierungsmatrizen konnten nicht für den Vergleich gefunden werden.")

## --- 5. THEMENEXTRAKTION ---
# Variable für die verarbeitete Spalte muss korrekt gesetzt sein:
verarbeitete_spalte = 'verarbeiteter_text' 

if not df.empty and verarbeitete_spalte in df.columns:
        
    # 1. Daten extrahieren: Gensim benötigt die Liste von Tokens (Liste von Listen)
    token_liste = df[verarbeitete_spalte].tolist()
    
    # 2. Fehlerprüfung: Stellen wir sicher, dass die Listen nicht leer sind
    if not any(token_liste):
        print("FEHLER: Die Liste der verarbeiteten Texte ('token_liste') ist leer. Bitte Vorverarbeitung prüfen.")
    else:
        try:
            # 3. Erstellen des Wörterbuchs (Dictionary)
            # Dies bildet jedes einzigartige Wort auf eine ID ab.
            id2word = Dictionary(token_liste)
            
            # 4. Filtern des Wörterbuchs
            # Entferne Wörter, die in weniger als 5 Dokumenten (no_below=5) oder 
            # in mehr als 80% der Dokumente (no_above=0.8) vorkommen.
            id2word.filter_extremes(no_below=5, no_above=0.8) 
            
            # 5. Erstellen des Korpus (Bag-of-Words-Darstellung)
            # Konvertiert die Texte in das Gensim-Format (Liste von Tupeln (Wort-ID, Zähler))
            corpus = [id2word.doc2bow(text) for text in token_liste]
            
            print(f"INFO: Dictionary erstellt: {len(id2word)} einzigartige Token nach Filterung.")
            print(f"INFO: Gensim Corpus erstellt: {len(corpus)} Dokumente.")
            print("INFO: Schritt 5.1 erfolgreich abgeschlossen.")

        except Exception as e:
            print(f"KRITISCHER FEHLER in Gensim-Vorbereitung (Dictionary/Corpus): {e}")
            print("Überprüfen Sie, ob Ihre 'token_liste'-Liste Listen von Strings enthält.")
            
else:
    print("ABBRUCH: DataFrame ist leer oder Spalte 'verarbeiteter_text' fehlt.")

# --- 5.2 PRAGMATISCHE FESTLEGUNG DER THEMENANZAHL (K) auf maximal 20 Themen ---
print("\n--- 5.2 Bestimmung der optimalen Themenanzahl ---")
opt_k = 10
print(f"INFO: Themenanzahl (K) manuell auf {opt_k} festgelegt.")


# --- 5.3 LDA MODELLIERUNG (Technik 1) ---
print(f"\n--- 5.3 LDA Modellierung (Finales Modell mit K={opt_k}) ---")

# Wir prüfen, ob die Gensim-Variablen aus 5.1 existieren
if 'corpus' in locals() and 'id2word' in locals() and 'token_liste' in locals():
    try:
        final_lda_model = LdaModel(
            corpus=corpus, 
            id2word=id2word, 
            num_topics=opt_k, # Verwendet das K aus Schritt 5.2
            random_state=42, 
            passes=15, 
            alpha='auto'
        )

        # Top Themen ausgeben
        print(f"INFO: Top 10 Wörter pro Thema (LDA):")
        lda_topics = final_lda_model.print_topics(num_words=10)
        for idx, topic in lda_topics:
            print(f"  Thema #{idx}: {topic}")

    except Exception as e:
        print(f"FEHLER: LDA-Training fehlgeschlagen: {e}")
else:
    print("FEHLER: Gensim-Variablen (corpus, id2word, token_liste) fehlen. Abschnitt 5.1 prüfen.")

# --- 5.3.1 INTERAKTIVE VISUALISIERUNG (pyLDAvis) ---

print(f"\n--- 5.3.1 Interaktive LDA-Visualisierung (pyLDAvis) ---")

# Prüfen, ob die benötigten Gensim-Objekte vorhanden sind
if 'final_lda_model' in locals() and 'corpus' in locals() and 'id2word' in locals():
    try:
        # 1. Daten für pyLDAvis vorbereiten
        # HINWEIS: 'mmap=' re' in neueren Versionen von pyLDAvis empfohlen
        prepared_data = pyLDAvis.gensim_models.prepare(
            final_lda_model, 
            corpus, 
            id2word, 
            mds='mmds' # Standard-Methode zur Dimensionsreduktion
        )
        
        # 2. Visualisierung als HTML-Datei speichern
        output_filepath = 'lda_visualisierung.html'
        pyLDAvis.save_html(prepared_data, output_filepath)
        
        print(f"INFO: Interaktive Visualisierung erfolgreich gespeichert: {output_filepath}")
        print("INFO: Öffnen Sie diese HTML-Datei in Ihrem Browser, um die Themen zu explorieren.")

    except Exception as e:
        print(f"FEHLER: pyLDAvis-Visualisierung fehlgeschlagen: {e}")
        print("Stellen Sie sicher, dass die Bibliothek 'pyLDAvis' korrekt installiert ist (pip install pyLDAvis).")
else:
    print("FEHLER: Benötigte Objekte (final_lda_model, corpus, id2word) für pyLDAvis nicht gefunden.")

# --- 5.4 NMF MODELLIERUNG (Technik 2: Non-Negative Matrix Factorization) ---
print(f"\n--- 5.4 NMF Modellierung (Finales Modell mit K={opt_k}) ---")

# NMF benötigt die Scikit-learn Tf-idf-Matrix (tfidf_vector)
if 'tfidf_vector' in locals() and 'tfidf_features' in locals():
    try:
        nmf_model = NMF(
            n_components=opt_k, # Verwendet das gleiche K wie LDA
            random_state=42,
            init='nndsvda',
            max_iter=1000
        )
        nmf_model.fit(tfidf_vector)

        # Funktion zur Ausgabe der Top-Wörter für NMF
        def display_topics(model, feature_names, no_top_words):
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
                topics.append(f"  Thema #{topic_idx}: {top_words}")
            return topics

        # Top Themen ausgeben
        nmf_topics = display_topics(nmf_model, tfidf_features, 10)
        print(f"INFO: Top 10 Wörter pro Thema (NMF):")
        for topic in nmf_topics:
            print(topic)

    except Exception as e:
        print(f"FEHLER: NMF-Training fehlgeschlagen: {e}")
else:
    print("FEHLER: Tf-idf-Matrizen (tfidf_vector, tfidf_features) fehlen. Schritt 4 prüfen.")

# --- 6. VISUALISIERUNG (Matplotlib) & DISKUSSION ---

print(f"\n--- 6. Finale Visualisierung (Matplotlib) & Diskussion ---")

# Prüfen, ob das finale LDA-Modell existiert
if 'final_lda_model' in locals():
    try:
        # Wähle ein interessantes Thema zum Visualisieren. 
        # Basierend auf Ihrem Output ist Thema #4 das "Abfall/Müll"-Thema.
        topic_id = 4 
        topic_terms = final_lda_model.show_topic(topic_id, topn=10)
        
        # Daten für den Matplotlib-Plot vorbereiten
        words = [term[0] for term in topic_terms]
        probabilities = [term[1] for term in topic_terms]
        
        # Plot erstellen (horizontales Balkendiagramm)
        plt.figure(figsize=(10, 7))
        plt.barh(words, probabilities, color='steelblue') 
        plt.xlabel("Wichtigkeit (Wahrscheinlichkeit) im Thema")
        plt.ylabel("Top-Wörter")
        plt.title(f"Matplotlib: Detaillierte Ansicht für LDA Thema #{topic_id} (Müll/Abfall)")
        plt.gca().invert_yaxis() # Das wichtigste Wort oben anzeigen
        plt.tight_layout()
        
        # Plot speichern
        plt.savefig('lda_topic_barchart.png')
        print("INFO: Matplotlib-Visualisierung 'lda_topic_barchart.png' gespeichert.")

    except Exception as e:
        print(f"FEHLER bei Matplotlib-Visualisierung von Thema #{topic_id}: {e}")
else:
    print("INFO: Finales LDA-Modell nicht gefunden, überspringe Matplotlib-Balkendiagramm.")

# --- 6. VISUALISIERUNG (Matplotlib) & DISKUSSION ---

print(f"\n--- 6. Finale Visualisierung (Matplotlib) & Diskussion ---")

# Prüfen, ob das finale LDA-Modell existiert
if 'final_lda_model' in locals():
    try:
        # Wähle ein interessantes Thema zum Visualisieren. 
        # Basierend auf Ihrem Output ist Thema #4 das "Abfall/Müll"-Thema.
        topic_id = 4 
        topic_terms = final_lda_model.show_topic(topic_id, topn=10)
        
        # Daten für den Matplotlib-Plot vorbereiten
        words = [term[0] for term in topic_terms]
        probabilities = [term[1] for term in topic_terms]
        
        # Plot erstellen (horizontales Balkendiagramm)
        plt.figure(figsize=(10, 7))
        plt.barh(words, probabilities, color='steelblue') 
        plt.xlabel("Wichtigkeit (Wahrscheinlichkeit) im Thema")
        plt.ylabel("Top-Wörter")
        plt.title(f"Matplotlib: Detaillierte Ansicht für LDA Thema #{topic_id} (Müll/Abfall)")
        plt.gca().invert_yaxis() # Das wichtigste Wort oben anzeigen
        plt.tight_layout()
        
        # Plot speichern
        plt.savefig('lda_topic_barchart.png')
        print("INFO: Matplotlib-Visualisierung 'lda_topic_barchart.png' gespeichert.")

    except Exception as e:
        print(f"FEHLER bei Matplotlib-Visualisierung von Thema #{topic_id}: {e}")
else:
    print("INFO: Finales LDA-Modell nicht gefunden, überspringe Matplotlib-Balkendiagramm.")

# --- 6. VISUALISIERUNG (Matplotlib) & DISKUSSION ---

print(f"\n--- 6. Finale Visualisierung (Matplotlib) & Diskussion ---")

# Prüfen, ob das finale LDA-Modell existiert
if 'final_lda_model' in locals():
    try:
        # Wähle ein interessantes Thema zum Visualisieren. 
        # Basierend auf Ihrem Output ist Thema #4 das "Abfall/Müll"-Thema.
        topic_id = 4 
        topic_terms = final_lda_model.show_topic(topic_id, topn=10)
        
        # Daten für den Matplotlib-Plot vorbereiten
        words = [term[0] for term in topic_terms]
        probabilities = [term[1] for term in topic_terms]
        
        # Plot erstellen (horizontales Balkendiagramm)
        plt.figure(figsize=(10, 7))
        plt.barh(words, probabilities, color='steelblue') 
        plt.xlabel("Wichtigkeit (Wahrscheinlichkeit) im Thema")
        plt.ylabel("Top-Wörter")
        plt.title(f"Matplotlib: Detaillierte Ansicht für LDA Thema #{topic_id} (Müll/Abfall)")
        plt.gca().invert_yaxis() # Das wichtigste Wort oben anzeigen
        plt.tight_layout()
        
        # Plot speichern
        plt.savefig('lda_topic_barchart.png')
        print("INFO: Matplotlib-Visualisierung 'lda_topic_barchart.png' gespeichert.")

    except Exception as e:
        print(f"FEHLER bei Matplotlib-Visualisierung von Thema #{topic_id}: {e}")
else:
    print("INFO: Finales LDA-Modell nicht gefunden, überspringe Matplotlib-Balkendiagramm.")

# --- Diskussion für Ihr PDF (Zusammenfassung der Phase 2) ---
print("\n**Diskussion der Erarbeitungsphase (für das PDF):**")
print("1. **Vorgehen:** Daten geladen, Textvorverarbeitung (Regex, Tokenisierung, HanTa-Lemmatisierung) durchgeführt.")
print("2. **Vektorisierung (Vergleich):** BoW (Input für LDA) und Tf-idf (Input für NMF) wurden verglichen. Tf-idf (Unigramme) lieferte interpretierbarere Features für NMF.")
print("3. **Themenanzahl (K):** K wurde pragmatisch auf 10 festgelegt, da die automatisierte Kohärenz-Berechnung bei ~70.000 Dokumenten zu Speicher-Abstürzen (RAM) führte.")
print("4. **Modellierung (Vergleich):** LDA (Technik 1) und NMF (Technik 2) wurden mit K=10 trainiert.")
print("5. **Visualisierung:** Die LDA-Ergebnisse wurden interaktiv mit pyLDAvis ('lda_visualisierung.html') und exemplarisch mit Matplotlib ('lda_topic_barchart.png') visualisiert.")
print("6. **Ergebnisse:** Die Modelle identifizierten klare Beschwerdethemen (z.B. 'Abfall/Müll', 'Defekte Verkehrszeichen', 'Straßenschäden').")
print("\nDas Skript und die Ergebnisse sind bereit für die Finalisierungsphase.")