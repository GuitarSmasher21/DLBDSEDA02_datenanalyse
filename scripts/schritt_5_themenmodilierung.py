## script/schritt_5_themenmodilierung.py
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.decomposition import NMF
import pyLDAvis
import pyLDAvis.gensim_models

def run(df, verarbeitete_spalte, opt_k, tfidf_vector, tfidf_features):
    # Ausführung Themenmodellierung (Gensim-Vorbereitung, LDA vs. NMF und pyLDAvis).
    
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

def vorbereitung_gensim(df, verarbeitete_spalte):
    # (Logik aus 5.1)
    if not df.empty and verarbeitete_spalte in df.columns:
        token_liste = df[verarbeitete_spalte].tolist() # Daten extrahieren: Gensim benötigt die Liste von Tokens (Liste von Listen)
        if not any(token_liste):
            print("FEHLER: Die Liste der verarbeiteten Texte ('token_liste') ist leer.")
            return None, None, None
        try:
            id2word = Dictionary(token_liste) # Erstellen des Wörterbuchs (Dictionary). Jedes Wort hat eine ID. 
            id2word.filter_extremes(no_below=5, no_above=0.8) # Wörterbuchs filtern. Entfernt Wörter, die in weniger als 5 Dokumenten (no_below=5) oder in mehr als 80% der Dokumente (no_above=0.8) vorkommen.
            corpus = [id2word.doc2bow(text) for text in token_liste] # Erstellt Korpus (Bag-of-Words-Darstellung) und konvertiert die Texte in das Gensim-Format (Liste von Tupeln (Wort-ID, Zähler)).
            print(f"INFO: Dictionary erstellt: {len(id2word)} einzigartige Token nach Filterung.") # Ausgabe zur Überprüfung.
            print(f"INFO: Gensim Corpus erstellt: {len(corpus)} Dokumente.") # Ausgabe zur Überprüfung.
            print("INFO: Schritt erfolgreich abgeschlossen.") # Ausgabe für Fortschritt.
            return id2word, corpus, token_liste
        except Exception as e:
            print(f"KRITISCHER FEHLER in Gensim-Vorbereitung: {e}")
            return None, None, None
    else:
        print("ABBRUCH: DataFrame ist leer oder Spalte 'verarbeiteter_text' fehlt.")
        return None, None, None

def lda_model_training(corpus, id2word, token_liste, opt_k):
    print(f"\n LDA Modellierung (Finales Modell mit K={opt_k})") # Ausgabe der gesetzen Themen (manuell festgelegt).
    if 'corpus' in locals() and 'id2word' in locals() and 'token_liste' in locals():
        try:
            final_lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=opt_k,random_state=42, passes=15, alpha='auto') # LDA Modell festlegen.
            print(f"INFO: Top 10 Wörter pro Thema (LDA):")
            lda_topics = final_lda_model.print_topics(num_words=10)
            for idx, topic in lda_topics:
                print(f"  Thema #{idx}: {topic}") # Ausgabe der 10 Themen.
            return final_lda_model
        except Exception as e:
            print(f"FEHLER: LDA-Training fehlgeschlagen: {e}")
    else:
        print("FEHLER: Gensim-Variablen fehlen. Abschnitt 5.1 prüfen.")
    return None

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

def nmf_model_training(tfidf_vector, tfidf_features, opt_k):
    print(f"\n NMF Modellierung (Finales Modell mit K={opt_k})") # Ausgabe der gesetzen Themen (manuell festgelegt).
    if 'tfidf_vector' in locals() and 'tfidf_features' in locals(): # NMF benötigt die Scikit-learn Tf-idf-Matrix (tfidf_vector)
        try:
            nmf_model = NMF(n_components=opt_k, random_state=42, init='nndsvda', max_iter=1000)
            nmf_model.fit(tfidf_vector)

            def display_topics(model, feature_names, no_top_words): # Funktion zur Ausgabe der Top-Wörter für NMF
                topics = []
                for topic_idx, topic in enumerate(model.components_):
                    top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
                    topics.append(f"  Thema #{topic_idx}: {top_words}") 
                return topics

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