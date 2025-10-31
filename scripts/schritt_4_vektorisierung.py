## pipeline/step_4_vectorize.py
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Module für Bag-of-Words + Tf-idf
import numpy as np # Vektorisierung

def run(df, vektor_spalte):
    # Auführung von BoW- und Tf-idf-Vektorisierung. Zudem ein Abgleich zwischen beiden Varianten.
    
    text_data = df[vektor_spalte]
    
    # --- 4.2 BoW ---
    print("\nINFO: Starte Bag-of-Words Vektorisierung...")
    bow_vector = CountVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2)) # CountVectorizer für BoW initialisieren für Wörter, die in weniger als 5 und in mehr als 80% der Dokumente vorkommen und Bigramme sind.
    bow_matrix = bow_vector.fit_transform(text_data) # Daten anpassen und in Dokument-Begriff-Matrix überführen.
    bow_feature = bow_vector.get_feature_names_out() 
    print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
    print(bow_feature[:10]) # Überprüfung der Begriffe, die für Matrix-Bildung genutzt wurden. Gibt erste 10 Themen aus.
    print(f"INFO: Bag-of-Words Feature-Matrix erstellt.")
    print(f"INFO: Dimensionen der BoW-Matrix: {bow_matrix.shape[0]} Dokumente x {bow_matrix.shape[1]} Features.")

    # --- 4.3 Tf-idf ---
    print("\nINFO: Starte TF-IDF Vektorisierung...")
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2)) # TfidfVectorizer initialisieren (gleiche Parameter wie BoW)
    tfidf_vector = tfidf_vectorizer.fit_transform(text_data)
    tfidf_features = tfidf_vectorizer.get_feature_names_out()
    print(f"INFO: Beispielhafte Feature-Namen (die ersten 10):")
    print(tfidf_features[:10]) # Überprüfung der Begriffe, die für Matrix-Bildung genutzt wurden. Gibt erste 10 Themen aus.
    print(f"INFO: TF-IDF Feature-Matrix erstellt.")
    print(f"INFO: Dimensionen der TF-IDF-Matrix: {tfidf_vector.shape[0]} Dokumente x {tfidf_vector.shape[1]} Features.")    

    # --- 4.4 Vergleich ---
    vergleich_vektoren(bow_matrix, bow_feature, tfidf_vector, tfidf_features) # Funktions-Aufruf
    
    return bow_matrix, bow_feature, tfidf_vector, tfidf_features # Rückgabe an "main.py"

def vergleich_vektoren(bow_matrix, bow_feature, tfidf_vector, tfidf_features): # Hilfsfunktion zur Ausgabe
   
    bow_shape = bow_matrix.shape
    tfidf_shape = tfidf_vector.shape
    print(f"INFO: Dimensionen: BoW {bow_shape} | Tf-idf {tfidf_shape}") # Definition allgemeiner Metriken (sollten gleich sein, da gleiche Parameter verwendet wurden).
    print(f"INFO: Vokabulargröße: {len(bow_feature)} Features") # Ausgabe Vokabulargröße.

    # Ermittlung der Top 10 Features für BoW (basierend auf absoluter Häufigkeit)
    bow_sum = np.sum(bow_matrix, axis=0)
    bow_counts = [(bow_feature[i], bow_sum[0, i]) for i in range(bow_matrix.shape[1])]
    bow_counts.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 BoW (häufigste Vorkommen):") # Ausgabe der gebildeten Matrix für Vergleich.
    for word, count in bow_counts[:10]:
         print(f"  {word}: {int(count)}")

    # Ermittlung der Top 10 Features für Tf-idf (basierend auf der gewichteten Wichtigkeit)
    tfidf_sum = np.sum(tfidf_vector, axis=0)
    tfidf_scores = [(tfidf_features[i], tfidf_sum[0, i]) for i in range(tfidf_vector.shape[1])]
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 Tf-idf (höchste Wichtigkeit/Spezifität):")
    for word, score in tfidf_scores[:10]:
         print(f"  {word}: {score:.4f}")