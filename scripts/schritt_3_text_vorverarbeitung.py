## scripts/schritt_3_text_vorverarbeitung.py
import re # Regular Expressions (Muster in Zeichenketten)
from nltk.tokenize import word_tokenize # Modul für Tokenisierung 

def run(df, text_spalte, verarbeitete_spalte, tagger_de, hanta_initialisiert, de_stoppwoerter):
  
    # Gesamte Text-Vorverarbeitungen (Regex, Tokenisierung, Lemmatisierung) ausgeführt.
    if not (text_spalte in df.columns):
        print(f"FEHLER: Textspalte '{text_spalte}' nicht im DataFrame gefunden.")
        return df # Ausgabe falscher Initial-Spalte

    print(f"\nINFO: Starte Vorverarbeitung der Spalte '{text_spalte}'...") # Spalte in "main.py" definiert. In unserem Fall muss das "Beschwerden" sein. 
    text_verarbeitung = df[text_spalte].astype(str).tolist() # Umwandlung des DataFrames in eine Liste
    text_verarbeitung_fertig = [] # Platzhalter für finale Version. 
    gesamt_texte = len(text_verarbeitung) # Zähler für Verarbeitungs-Fortschritt.

    for i, text in enumerate(text_verarbeitung): 
        text = re.sub(r'[^a-zA-Zäöüß\s]', ' ', text) # Entfernen von allen Zeichen, außer Klein-/Großbuchstaben und Sonderzeichen.
        tokens = word_tokenize(text, 'german') # Tokenisierung mit deutschem Wortschatz. 
        tokens = [word for word in tokens if word not in de_stoppwoerter and len(word) > 2] # Token prüfen und Wörter die kleiner als 2 Zeichen sind raus nehmen.
        
        final_tokens = [] # Platzhalter für finale Version.
        
        if hanta_initialisiert:
            try:
                tagged_tokens = tagger_de.tag_sent(tokens) # Token in HanTa laden.
                final_tokens = [lemma_tag[1] for lemma_tag in tagged_tokens] # HanTa für Index = 1 (das eigentliche Wort) durchlaufen.
            except Exception as e:
                print(f"WARNUNG: HanTa Fehler bei Text {i}: {e}")  # Fehlerausgabe
                final_tokens = [word.lower() for word in tokens if word not in de_stoppwoerter and len(word) > 2] # Weitere Behandlung ohne HanTa.

        elif not hanta_initialisiert:
            try:
                final_tokens = [tagger_de.lemmatize(word) for word in tokens] # Alternative Lemmartizer genutzt (anstelle HanTa).
            except Exception as e:
                 print(f"WARNUNG: Fallback-Lemmatizer fehlgeschlagen: {e}") # Fehlerausgabe für alternativen Lemmartizer
                 final_tokens = tokens # Rückgabe ohne Tokenisierung + Lemmartisierung.

        text_verarbeitung_fertig.append(final_tokens) # Rückgabe finale Version
        
        if (i + 1) % 1000 == 0: 
            print(f"INFO: {i + 1}/{gesamt_texte} Zeilen verarbeitet...") # Fortschritts-Zähler

    df[verarbeitete_spalte] = text_verarbeitung_fertig # Fertige Daten in DataFrame zurück schreiben.
    df['vektor_eingabe'] = df[verarbeitete_spalte].str.join(' ') # Einzelwörter wieder in einen String zusammenfassen für weitere Verarbeitung. 
    return df # Rückgabe an "main.py"