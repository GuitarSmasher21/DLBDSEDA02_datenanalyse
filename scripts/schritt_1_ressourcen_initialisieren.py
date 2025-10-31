## scripts/schritt_1_ressourcen_initialisieren.py
import nltk # Tokenisierung
from nltk.corpus import stopwords # Liste für Stoppwörter aus NLTK
from HanTa import HanoverTagger as ht # Lemmartisierung über HanTa
from nltk.stem import WordNetLemmatizer # Alternative Lemmatisierung

def run():
    # HanTa-Tagger initialisieren und NLTK-Ressourcen laden

    try: 
        tagger_de = ht.HanoverTagger('morphmodel_ger.pgz')  # HanTa-Modell laden. 
        print("HanTa Lemmatizer erfolgreich initialisiert.")
        hanta_initialisiert = True
    except:
        print("FEHLER: HanTa nicht gefunden oder falsche Initialisierung. Nutze Fallback-Lemmatizer.")
        lemmatizer = WordNetLemmatizer()
        tagger_de = lemmatizer # Alternative Lemmatisierung
        hanta_initialisiert = False

    try:
        nltk.download('popular', quiet=True) # Beliebte NLTK Datensätze installieren (basierend auf Dokumentation).
        de_stoppwoerter = set(stopwords.words('german')) # Deutsche Stoppwörter "füllen".
    except:
        print("NLTK Ressourcen konnt nicht geladen werden oder Stoppwörter nicht verfügbar.") # Fehlermeldung für NLTK ausgeben.
        de_stoppwoerter = set() # Stoppwörter im Zweifel leer setzen

    return tagger_de, hanta_initialisiert, de_stoppwoerter # Ergebnisse an "main.py" zurückgeben