# Projekt Data Analysis (DLBDSEDA02_D)
# Aufgabe 1: NLP-Techniken zur Analyse einer Textsammlung
Dieses Repository enthält die Python-Implementierung für Aufgabe 1 im Rahmen des Kurses "Projekt: Data Analysis" (DLBDSEDA02_D) an der IU Internationalen Hochschule.

Die Analyse wird anhand folgender Schritte ausgeführt:
* Datenimport und -bereinigung (JSON-Normalisierung)
* Text-Vorverarbeitung (Tokenisierung, Regex-Bereinigung, Stoppwort-Entfernung)
* Lemmatisierung mittels HanTa
* Feature-Extraktion durch Bag-of-Words (BoW) und Tf-idf
* Themenmodellierung mit LDA (Latent Dirichlet Allocation) und NMF (Non-Negative Matrix Factorization)
* Visualisierung der Ergebnisse mittels pyLDAvis und matplotlib

### 1. Installation

Es wird empfohlen, eine virtuelle Umgebung (z.B. `conda` oder `venv`) zu verwenden.
Dazu stehen die Umgebungsdateien "requirements.txt" und "requirements.yml" zur Verfügung.

## 2. Script-Aufbau

Der Code ist modular aufgebaut, um die Lesbarkeit und Wartbarkeit zu verbessern.
* main.py: Das Hauptskript, das den gesamten Workflow steuert und die einzelnen Module aufruft.
* schritt_1_ressourcen_initialisieren.py: Initialisiert HanTa und NLTK-Ressourcen.
* schritt_2_daten_laden.py: Lädt und normalisiert die JSON-Daten.
* schritt_3_text_vorverarbeitung.py: Führt die NLP-Vorverarbeitung durch.
* schritt_4_vektorisierung.py: Führt BoW- und Tf-idf-Vektorisierung sowie den Vergleich durch.
* schritt_5_themenmodilierung.py: Trainiert die LDA- und NMF-Modelle und erstellt die pyLDAvis-Visualisierung.
* schritt_6_visualisierung.py: Erstellt das finale matplotlib-Balkendiagramm.
