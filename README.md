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

## 1. Installation

Es wird empfohlen, eine virtuelle Umgebung (z.B. `conda` oder `venv`) zu verwenden.

Im Ordner "environment" befinden sich die Umgebungsdateien "requirements.txt" und "requirements.yml", welche entsprechend in eine neue virtuelle Umgebung importiert werden können.

## 2. Script-Aufbau

Der Code wurde in eine Hauptdatei zusammengefasst (main.py). Für die Ausführung muss die Datei "unstructured_data.json" heruntergeladen und in einem lokalen Verzeichnis abgelegt werden.
Der verwendete Dateipfad ist ebenfalls in der Hauptdatei (main.py) anzupassen, damit auf die Datei im Programmablauf zugegriffen werden kann.
Ergebnisse, die sich aus der Ausführung des Skripts ergeben, werden entsprechend im selben lokalen Pfad wie die zuvor geladene Datei abgelegt. 

## 3. Ergebnisse
Die visuellen Ergebnisse aus der Python-Script Ausführung befinden sich im Ordner "results".
