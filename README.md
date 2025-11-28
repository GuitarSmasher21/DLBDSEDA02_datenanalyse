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

Es wird empfohlen, das GitHub Repository auf den lokalen Rechner zu laden. Hierfür sollten beispielhaft folgende Schritte für VisualStudio Code ausgeführt werden:

1. Neuen Ordner anlegen
2. VisualStudio Code öffnen
3. "GitRepository klonen..." auswählen
4. URL aus dem Full Abstract angeben und mit "Enter-Taste" bestätigen
5. In Schritt 1 gewählten Ordner auswählen
6. Öffnen" wählen
7. Nun steht das GitHub Projekt auf dem lokalen Rechner bereit.

Es wird außerdem empfohlen, eine virtuelle Umgebung (z.B. conda oder venv) zu verwenden. Im Ordner "environment" befinden sich die Umgebungsdateien "requirements.txt" und "requirements.yml", welche entsprechend in eine neue virtuelle Umgebung importiert werden können. Der Import ist über zwei Befehle möglich:

* pip pip install -r environments.txt
* conda conda create --name data_analysis_schrieber --file environments.txt

## 2. Script-Aufbau
Der Code wurde in eine Hauptdatei zusammengefasst (main.py) und kann nur über diese Hauptdatei ausgeführt werden.

## 3. Script-Ausführung
* main.py
** Zusammengefasster Hauptcode
** Zur Ausführung wird die Datei "unstructured_data.json" in einem lokalen Verzeichnis benötigt. Diese muss also vorab heruntergeladen werden.
** Dateipfad ist im Hauptcode in Zeile 298 anzupassen
** Ergebnisse nach Ausführung des Codes befinden sich im selben lokalen Pfad wie im vorherigen Schritt definiert.

* download_data.py
** Hilfssrkipt, mit welchem die Datei "unstructured_data.json" heruntergeladen wurde.
** Dieses Skript muss *nicht* ausgeführt werden und dient nur zur Dokumentation.

## 4. Ergebnisse
Die visuellen Ergebnisse aus der Python-Script Ausführung befinden sich im Ordner "results".
