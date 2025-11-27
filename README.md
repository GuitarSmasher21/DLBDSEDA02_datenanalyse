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

Es wird empfohlen, das GitHub Repository auf den lokalen Rechner zu laden.
Hierfür sollten beispielhaft folgende Schritte für VisualStudio Code ausgeführt werden:
1. Neuen Ordner anlegen
2. VisualStudio Code öffnen
3. "GitRepository klonen..." auswählen
4. URL aus dem Full Abstract angeben und mit "Enter-Taste" bestätigen
5. In Schritt 1 gewählten Ordner auswählen
6. "Öffnen" wählen
7. Nun steht das GitHub Projekt auf dem lokalen Rechner bereit.

Es wird außerdem empfohlen, eine virtuelle Umgebung (z.B. `conda` oder `venv`) zu verwenden.
Im Ordner "environment" befinden sich die Umgebungsdateien "requirements.txt" und "requirements.yml", welche entsprechend in eine neue virtuelle Umgebung importiert werden können.
Der Import ist über zwei Befehle möglich:
1. pip
_pip install -r environments.txt_
2. conda
_conda create --name data_analysis_schrieber --file environments.txt_

## 2. Script-Aufbau

Der Code wurde in eine Hauptdatei zusammengefasst (main.py) und kann nur über diese Hauptdatei ausgeführt werden.

## 3. Script-Ausführung

Vor der Ausführung muss die Datei "unstructured_data.json" heruntergeladen und in einem lokalen Verzeichnis abgelegt werden. Der Dateipfad ist zu notieren.
Der Dateipfad ist anschließend in der Hauptdatei (main.py) in Zeile 298 anzupassen, damit auf die Datei im Programmablauf zugegriffen werden kann.
Ergebnisse, die sich aus der Ausführung des Skripts ergeben, werden entsprechend im selben lokalen Pfad abgelegt. 

## 3. Ergebnisse
Die visuellen Ergebnisse aus der Python-Script Ausführung befinden sich im Ordner "results".
