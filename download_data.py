## Module importieren
import requests


## Python Code
data_url = "https://ordnungsamt.berlin.de/frontend.webservice.opendata/api/meldungen" ## URL definieren
local_filename = "/Users/michi/downloads/unstructured_data.json" ## Datei lokal speichern

try:
    response = requests.get(data_url) ## Daten abrufen
    response.raise_for_status() ## Löst einen HTTPError für schlechte Antworten (4xx oder 5xx) aus

    with open(local_filename, 'wb') as file: ## Datensatz lokal speichern
        file.write(response.content)
    
    print(f"Daten erfolgreich in '{local_filename}' heruntergeladen.") ## Ausgabe, wenn Erfolg

except requests.exceptions.RequestException as e:
    print(f"Fehler beim Herunterladen der Daten: {e}") ## Fehler-Handleing