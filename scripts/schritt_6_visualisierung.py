## scripts/schritt_6_visualisierung.py
import matplotlib.pyplot as plt # Modul Visualisierung Coherence Score

def run(model, topic_id=4, filepath='lda_themen_diagramm.png'):
    
    # Erstellung und Speicherung eines Matplotlib-Balkendiagramm.

    print(f"\n Finale Visualisierung (Matplotlib)")
    if model: # Prüfen, ob das finale LDA-Modell existiert
        try:
            topic_terms = model.show_topic(topic_id, topn=10) # Wahl eines interessanten Themas zum Visualisieren. 
            words = [term[0] for term in topic_terms]  # Daten für den Matplotlib-Plot vorbereiten.
            probabilities = [term[1] for term in topic_terms] 
            
            # Plot erstellen (horizontales Balkendiagramm)
            plt.figure(figsize=(10, 7))
            plt.barh(words, probabilities, color='steelblue') 
            plt.xlabel("Wichtigkeit (Wahrscheinlichkeit) im Thema")
            plt.ylabel("Top-Wörter")
            plt.title(f"Matplotlib: Detaillierte Ansicht für LDA Thema #{topic_id}")
            plt.gca().invert_yaxis() # Wichtigste Wort oben anzeigen
            plt.tight_layout()
            
            plt.savefig(filepath)
            print(f"INFO: Matplotlib-Visualisierung '{filepath}' gespeichert.")
        except Exception as e:
            print(f"FEHLER bei Matplotlib-Visualisierung von Thema #{topic_id}: {e}")
    else:
        print("INFO: Finales LDA-Modell nicht gefunden, überspringe Matplotlib-Balkendiagramm.")