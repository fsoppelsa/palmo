"""
Script per il download e la pulizia del corpus Sherlock Holmes da Project Gutenberg.

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
Università degli Studi di Palermo - Corso di Natural Language Processing
Academic Year 2025/2026
"""
import requests

# Dizionario: nome_logico -> URL Project Gutenberg (versioni testo UTF-8 con -0)
URLS = {
    "study_in_scarlet":      "https://www.gutenberg.org/files/244/244-0.txt",
    "sign_of_four":          "https://www.gutenberg.org/files/2097/2097-0.txt",
    "adventures":            "https://www.gutenberg.org/files/1661/1661-0.txt",
    "memoirs":               "https://www.gutenberg.org/files/834/834-0.txt",
    "hound_baskervilles":    "https://www.gutenberg.org/files/2852/2852-0.txt",
    "return":                "https://www.gutenberg.org/files/108/108-0.txt",
}

def clean_gutenberg_text(text: str) -> str:
    """
    Rimuove header e footer tipici di Project Gutenberg
    mantenendo solo il corpo del libro.
    Se non trova i marker, restituisce il testo originale.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx   = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        # Spostiamo l'indice subito dopo la riga del marker
        # per evitare di includere l'intestazione tecnica
        # (si può raffinare cercando la prima doppia newline dopo il marker)
        body = text[start_idx:end_idx]
        # Togliamo la prima riga (il marker) e spazi iniziali
        body = body.splitlines()
        if body:
            body = body[1:]  # salta la riga del marker
        body = "\n".join(body).strip()
        return body
    else:
        # Nessun marker trovato: ritorna il testo così com'è (fallback)
        return text.strip()

def download_and_merge_sherlock(output_path: str = "sherlock.txt"):
    all_docs = []

    print("Inizio download e pulizia dei testi di Sherlock Holmes...\n")

    for name, url in URLS.items():
        print(f"Scarico: {name}  ->  {url}")
        try:
            resp = requests.get(url)
            resp.encoding = "utf-8"  # assicuriamo UTF-8
            raw_text = resp.text

            clean_text = clean_gutenberg_text(raw_text)

            print(f"  Lunghezza dopo pulizia: {len(clean_text)} caratteri\n")

            all_docs.append(clean_text)
        except Exception as e:
            print(f"  ERRORE su {name}: {e}\n")

    # Ora concateno tutti i documenti con il tag <eos> tra uno e l'altro
    # Esempio: [doc1, doc2, doc3] -> "doc1\n\n<eos>\n\ndoc2\n\n<eos>\n\ndoc3"
    merged = ("\n\n<eos>\n\n").join(all_docs)

    print(f"Corpus finale: {len(merged)} caratteri totali.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)

    print(f"\nFile salvato come: {output_path}")
    print("Ogni documento è separato da una riga con il tag <eos>.")

if __name__ == "__main__":
    download_and_merge_sherlock()
