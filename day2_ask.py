"""
DAG 2a — Vragen stellen over de leverancierscontracten.

Draaien:  python day2_ask.py "Mag Delcourt zijn prijzen verhogen?"
Of zonder vraag, dan kan je er meerdere na elkaar typen (stoppen: Enter op een lege regel).
"""

import os
import sys

import rag

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
TOP_K = int(os.getenv("TOP_K", "3"))

# ---------- 4. GENEREREN ----------
# De twee belangrijkste regels van elke RAG-prompt:
#   - antwoord ALLEEN op basis van de gegeven fragmenten (anders verzint het model iets),
#   - vermeld de BRON (zodat een mens het kan controleren).
SYSTEM = """Je beantwoordt vragen over leverancierscontracten van een Belgisch KMO.
Gebruik UITSLUITEND de contractfragmenten hieronder. Staat het antwoord er niet in,
zeg dan letterlijk: "Dat staat niet in de gevonden contractfragmenten."
Vermeld na elk feit de bron tussen haakjes, bv. (Brouwers Logistiek – Artikel 3).
Antwoord in het Nederlands, ook als het contract in een andere taal is."""


def answer(question: str) -> str:
    hits = rag.search(question, k=TOP_K)

    print("\n  Gevonden fragmenten (de 'R' van RAG):")
    for h in hits:
        print(f"   {h['score']:.3f}  {h['source']:<26} {h['section']}")

    context = "\n\n---\n\n".join(h["text"] for h in hits)
    response = rag.client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Contractfragmenten:\n\n{context}\n\nVraag: {question}"}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\n" + answer(" ".join(sys.argv[1:])))
    else:
        while (q := input("\nVraag (Enter = stoppen): ").strip()):
            print("\n" + answer(q))
