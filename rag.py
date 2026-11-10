"""
DAG 2 — RAG zonder framework: chunken, embedden, zoeken.

RAG = Retrieval-Augmented Generation:
  1. CHUNKEN    Knip documenten in stukjes (chunks).
  2. EMBEDDEN   Zet elk stukje om in een vector: een lijst getallen die de BETEKENIS vat.
                Stukjes over hetzelfde onderwerp krijgen vectoren die dicht bij elkaar liggen,
                ook als ze in een andere taal of met andere woorden geschreven zijn.
  3. ZOEKEN     Zet de vraag ook om in een vector en zoek de stukjes die er het dichtst bij liggen.
  4. GENEREREN  Geef ALLEEN die stukjes aan het model, samen met de vraag.

Net als bij dag 1 doen we dit eerst met de hand, zodat je ziet wat een vector-database
(Chroma, pgvector, Azure AI Search, ...) of een framework (LlamaIndex, LangChain) voor je doet.
"""

import json
import math
import os
import re
from pathlib import Path

from openai import OpenAI

CONTRACTS = Path(__file__).parent / "data" / "contracts"
INDEX_FILE = Path(__file__).parent / "contracts_index.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# "section": knip per artikel (volgt de structuur van het contract)
# "fixed":   knip elke CHUNK_SIZE tekens, ongeacht de inhoud (oefening 3)
CHUNK_MODE = os.getenv("CHUNK_MODE", "section")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------- 1. CHUNKEN ----------

def chunk_document(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()

    if CHUNK_MODE == "fixed":
        pieces = [("deel %d" % (i // CHUNK_SIZE + 1), text[i:i + CHUNK_SIZE])
                  for i in range(0, len(text), CHUNK_SIZE)]
    else:
        # Elk stuk begint bij een "## "-kop. Het stuk vóór de eerste kop is de inleiding.
        parts = re.split(r"\n(?=## )", text)
        pieces = []
        for part in parts:
            first_line = part.strip().splitlines()[0]
            section = first_line.lstrip("# ").strip() if first_line.startswith("##") else "Inleiding"
            pieces.append((section, part.strip()))

    # Metadata meegeven is cruciaal: daarmee kan het model later de BRON vermelden.
    return [{"source": path.name, "title": title, "section": section,
             "text": f"[{title} – {section}]\n{body}"}
            for section, body in pieces if body.strip()]


# ---------- 2. EMBEDDEN ----------

def embed(texts: list[str]) -> list[list[float]]:
    response = client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index() -> list[dict]:
    chunks = [c for p in sorted(CONTRACTS.glob("*.md")) for c in chunk_document(p)]
    vectors = embed([c["text"] for c in chunks])  # één API-call voor alle stukjes
    for c, v in zip(chunks, vectors):
        c["embedding"] = v
    INDEX_FILE.write_text(json.dumps({"mode": CHUNK_MODE, "size": CHUNK_SIZE,
                                      "model": EMBEDDING_MODEL, "chunks": chunks}),
                          encoding="utf-8")
    return chunks


def load_index() -> list[dict]:
    """Laad de index; bouw ze opnieuw als ze niet bestaat of met andere instellingen gemaakt is."""
    if INDEX_FILE.exists():
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if (data["mode"], data["size"], data["model"]) == (CHUNK_MODE, CHUNK_SIZE, EMBEDDING_MODEL):
            return data["chunks"]
    print(f"(index wordt opgebouwd: modus={CHUNK_MODE}, model={EMBEDDING_MODEL})")
    return build_index()


# ---------- 3. ZOEKEN ----------

def cosine(a: list[float], b: list[float]) -> float:
    """Hoe dicht liggen twee vectoren bij elkaar? 1 = zelfde richting, 0 = niets gemeen."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def search(question: str, k: int = 3, source: str | None = None) -> list[dict]:
    chunks = load_index()
    if source:  # filteren op metadata, bv. alleen het contract van één leverancier
        chunks = [c for c in chunks if c["source"] == source]
    q = embed([question])[0]
    scored = [{**c, "score": cosine(q, c["embedding"])} for c in chunks]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return [{k_: v for k_, v in c.items() if k_ != "embedding"} for c in scored[:k]]
