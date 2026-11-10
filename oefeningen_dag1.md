# AP-agent: spoedcursus agents bouwen (5 dagen)

Eén use case die de hele week meegaat: **een agent die inkomende facturen controleert
tegen leveranciersmaster en bestelbonnen**, en ze goedkeurt of doorstuurt naar een mens.
Dat is een PoC die bij bijna elk portfoliobedrijf meteen herkenbaar is.

| Dag | Wat je bouwt | Wat je ermee leert |
|-----|--------------|--------------------|
| 1 | Agent zonder framework → dezelfde agent met de OpenAI Agents SDK | Wat een agent werkelijk is: tool calling + een lus |
| 2 | Vragen beantwoorden over leverancierscontracten (RAG) | Embeddings, chunking, retrieval, bronvermelding |
| 3 | Dezelfde flow in LangGraph, met een goedkeuringsstap door een mens | State, vertakkingen, checkpoints, human-in-the-loop |
| 4 | De ERP-data beschikbaar maken via een MCP-server + dezelfde flow in n8n | Koppelingen met systemen; low-code versus code |
| 5 | Eenvoudige interface, evaluatieset, kosten per factuur | Van PoC naar iets wat je een CFO kan tonen |

## Setup (±10 min)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install openai openai-agents pydanticpython3 -m venv .venv
export OPENAI_API_KEY=sk-...     # Windows: set OPENAI_API_KEY=sk-...
```

Een API-key maak je aan op platform.openai.com (zet er een paar euro krediet op; deze hele
week kost hooguit enkele euro's). Het standaardmodel is `gpt-4.1-mini`. Een ander model
kies je met `export MODEL=...` (kijk op platform.openai.com/docs/models welke modellen
er nu zijn).

## De testdata

`data/invoices/` bevat vier facturen. Elk heeft een eigen valkuil, en ze zijn in verschillende talen,
zoals bij een Belgisch bedrijf met leveranciers in BE/FR/DE:

| Factuur | Taal | Wat er moet gebeuren |
|---------|------|----------------------|
| INV-1001 | NL | Alles klopt → goedkeuren |
| INV-1002 | FR | Eenheidsprijs 1,32 i.p.v. 1,15 op de PO → tegenhouden |
| INV-1003 | EN | Ander IBAN dan in de master ("new bank details") → klassieke fraude, tegenhouden |
| INV-1004 | DE | Leverancier is geblokkeerd → tegenhouden |

## Dag 1

```bash
python day1_raw_loop.py      # zie elke tool-aanroep voorbijkomen
python day1_agents_sdk.py    # zelfde resultaat, als gestructureerde tabel
```

Lees eerst `tools.py`, dan `day1_raw_loop.py`, dan `day1_agents_sdk.py`. Belangrijke
ontwerpkeuze: **de LLM leest en interpreteert de factuur, maar de controle zelf zit in gewone
code** (`check_invoice`). Dat is precies wat je in het interview wil kunnen uitleggen.

### Oefeningen (doe ze echt zelf, desnoods met Claude Code of Cursor als hulp)

1. **Breek het.** Haal in `day1_raw_loop.py` de zin over "decimale getallen met punt" uit de
   tool-beschrijving. Wat gebeurt er met de Franse factuur (1,32)? Kijk naar de print-output.
2. **Nieuwe regel.** Voeg in `check_invoice` een controle toe: totaal boven 5.000 EUR moet
   altijd door een mens. Maak een INV-1005 die daarop faalt.
3. **Dubbele factuur.** INV-1003 factureert dezelfde PO als INV-1001. Voeg een tool of regel
   toe die dubbel factureren detecteert.
4. **Tracing.** Draai `day1_agents_sdk.py` en open platform.openai.com/traces. Hoeveel stappen,
   hoeveel tokens? Reken de kost per factuur uit.
5. **Ander model.** Draai met een kleiner en een groter model. Wat verandert er aan kwaliteit
   en kost?

### Wat je hieruit meeneemt naar het interview

- *"Een agent is een LLM in een lus met tools. Het framework is bijzaak; de tools en de
  guardrails zijn het eigenlijke werk."*
- *"Ik laat het model interpreteren (meertalige, rommelige documenten) en laat code
  beslissen (regels, rekenwerk). Uitzonderingen gaan naar een mens."*
- *"Een PoC als deze staat er bij een portfoliobedrijf in een paar dagen. Daarna is het werk:
  koppelen aan het ERP, een evaluatieset op echte facturen, monitoring, en de
  goedkeuringsflow inbedden in het proces van de financeafdeling."*
