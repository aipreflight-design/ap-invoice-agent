# Dag 2: RAG op leverancierscontracten

**Probleem:** je agent weet wat er op de factuur en de bestelbon staat, maar niet wat er in het
contract staat. Mag Delcourt zijn prijs verhogen? Is een spoedtoeslag toegelaten? Wat moet je doen
bij een nieuw IBAN? Dat staat in contracten, en die kent het model niet.

**Oplossing: RAG** (Retrieval-Augmented Generation): zoek eerst de relevante stukken op en geef
alleen die aan het model.

## Nieuwe bestanden

| Bestand | Wat |
|---|---|
| `data/contracts/` | Drie contracten: Brouwers (NL), Delcourt (FR), TechParts (EN) |
| `rag.py` | RAG zonder framework: chunken → embedden → zoeken |
| `day2_ask.py` | Vragen stellen over de contracten, met bronvermelding |
| `day2_agent.py` | De AP-agent van dag 1 + een tool `search_contract` |

Niets nieuws te installeren: embeddings gebruiken hetzelfde `openai`-pakket en dezelfde API-key.

## De vier stappen van RAG (lees `rag.py`)

1. **Chunken:** de contracten in stukjes knippen, hier per artikel.
2. **Embedden:** elk stukje omzetten in een *vector*, een lijst van 1.536 getallen die de
   **betekenis** vat. Teksten over hetzelfde onderwerp krijgen vectoren die dicht bij elkaar liggen,
   ook in een andere taal. Kost: een fractie van een cent voor alle drie de contracten.
3. **Zoeken:** de vraag ook omzetten in een vector en de stukjes zoeken die er het dichtst bij liggen
   (*cosine similarity*: 1 = zelfde betekenis, 0 = niets gemeen).
4. **Genereren:** alleen die stukjes + de vraag naar het model sturen, met de opdracht om
   **alleen** daarop te antwoorden en de **bron** te vermelden.

De index wordt de eerste keer automatisch opgebouwd en bewaard in `contracts_index.json`.
Bekijk dat bestand gerust: je ziet de stukjes tekst met hun lange lijst getallen.

## Oefeningen

### 1. Stel vragen, ook over contracten in een andere taal

```
python day2_ask.py "Mag Delcourt zijn prijzen verhogen, en met hoeveel?"
python day2_ask.py "Wanneer mag Brouwers een spoedtoeslag aanrekenen?"
python day2_ask.py "Wat moeten we doen als een transporteur een nieuw rekeningnummer doorgeeft?"
```

Kijk naar de **gevonden fragmenten** en hun **score** boven elk antwoord. Je vraag is in het
Nederlands, het contract van Delcourt in het Frans. Wordt het juiste artikel toch gevonden?

### 2. Vraag iets wat er níet in staat

```
python day2_ask.py "Wat is de opzegtermijn van het contract met TechParts?"
```

Het TechParts-contract heeft bewust geen opzegclausule. Zegt het model eerlijk dat het dat
niet weet? Kijk dan naar de fragmenten: welke artikels vond het, en waarom?

**Breek het daarna:** verwijder in `day2_ask.py` in `SYSTEM` de zin die begint met
"Gebruik UITSLUITEND" en de zin "Staat het antwoord er niet in...". Stel dezelfde vraag opnieuw.
Begint het model nu te gokken (bv. "meestal drie maanden")? Dat heet *hallucineren*. Zet de
zinnen daarna terug.

### 3. Slecht chunken

```
CHUNK_MODE=fixed python day2_ask.py "Mag Delcourt zijn prijzen verhogen, en met hoeveel?"
```

Nu wordt elke 300 tekens geknipt, midden in zinnen en artikels. Wordt het artikel over de
indexering nog goed gevonden? Is het antwoord nog volledig (het plafond van 3% én de
verplichte melding 30 dagen vooraf)? Probeer ook `CHUNK_SIZE=100`.

**Les:** de meeste RAG-problemen zijn geen modelproblemen maar **chunk-problemen**. Knip
volgens de structuur van het document.

### 4. De agent met contracten

```
rm -f data/invoice_ledger.csv approval_queue.jsonl
python day2_agent.py
```

Vergelijk de redenen met die van dag 1. Verwacht:
- **INV-1002 (Delcourt):** de prijs steeg met ~15%, het contract laat maximaal 3% per jaar toe,
  met een verplichte melding vooraf (Article 3).
- **INV-1003 (Brouwers, nieuw IBAN):** wijzigingen enkel via een ondertekende brief en telefonische
  bevestiging via het nummer uit het contract (Artikel 5).
- **INV-1004 (TechParts, geblokkeerd):** tijdens een schorsing worden geen facturen betaald
  tot het kwaliteitsincident afgesloten is (Section 3).

Daarmee weet de financeafdeling niet alleen **dat** er iets mis is, maar ook **wat ze moet
doen**. Dat verlaagt de kost van elke uitzondering, en dat was de echte kost (zie dag 1).

Vergelijk ook het tokenverbruik met dag 1. Wat kost het extra opzoeken?

### 5. Denkvraag voor het interview: heb je RAG hier eigenlijk nodig?

De drie contracten samen zijn een paar duizend tokens. Moderne modellen kunnen een miljoen
tokens tegelijk lezen. Je zou ze dus ook gewoon **volledig in de prompt** kunnen zetten.

Wanneer is RAG dan wél nodig? Denk aan:
- honderden of duizenden documenten (kost en snelheid),
- documenten die vaak veranderen,
- **rechten**: niet iedereen mag elk document zien (zie askHR),
- bronvermelding en controleerbaarheid.

Een goed antwoord in het interview begint met: *"Bij een handvol documenten zet ik ze gewoon
in de prompt. RAG wordt nodig zodra..."*
