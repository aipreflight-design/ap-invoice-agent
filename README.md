# AP agent: checking supplier invoices with an LLM, safely

A small proof of concept of an accounts-payable agent for a Belgian SME. It reads incoming
supplier invoices (in Dutch, French, German or English), checks them against the supplier
master data, purchase orders, previously booked invoices and supplier contracts, and either
approves them for payment or routes them to a human with a concrete reason.

Built in a few days to get hands-on with tool calling, the OpenAI Agents SDK and RAG.
The data is fictional.

## Design principles

1. **The model reads, the code decides.** The LLM extracts fields from messy, multilingual
   invoices. All business rules (price tolerance, IBAN check, blocked suppliers, approval limit,
   duplicate invoices, cumulative quantity per PO) live in plain Python in `tools.py`.
2. **Enforce in the schema, not the prompt.** Types and required fields are part of the tool
   schema. With a loose schema (`unit_price` as a string), the model copied "850,00" from the
   invoice, the check failed, and a blocked supplier was flagged for the wrong reason.
3. **The agent can't skip the check.** `approve_for_payment` refuses any invoice that has not
   passed `check_invoice` in the same run. A confused or manipulated model can at worst produce
   a wrong recommendation, not a wrong payment.
4. **Fail closed, with the right reason.** Anything doubtful goes to a human. A wrong reason is
   worse than no reason, because it leads to a wrong approval.
5. **The contract informs the human, it doesn't decide.** When a check fails, the agent looks up
   the relevant contract clause (RAG) and tells finance what to do, e.g. "Article 3 allows max. 3%
   indexation per year with 30 days' notice; this invoice is +15%".

## Test cases

| Invoice | Language | Expected |
|---|---|---|
| INV-1001 | NL | Approved |
| INV-1002 | FR | Flagged: unit price above PO (contract allows max. 3% indexation) |
| INV-1003 | EN | Flagged: IBAN differs from master data, possible fraud (contract: bank changes need a signed letter + call-back) |
| INV-1004 | DE | Flagged: supplier blocked (contract: no payments during a quality suspension) |
| INV-1005 | NL | Flagged: above the €5,000 approval limit |
| INV-1006 | NL | Flagged: PO already fully invoiced (duplicate) |

## Files

| File | What it shows |
|---|---|
| `tools.py` | Business logic: deterministic checks, ledger, payment lock |
| `day1_raw_loop.py` | The agent without a framework: tool calling and the agent loop in ~40 lines |
| `day1_agents_sdk.py` | The same agent with the OpenAI Agents SDK: schemas from type hints, structured output, tracing, token usage |
| `rag.py` | RAG without a framework: chunking (per contract article, or fixed-size for comparison), embeddings, cosine similarity search, metadata filter |
| `day2_ask.py` | Questions about the contracts, answered only from retrieved clauses, with sources |
| `day2_agent.py` | The full agent: invoice checks + contract lookup as a tool ("agentic RAG") |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai openai-agents pydantic
export OPENAI_API_KEY=sk-...

rm -f data/invoice_ledger.csv approval_queue.jsonl   # start clean
python day2_agent.py
python day2_ask.py "Can Delcourt raise its prices, and by how much?"
CHUNK_MODE=fixed python day2_ask.py "Can Delcourt raise its prices, and by how much?"
```

## What I learned

- **Cost is a rounding error:** ~1,100 input and ~160 output tokens per invoice with gpt-4.1-mini,
  under $0.001 per invoice. The business case is driven by the straight-through rate and the
  cost of handling exceptions, not by model price.
- **Most RAG problems are retrieval problems.** Chunking per article works; fixed 300-character
  chunks cut clauses in half and the answers got incomplete. Cross-lingual retrieval works but
  isn't perfect: a French clause on price indexation scored lower than an unrelated Dutch sentence
  in a direct comparison. Filtering on metadata (only the supplier's own contract) helps a lot.
- **With a handful of documents you don't need RAG**: put them in the prompt. RAG matters with many
  documents, frequent changes, permissions and citations.

## What production would need

- Read Peppol e-invoices (structured XML, mandatory for Belgian B2B since January 2026) directly,
  and use an LLM only for foreign PDFs.
- ERP integration (e.g. Exact Online, Odoo, SAP Business One) through an existing MCP server or API,
  with least-privilege access.
- An evaluation set of real invoices, run on every change: never approve a bad invoice, then
  maximise the straight-through rate.
- EU hosting, `store=False`, tracing to an own system, an approval screen for finance, monitoring.
