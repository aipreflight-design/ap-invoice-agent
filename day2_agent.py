"""
DAG 2b — De AP-agent krijgt toegang tot de contracten ("agentic RAG").

Verschil met day2_ask.py: daar zoekt de CODE altijd eerst en geeft de fragmenten aan het model.
Hier is zoeken een TOOL: het model beslist zelf WANNEER en WAAROP het zoekt.
Bijvoorbeeld: alleen bij een prijsafwijking, en dan gericht in het contract van die leverancier.

Belangrijk ontwerpprincipe: het contract INFORMEERT de mens, maar BESLIST niet.
check_invoice blijft de enige poort naar goedkeuring. De agent mag het contract gebruiken
om een betere reden te schrijven, niet om een regel te omzeilen.

Draaien:  rm -f data/invoice_ledger.csv approval_queue.jsonl
          python day2_agent.py
"""

import os

from agents import Agent, Runner, function_tool

import rag
from day1_agents_sdk import (BatchResult, approve_for_payment, check_invoice,
                             flag_for_approval, list_invoices, read_invoice)

MODEL = os.getenv("MODEL", "gpt-4.1-mini")

CONTRACT_FILES = {  # welk contract hoort bij welke leverancier (in het echt: uit het ERP)
    "BE0412345678": "brouwers_logistiek.md",
    "BE0498765432": "delcourt_emballages.md",
    "DE811234567": "techparts_gmbh.md",
}


@function_tool
def search_contract(vat_number: str, question: str) -> list[dict]:
    """Zoek in het contract van één leverancier naar de relevante artikels.

    Args:
        vat_number: BTW-nummer van de leverancier, bv. BE0498765432.
        question: Waarnaar je zoekt, bv. "Mag de prijs verhoogd worden?" of "Wijziging bankgegevens".
    """
    source = CONTRACT_FILES.get(vat_number.replace(" ", "").replace(".", "").upper())
    if source is None:
        return [{"error": f"geen contract gevonden voor BTW-nummer {vat_number}"}]
    hits = rag.search(question, k=2, source=source)
    # Alleen teruggeven wat het model nodig heeft: geen scores, geen vectoren.
    return [{"contract": h["title"], "artikel": h["section"], "tekst": h["text"]} for h in hits]


ap_agent = Agent(
    name="AP-assistent met contracten",
    model=MODEL,
    instructions=(
        "Je bent een accounts-payable assistent voor een Belgisch KMO. "
        "Verwerk elke factuur in de inbox: lees ze (NL/FR/DE/EN), haal factuurnummer, BTW-nummer, "
        "IBAN, PO-nummer, aantal en eenheidsprijs eruit en roep check_invoice aan. "
        "Alles ok: approve_for_payment. "
        "Meldt check_invoice een probleem, zoek dan met search_contract in het contract van die "
        "leverancier wat er over dat probleem staat, en roep daarna flag_for_approval aan. "
        "Schrijf in de reden: het probleem, wat het contract erover zegt, met artikelnummer, "
        "en wat de financeafdeling concreet moet doen. "
        "Je mag NOOIT een factuur goedkeuren op basis van het contract als check_invoice een probleem meldt. "
        "Schrijf de redenen in het Nederlands."
    ),
    tools=[list_invoices, read_invoice, check_invoice, search_contract,
           flag_for_approval, approve_for_payment],
    output_type=BatchResult,
)


if __name__ == "__main__":
    result = Runner.run_sync(ap_agent, "Verwerk alle facturen in de inbox.", max_turns=40)
    for d in result.final_output.decisions:
        print(f"\n{d.invoice}  [{d.decision}]  {d.vendor}\n   {d.reason}")

    usage = result.context_wrapper.usage
    n = len(result.final_output.decisions)
    print(f"\nModelaanroepen: {usage.requests}")
    print(f"Input tokens: {usage.input_tokens}  Output tokens: {usage.output_tokens}")
    print(f"Per factuur: {usage.input_tokens / n:.0f} input en {usage.output_tokens / n:.0f} output tokens")
