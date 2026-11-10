"""
DAG 1b — Dezelfde agent met de OpenAI Agents SDK.

Vergelijk met day1_raw_loop.py:
  - Geen handgeschreven JSON-schema's meer: @function_tool leest de
    type hints en de docstring van je functie.
  - Geen eigen lus: Runner doet de lus, foutafhandeling en max_turns.
  - Gratis tracing: elke run is zichtbaar op platform.openai.com/traces.
  - output_type: het eindresultaat is gestructureerde data (Pydantic) i.p.v. vrije tekst,
    zodat een ander systeem (ERP, dashboard) het kan inlezen.

Draaien:  python day1_agents_sdk.py
"""

import os

from agents import Agent, Runner, function_tool
from pydantic import BaseModel

import tools

MODEL = os.getenv("MODEL", "gpt-4.1-mini")


# Dunne wrappers: de docstring wordt de tool-beschrijving die het model ziet.
@function_tool
def list_invoices() -> list[str]:
    """Lijst alle facturen in de inbox."""
    return tools.list_invoices()


@function_tool
def read_invoice(filename: str) -> str:
    """Lees de volledige tekst van een factuur."""
    return tools.read_invoice(filename)


@function_tool
def check_invoice(invoice_number: str, vat_number: str, iban: str, po_number: str,
                  quantity: float, unit_price: float) -> dict:
    """Controleer factuurgegevens tegen leveranciersmaster, bestelbon en eerder geboekte facturen.

    Args:
        invoice_number: Factuurnummer zoals op de factuur, bv. INV-1001.
        vat_number: BTW-nummer van de leverancier, bv. BE0412345678.
        iban: IBAN zoals op de factuur.
        po_number: Nummer van de bestelbon, bv. PO-2026-0412.
        quantity: Gefactureerd aantal.
        unit_price: Eenheidsprijs als decimaal getal met punt (1.15, niet 1,15).
    """
    return tools.check_invoice(invoice_number, vat_number, iban, po_number, quantity, unit_price)


@function_tool
def flag_for_approval(invoice: str, reason: str) -> str:
    """Zet een factuur in de wachtrij voor menselijke controle."""
    return tools.flag_for_approval(invoice, reason)


@function_tool
def approve_for_payment(invoice: str) -> str:
    """Keur een correcte factuur goed voor betaling."""
    return tools.approve_for_payment(invoice)


class InvoiceDecision(BaseModel):
    invoice: str
    vendor: str
    decision: str  # "approved" of "flagged"
    reason: str


class BatchResult(BaseModel):
    decisions: list[InvoiceDecision]


ap_agent = Agent(
    name="AP-assistent",
    model=MODEL,
    instructions=(
        "Je bent een accounts-payable assistent voor een Belgisch KMO. "
        "Verwerk elke factuur in de inbox: lees ze (NL/FR/DE/EN), haal BTW-nummer, IBAN, "
        "PO-nummer, aantal en eenheidsprijs eruit en roep check_invoice aan. "
        "Alles ok: approve_for_payment. Anders: flag_for_approval met een korte reden. "
        "Schrijf de redenen in het Nederlands."
    ),
    tools=[list_invoices, read_invoice, check_invoice, flag_for_approval, approve_for_payment],
    output_type=BatchResult,
)


if __name__ == "__main__":
    result = Runner.run_sync(ap_agent, "Verwerk alle facturen in de inbox.", max_turns=25)
    for d in result.final_output.decisions:
        print(f"{d.invoice:<12} {d.decision:<9} {d.vendor:<25} {d.reason}")

    usage = result.context_wrapper.usage
    n = len(result.final_output.decisions)
    print(f"\nModelaanroepen: {usage.requests}")
    print(f"Input tokens: {usage.input_tokens}  Output tokens: {usage.output_tokens}")
    print(f"Per factuur: {usage.input_tokens / n:.0f} input en {usage.output_tokens / n:.0f} output tokens")
