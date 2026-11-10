"""
DAG 1a — Een agent zonder framework.

Een "agent" is niet meer dan dit:
  1. Je stuurt het model een vraag + een lijst tools (naam, beschrijving, JSON-schema).
  2. Het model antwoordt ofwel met tekst (klaar), ofwel met "roep tool X aan met args Y".
  3. Jij voert die tool uit in je eigen code en stuurt het resultaat terug.
  4. Herhaal tot het model klaar is.

Alles wat LangGraph, de OpenAI Agents SDK of de Claude Agent SDK doen,
is een (veel) robuustere versie van deze ~40 regels lus.

Draaien:  python day1_raw_loop.py
"""

import json
import os

from openai import OpenAI

import tools

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_STEPS = 25  # vangnet: een agent mag nooit eindeloos doorlopen
client = OpenAI()  # leest OPENAI_API_KEY uit je omgeving

SYSTEM = """Je bent een accounts-payable assistent voor een Belgisch KMO.
Voor elke factuur in de inbox:
- lees de factuur (die kan in NL, FR, DE of EN zijn),
- haal BTW-nummer, IBAN, PO-nummer, aantal en eenheidsprijs eruit,
- roep check_invoice aan,
- is alles ok: approve_for_payment; anders: flag_for_approval met een korte reden.
Rond af met een korte samenvatting per factuur in het Nederlands."""

# Het model ziet je Python-code niet. Het ziet alleen deze beschrijvingen.
# De kwaliteit van naam + description bepaalt in grote mate hoe goed de agent werkt.
TOOLS = [
    {"type": "function", "function": {
        "name": "list_invoices",
        "description": "Lijst alle facturen in de inbox.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_invoice",
        "description": "Lees de volledige tekst van een factuur.",
        "parameters": {"type": "object",
                       "properties": {"filename": {"type": "string"}},
                       "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "check_invoice",
        "description": "Controleer factuurgegevens tegen leveranciersmaster en bestelbon. "
                       "Getallen als decimale getallen met punt (1.15, niet 1,15).",
        "parameters": {"type": "object",
                       "properties": {
                           "invoice_number": {"type": "string"},
                           "vat_number": {"type": "string"},
                           "iban": {"type": "string"},
                           "po_number": {"type": "string"},
                           "quantity": {"type": "number"},
                           "unit_price": {"type": "number"}},
                       "required": ["invoice_number", "vat_number", "iban", "po_number", "quantity", "unit_price"]}}},
    {"type": "function", "function": {
        "name": "flag_for_approval",
        "description": "Zet een factuur in de wachtrij voor menselijke controle.",
        "parameters": {"type": "object",
                       "properties": {"invoice": {"type": "string"},
                                      "reason": {"type": "string"}},
                       "required": ["invoice", "reason"]}}},
    {"type": "function", "function": {
        "name": "approve_for_payment",
        "description": "Keur een correcte factuur goed voor betaling.",
        "parameters": {"type": "object",
                       "properties": {"invoice": {"type": "string"}},
                       "required": ["invoice"]}}},
]

# Koppeling tussen de naam die het model gebruikt en de echte functie.
REGISTRY = {
    "list_invoices": tools.list_invoices,
    "read_invoice": tools.read_invoice,
    "check_invoice": tools.check_invoice,
    "flag_for_approval": tools.flag_for_approval,
    "approve_for_payment": tools.approve_for_payment,
}


def run(task: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = response.choices[0].message
        messages.append(msg)  # het antwoord van het model hoort in de geschiedenis

        if not msg.tool_calls:  # geen tool meer nodig: het model is klaar
            return msg.content

        for call in msg.tool_calls:  # het model kan meerdere tools tegelijk vragen
            args = json.loads(call.function.arguments or "{}")
            print(f"  [stap {step}] {call.function.name}({args})")
            try:
                result = REGISTRY[call.function.name](**args)
            except Exception as e:  # fouten teruggeven, zodat het model zich kan herstellen
                result = f"ERROR: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "Gestopt: maximum aantal stappen bereikt."


if __name__ == "__main__":
    print(run("Verwerk alle facturen in de inbox."))
