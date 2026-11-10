"""
Bedrijfslogica voor de AP-agent (accounts payable).

Dit zijn gewone Python-functies. Ze weten niets van LLM's of frameworks.
Dat is bewust: de tools zijn het "echte" werk. Het framework (of je eigen
lus) beslist alleen WANNEER welke tool met WELKE argumenten wordt aangeroepen.
In een echt portfoliobedrijf vervang je de CSV's door een ERP-koppeling
(Exact, Odoo, SAP B1, ...) en blijft de rest gelijk.
"""

import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
APPROVAL_QUEUE = Path(__file__).parent / "approval_queue.jsonl"
LEDGER = DATA / "invoice_ledger.csv"  # "het ERP": wat al goedgekeurd en geboekt is
LEDGER_FIELDS = ["invoice_number", "vendor_id", "po_number", "quantity", "amount"]

PRICE_TOLERANCE = 0.02  # 2% afwijking op eenheidsprijs is nog OK
APPROVAL_LIMIT = 5000  # boven dit bedrag (excl. btw) altijd een mens

# Facturen die in DEZE run succesvol gecontroleerd zijn, en dus goedgekeurd MOGEN worden.
# approve_for_payment kijkt hierin: zonder geslaagde controle geen goedkeuring.
_checked_ok: dict[str, dict] = {}


def _read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_ledger() -> list[dict]:
    return _read_csv(LEDGER.name) if LEDGER.exists() else []


def _normalize_invoice(invoice: str) -> str:
    """'inv-1001.txt' en 'INV-1001' moeten als hetzelfde factuurnummer tellen."""
    return invoice.strip().upper().removesuffix(".TXT")


def list_invoices() -> list[str]:
    """Geef de bestandsnamen van alle facturen in de inbox."""
    return sorted(p.name for p in (DATA / "invoices").glob("*.txt"))


def read_invoice(filename: str) -> str:
    """Lees de ruwe tekst van één factuur."""
    path = (DATA / "invoices" / filename).resolve()
    if path.parent != (DATA / "invoices").resolve() or not path.exists():
        return f"ERROR: factuur '{filename}' niet gevonden"
    return path.read_text(encoding="utf-8")


def lookup_vendor(vat_number: str) -> dict:
    """Zoek een leverancier op via BTW-nummer in de leveranciersmaster."""
    vat = vat_number.replace(" ", "").replace(".", "").upper()
    for v in _read_csv("vendors.csv"):
        if v["vat_number"] == vat:
            return v
    return {"error": f"geen leverancier met BTW-nummer {vat_number}"}


def get_purchase_order(po_number: str) -> dict:
    """Haal een bestelbon (PO) op."""
    for po in _read_csv("purchase_orders.csv"):
        if po["po_number"] == po_number.strip().upper():
            return po
    return {"error": f"bestelbon {po_number} niet gevonden"}


def check_invoice(invoice_number: str, vat_number: str, iban: str, po_number: str,
                  quantity: float, unit_price: float) -> dict:
    """
    Deterministische controle van een factuur tegen leveranciersmaster, PO en grootboek.
    Belangrijk ontwerpprincipe: de LLM LEEST de factuur en haalt de velden eruit,
    maar de CONTROLE zelf gebeurt in code. Rekenen en regels laat je niet aan een LLM over.
    """
    invoice = _normalize_invoice(invoice_number)
    _checked_ok.pop(invoice, None)  # een oude geslaagde controle telt niet meer
    issues = []

    vendor = lookup_vendor(vat_number)
    if "error" in vendor:
        return {"ok": False, "issues": [vendor["error"]]}
    if vendor["status"] != "active":
        issues.append(f"leverancier {vendor['name']} heeft status '{vendor['status']}'")
    if iban.replace(" ", "").upper() != vendor["iban"]:
        issues.append("IBAN op factuur wijkt af van leveranciersmaster (mogelijke fraude)")

    ledger = _read_ledger()

    # NIEUW: zelfde leverancier + zelfde factuurnummer = dubbele factuur
    if any(r["invoice_number"] == invoice and r["vendor_id"] == vendor["vendor_id"]
           for r in ledger):
        issues.append(f"factuur {invoice} van {vendor['name']} is al geboekt (dubbele factuur)")

    po = get_purchase_order(po_number)
    if "error" in po:
        issues.append(po["error"])
    else:
        if po["vendor_id"] != vendor["vendor_id"]:
            issues.append("bestelbon hoort bij een andere leverancier")

        # NIEUW: niet alleen deze factuur, maar alles wat al op deze PO gefactureerd is
        already = sum(float(r["quantity"]) for r in ledger
                      if r["po_number"] == po["po_number"])
        if already + float(quantity) > float(po["quantity"]):
            issues.append(f"PO {po['po_number']}: al {already:g} gefactureerd, met deze factuur "
                          f"erbij {already + float(quantity):g}, maar besteld is {po['quantity']}")

        po_price = float(po["unit_price"])
        if abs(float(unit_price) - po_price) / po_price > PRICE_TOLERANCE:
            issues.append(f"eenheidsprijs {unit_price} wijkt af van PO-prijs {po_price}")

    total = float(quantity) * float(unit_price)
    if total > APPROVAL_LIMIT:
        issues.append(f"totaal {total:.2f} EUR ligt boven de goedkeuringslimiet "
                      f"van {APPROVAL_LIMIT} EUR")

    if not issues:  # onthouden: deze factuur mag nu goedgekeurd worden
        _checked_ok[invoice] = {"invoice_number": invoice, "vendor_id": vendor["vendor_id"],
                                "po_number": po["po_number"], "quantity": quantity,
                                "amount": round(total, 2)}

    return {"ok": not issues, "vendor": vendor["name"], "issues": issues}


def flag_for_approval(invoice: str, reason: str) -> str:
    """Zet een factuur in de wachtrij voor menselijke goedkeuring."""
    with open(APPROVAL_QUEUE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"invoice": invoice, "reason": reason}, ensure_ascii=False) + "\n")
    return f"{invoice} in goedkeuringswachtrij gezet"


def approve_for_payment(invoice: str) -> str:
    """Keur een factuur goed en boek ze in het grootboek (in het echt: in het ERP)."""
    key = _normalize_invoice(invoice)
    details = _checked_ok.pop(key, None)

    # NIEUW: het slot. Zonder geslaagde check_invoice in deze run: geen goedkeuring.
    if details is None:
        return (f"GEWEIGERD: {key} is niet succesvol gecontroleerd met check_invoice. "
                f"Goedkeuren kan alleen na een geslaagde controle; zet de factuur anders "
                f"in de goedkeuringswachtrij.")

    new_file = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(details)
    return f"{key} goedgekeurd voor betaling en geboekt"
