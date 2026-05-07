import re
from typing import Any


NUMERIC_FIELDS = ("gross_pay", "tax", "deductions", "net_pay")
OUTPUT_FIELDS = (
    "employee_name",
    "gross_pay",
    "tax",
    "deductions",
    "net_pay",
    "pay_period",
    "invoice_number",
)
MISSING_VALUES = {None, "", "null", "none", "n/a", "na", "not available", "missing"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    return False


def parse_money(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if not isinstance(value, str):
        return None

    match = re.search(r"-?[\d.,]+", value.replace(" ", ""))
    if not match:
        return None

    token = match.group(0)
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(".", "").replace(",", ".")

    try:
        return round(float(token), 2)
    except ValueError:
        return None


def normalize_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def normalize_invoice(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    digits = re.findall(r"\d+", text)
    if not digits:
        return None if text.lower() in MISSING_VALUES else text
    return "".join(digits)


def _infer_missing_arithmetic(record: dict[str, Any]) -> None:
    gross = record.get("gross_pay")
    tax = record.get("tax")
    deductions = record.get("deductions")
    net = record.get("net_pay")

    if deductions is None and None not in (gross, tax, net):
        inferred = round(gross - tax - net, 2)
        if inferred >= 0:
            record["deductions"] = inferred
        return

    if net is None and None not in (gross, tax, deductions):
        inferred = round(gross - tax - deductions, 2)
        if inferred >= 0:
            record["net_pay"] = inferred
        return

    if tax is None and None not in (gross, deductions, net):
        inferred = round(gross - deductions - net, 2)
        if inferred >= 0:
            record["tax"] = inferred


def normalize_extraction(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None

    normalized = {
        "employee_name": normalize_text(record.get("employee_name")),
        "gross_pay": parse_money(record.get("gross_pay")),
        "tax": parse_money(record.get("tax")),
        "deductions": parse_money(record.get("deductions")),
        "net_pay": parse_money(record.get("net_pay")),
        "pay_period": normalize_text(record.get("pay_period")),
        "invoice_number": normalize_invoice(record.get("invoice_number")),
    }
    _infer_missing_arithmetic(normalized)
    return {field: normalized.get(field) for field in OUTPUT_FIELDS}
