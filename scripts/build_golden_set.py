import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import EXTRACTION_INSTRUCTION


DEFAULT_RECORDS_PER_CATEGORY = 20
DEFAULT_SEED = 20260507

NAMES = [
    "Ana Silva",
    "Bruno Costa",
    "Carla Mendes",
    "Daniel Rocha",
    "Evelyn Brooks",
    "Felipe Santos",
    "Grace Turner",
    "Helena Martins",
    "Igor Almeida",
    "Janice Fernandez",
    "Jordan Miles",
    "Lee Wong",
    "Marta Costa",
    "Nadia Lima",
    "Oscar Pereira",
    "Priya Shah",
    "Rafael Nunes",
    "Sofia Marin",
    "Thomas Blake",
    "Yara Oliveira",
]

EN_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

PT_MONTHS = [
    "Janeiro",
    "Fevereiro",
    "Marco",
    "Abril",
    "Maio",
    "Junho",
    "Julho",
    "Agosto",
    "Setembro",
    "Outubro",
    "Novembro",
    "Dezembro",
]


@dataclass
class PayrollCase:
    employee_name: str
    gross_pay: float
    tax: float | None
    deductions: float | None
    net_pay: float
    pay_period: str
    invoice_number: str | None


def _money_us(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _money_br(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}{suffix}"


def _money_eu(value: float | None, suffix: str = " EUR") -> str:
    if value is None:
        return "N/A"
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted}{suffix}"


def _corrupt_text(text: str, rng: random.Random, corruption_rate: float = 0.08) -> str:
    chars = list(text)
    eligible = [i for i, char in enumerate(chars) if char.isalpha()]
    count = max(1, int(len(eligible) * corruption_rate))
    for idx in rng.sample(eligible, min(count, len(eligible))):
        chars[idx] = rng.choice("!@#$%^&*+")
    return "".join(chars)


def _make_case(rng: random.Random, idx: int, *, low=False, high=False, missing_invoice=False, missing_deductions=False) -> PayrollCase:
    if low:
        gross = round(rng.uniform(420, 950), 2)
    elif high:
        gross = round(rng.uniform(25000, 65000), 2)
    else:
        gross = round(rng.uniform(2200, 9800), 2)

    tax = round(gross * rng.uniform(0.08, 0.24), 2)
    deductions = None if missing_deductions else round(rng.uniform(40, min(850, gross * 0.12)), 2)
    net = round(gross - tax - (deductions or 0), 2)
    month_index = rng.randrange(12)

    return PayrollCase(
        employee_name=NAMES[idx % len(NAMES)],
        gross_pay=gross,
        tax=tax,
        deductions=deductions,
        net_pay=net,
        pay_period=f"{EN_MONTHS[month_index]} {rng.randint(2023, 2026)}",
        invoice_number=None if missing_invoice else str(10000 + idx * 37 + rng.randint(0, 899)),
    )


def _record(case: PayrollCase, text: str, category: str, noise_level: float, idx: int) -> dict:
    return {
        "instruction": EXTRACTION_INSTRUCTION,
        "input": text,
        "output": json.dumps(asdict(case)),
        "template_id": category,
        "golden_category": category,
        "difficulty": "ood",
        "noise_level": noise_level,
        "case_id": f"{category}_{idx:03d}",
    }


def _build_text(category: str, case: PayrollCase, rng: random.Random) -> tuple[str, float]:
    if category == "golden_pt_br_currency":
        month = rng.choice(PT_MONTHS)
        case.pay_period = f"{month} {case.pay_period.split()[-1]}"
        text = (
            "HOLERITE\n"
            f"Funcionario: {case.employee_name}\n"
            f"Periodo: {case.pay_period}\n"
            f"Salario Bruto: {_money_br(case.gross_pay)}\n"
            f"Imposto: {_money_br(case.tax)}\n"
            f"Descontos: {_money_br(case.deductions)}\n"
            f"Liquido: {_money_br(case.net_pay)}\n"
            f"Documento: {case.invoice_number}"
        )
        return text, 0.0

    if category == "golden_eu_format":
        month = rng.choice(PT_MONTHS)
        case.pay_period = f"{month} {case.pay_period.split()[-1]}"
        text = (
            "SALARY NOTE\n"
            f"{case.employee_name} // ref {case.invoice_number} // period {case.pay_period}\n"
            f"Bruto {_money_eu(case.gross_pay)}\n"
            f"Impostos {_money_eu(case.tax)}\n"
            f"Outros {_money_eu(case.deductions)}\n"
            f"Liquido {_money_eu(case.net_pay)}"
        )
        return text, 0.0

    if category == "golden_missing_invoice":
        text = (
            "PAYSLIP\n"
            f"Name: {case.employee_name}\n"
            f"Period: {case.pay_period}\n"
            f"Gross: {_money_us(case.gross_pay)}\n"
            f"Tax Amount: {_money_us(case.tax)}\n"
            f"Deductions: {_money_us(case.deductions)}\n"
            f"Total Net: {_money_us(case.net_pay)}"
        )
        return text, 0.0

    if category == "golden_missing_deductions":
        text = (
            "PAYSLIP\n"
            f"Name: {case.employee_name}\n"
            f"ID: {case.invoice_number}\n"
            f"Dates: {case.pay_period}\n"
            f"Earnings: {case.gross_pay:.2f}\n"
            f"Taxes: {case.tax:.2f}\n"
            f"Payable: {case.net_pay:.2f}"
        )
        return text, 0.0

    if category == "golden_heavy_ocr":
        clean = (
            f"Earnings Statement for {case.employee_name}. Invoice {case.invoice_number} "
            f"for period {case.pay_period}. Gross pay was {case.gross_pay:.2f}, "
            f"taxes were {case.tax:.2f}, deductions were {case.deductions:.2f}, "
            f"net was {case.net_pay:.2f}."
        )
        return _corrupt_text(clean, rng), 0.08

    if category == "golden_unseen_template":
        text = (
            "Compensation Memo\n"
            f"Worker {case.employee_name}\n"
            f"Reference {case.invoice_number}\n"
            f"Window {case.pay_period}\n"
            f"Gross compensation {case.gross_pay:.2f}\n"
            f"Tax held {case.tax:.2f}\n"
            f"Voluntary deductions {case.deductions:.2f}\n"
            f"Net settlement {case.net_pay:.2f}"
        )
        return text, 0.0

    if category == "golden_low_gross":
        text = (
            "Payroll excerpt\n"
            f"Employee: {case.employee_name}\n"
            f"Invoice #: {case.invoice_number}\n"
            f"Period: {case.pay_period}\n"
            f"Gross: {_money_us(case.gross_pay)}\n"
            f"Tax Amount: {_money_us(case.tax)}\n"
            f"Deductions: {_money_us(case.deductions)}\n"
            f"Total Net: {_money_us(case.net_pay)}"
        )
        return text, 0.0

    if category == "golden_high_gross":
        text = (
            "Statement of Earnings\n"
            f"Name: {case.employee_name}\n"
            f"Invoice #: {case.invoice_number}\n"
            f"Period: {case.pay_period}\n"
            f"Gross: {_money_us(case.gross_pay)}\n"
            f"Tax Amount: {_money_us(case.tax)}\n"
            f"Deductions: {_money_us(case.deductions)}\n"
            f"Total Net: {_money_us(case.net_pay)}"
        )
        return text, 0.0

    if category == "golden_reference_noise":
        text = (
            "ADP EXPORT ROW\n"
            f"emp={case.employee_name}; ref=Ref# {case.invoice_number}; cycle={case.pay_period};\n"
            f"gross_pay={case.gross_pay:.2f}; tax_withheld={case.tax:.2f}; "
            f"other_ded={case.deductions:.2f}; net_pay={case.net_pay:.2f}; status=posted"
        )
        return text, 0.015

    if category == "golden_compact_table":
        text = (
            "PAYROLL GRID\n"
            "employee | period | document | gross | tax | deduct | net\n"
            f"{case.employee_name} | {case.pay_period} | {case.invoice_number} | "
            f"{case.gross_pay:.2f} | {case.tax:.2f} | {case.deductions:.2f} | {case.net_pay:.2f}"
        )
        return text, 0.0

    raise ValueError(f"Unknown golden category: {category}")


def build_golden_records(records_per_category: int = DEFAULT_RECORDS_PER_CATEGORY, seed: int = DEFAULT_SEED) -> list[dict]:
    rng = random.Random(seed)
    categories = [
        "golden_pt_br_currency",
        "golden_eu_format",
        "golden_missing_invoice",
        "golden_missing_deductions",
        "golden_heavy_ocr",
        "golden_unseen_template",
        "golden_low_gross",
        "golden_high_gross",
        "golden_reference_noise",
        "golden_compact_table",
    ]

    records = []
    for category in categories:
        for offset in range(records_per_category):
            idx = len(records)
            case = _make_case(
                rng,
                idx,
                low=category == "golden_low_gross",
                high=category == "golden_high_gross",
                missing_invoice=category == "golden_missing_invoice",
                missing_deductions=category == "golden_missing_deductions",
            )
            text, noise_level = _build_text(category, case, rng)
            records.append(_record(case, text, category, noise_level, offset))

    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_golden_set(
    path: str = "data/golden.jsonl",
    records_per_category: int = DEFAULT_RECORDS_PER_CATEGORY,
    seed: int = DEFAULT_SEED,
) -> None:
    write_jsonl(path, build_golden_records(records_per_category=records_per_category, seed=seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a categorized OOD golden set for DocTune.")
    parser.add_argument("--output", default="data/golden.jsonl")
    parser.add_argument("--records-per-category", type=int, default=DEFAULT_RECORDS_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write_golden_set(
        path=args.output,
        records_per_category=args.records_per_category,
        seed=args.seed,
    )
