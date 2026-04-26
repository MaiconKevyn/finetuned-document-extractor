import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import EXTRACTION_INSTRUCTION


def build_golden_records() -> list[dict]:
    return [
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "HOLERITE\nFuncionaria: Ana Silva\nPeriodo: Marco 2025\n"
                "Salario Bruto: R$ 5.000,00\nImposto: R$ 750,00\nDescontos: R$ 200,00\n"
                "Liquido: R$ 4.050,00\nDocumento: 84201"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Ana Silva",
                    "gross_pay": 5000.0,
                    "tax": 750.0,
                    "deductions": 200.0,
                    "net_pay": 4050.0,
                    "pay_period": "Marco 2025",
                    "invoice_number": "84201",
                }
            ),
            "template_id": "golden_pt_br",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "PAYSLIP\nName: Jordan Miles\nPeriod: April 2025\nGross: $6200.00\n"
                "Tax Amount: $1240.00\nDeductions: $300.00\nTotal Net: $4660.00"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Jordan Miles",
                    "gross_pay": 6200.0,
                    "tax": 1240.0,
                    "deductions": 300.0,
                    "net_pay": 4660.0,
                    "pay_period": "April 2025",
                    "invoice_number": None,
                }
            ),
            "template_id": "golden_missing_invoice",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "E@rnings St@tement for Car@la Mendes. Inv#ice 34901 for per!od June 2023.\n"
                "Your gro$s pay was 3100.00 with ta^es of 465.00 and deduc#ions of 200.00.\n"
                "Resu@ting net: 2435.00."
            ),
            "output": json.dumps(
                {
                    "employee_name": "Carala Mendes",
                    "gross_pay": 3100.0,
                    "tax": 465.0,
                    "deductions": 200.0,
                    "net_pay": 2435.0,
                    "pay_period": "June 2023",
                    "invoice_number": "34901",
                }
            ),
            "template_id": "golden_heavy_ocr",
            "noise_level": 0.08,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "Compensation Memo\nEmployee Alex Turner\nReference 11007\nWindow February 2024\n"
                "Gross compensation 2800.00\nTax held 280.00\nDeductions 120.00\nNet settlement 2400.00"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Alex Turner",
                    "gross_pay": 2800.0,
                    "tax": 280.0,
                    "deductions": 120.0,
                    "net_pay": 2400.0,
                    "pay_period": "February 2024",
                    "invoice_number": "11007",
                }
            ),
            "template_id": "golden_unseen_template",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "Payroll excerpt\nEmployee: Lee Wong\nInvoice #: 45011\nPeriod: September 2025\n"
                "Gross: $520.00\nTax Amount: $52.00\nDeductions: $20.00\nTotal Net: $448.00"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Lee Wong",
                    "gross_pay": 520.0,
                    "tax": 52.0,
                    "deductions": 20.0,
                    "net_pay": 448.0,
                    "pay_period": "September 2025",
                    "invoice_number": "45011",
                }
            ),
            "template_id": "golden_low_gross",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "Statement of Earnings\nName: Evelyn Brooks\nInvoice #: 99102\nPeriod: December 2025\n"
                "Gross: $51200.00\nTax Amount: $12800.00\nDeductions: $950.00\nTotal Net: $37450.00"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Evelyn Brooks",
                    "gross_pay": 51200.0,
                    "tax": 12800.0,
                    "deductions": 950.0,
                    "net_pay": 37450.0,
                    "pay_period": "December 2025",
                    "invoice_number": "99102",
                }
            ),
            "template_id": "golden_high_gross",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "PAYSLIP\nName: Sofia Marin\nID: 87321\nDates: August 2025\n"
                "Earnings: 7100.00\nTaxes: 1420.00\nPayable: 5680.00"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Sofia Marin",
                    "gross_pay": 7100.0,
                    "tax": 1420.0,
                    "deductions": None,
                    "net_pay": 5680.0,
                    "pay_period": "August 2025",
                    "invoice_number": "87321",
                }
            ),
            "template_id": "golden_missing_deductions",
            "noise_level": 0.0,
        },
        {
            "instruction": EXTRACTION_INSTRUCTION,
            "input": (
                "SALARY NOTE\nMarta Costa // ref 77842 // period Agosto 2025\n"
                "Bruto 7.200,00 EUR\nImpostos 1.440,00 EUR\nOutros 320,00 EUR\nLiquido 5.440,00 EUR"
            ),
            "output": json.dumps(
                {
                    "employee_name": "Marta Costa",
                    "gross_pay": 7200.0,
                    "tax": 1440.0,
                    "deductions": 320.0,
                    "net_pay": 5440.0,
                    "pay_period": "Agosto 2025",
                    "invoice_number": "77842",
                }
            ),
            "template_id": "golden_eu_format",
            "noise_level": 0.0,
        },
    ]


def write_golden_set(path: str = "data/golden.jsonl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in build_golden_records():
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    write_golden_set()
