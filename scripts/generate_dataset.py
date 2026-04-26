import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faker import Faker
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test envs
    def tqdm(iterable):
        return iterable

from src.prompts import EXTRACTION_INSTRUCTION

SEED = 42
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1

random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


@dataclass
class PayStub:
    employee_name: str
    gross_pay: float
    tax: float
    deductions: float
    net_pay: float
    pay_period: str
    invoice_number: str


def generate_paystub_text(data: PayStub) -> tuple[str, str]:
    templates = [
        (
            "key_value",
            (
                f"Employee: {data.employee_name}\n"
                f"Invoice #: {data.invoice_number}\n"
                f"Period: {data.pay_period}\n"
                f"Gross: ${data.gross_pay:.2f}\n"
                f"Tax Amount: ${data.tax:.2f}\n"
                f"Deductions: ${data.deductions:.2f}\n"
                f"Total Net: ${data.net_pay:.2f}"
            ),
        ),
        (
            "abbreviated",
            (
                f"PAYSLIP\n"
                f"Name: {data.employee_name}\n"
                f"ID: {data.invoice_number}\n"
                f"Dates: {data.pay_period}\n"
                f"Earnings: {data.gross_pay:.2f}\n"
                f"Taxes: {data.tax:.2f}\n"
                f"Other: {data.deductions:.2f}\n"
                f"Payable: {data.net_pay:.2f}"
            ),
        ),
        (
            "narrative",
            (
                f"Earnings Statement for {data.employee_name}. "
                f"Invoice {data.invoice_number} for period {data.pay_period}. "
                f"Your gross pay was {data.gross_pay:.2f} with taxes of {data.tax:.2f} "
                f"and deductions of {data.deductions:.2f}. "
                f"Resulting net: {data.net_pay:.2f}."
            ),
        ),
        (
            "table",
            (
                "STATEMENT OF EARNINGS\n"
                f"{data.employee_name} | Ref: {data.invoice_number}\n"
                f"{data.pay_period}\n"
                "| Gross       | Tax        | Deductions | Net        |\n"
                f"| {data.gross_pay:.2f} | {data.tax:.2f} | {data.deductions:.2f} | {data.net_pay:.2f} |"
            ),
        ),
        (
            "indented",
            (
                "Pay Summary\n"
                f"To: {data.employee_name}\n"
                f"Doc: {data.invoice_number} [{data.pay_period}]\n"
                f"Base Salary:          {data.gross_pay:.2f}\n"
                f"  (-) Tax withheld:   {data.tax:.2f}\n"
                f"  (-) Other deductions: {data.deductions:.2f}\n"
                f"  (=) Amount due:     {data.net_pay:.2f}"
            ),
        ),
    ]
    return random.choice(templates)


def add_noise(text: str, corruption_rate: float = 0.02) -> tuple[str, float]:
    """Simulate OCR corruption and return the noisy text plus actual corruption rate."""
    noise_chars = "!@#$%^&*()_+"
    chars = list(text)

    eligible = [i for i, c in enumerate(chars) if c not in "0123456789\n "]
    if not eligible:
        return text, 0.0

    noise_count = max(1, int(len(eligible) * corruption_rate))
    indices_to_corrupt = random.sample(eligible, min(noise_count, len(eligible)))
    for idx in indices_to_corrupt:
        chars[idx] = random.choice(noise_chars)

    if random.random() > 0.5:
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, "\n  ")

    actual_noise_level = round(len(indices_to_corrupt) / len(eligible), 4)
    return "".join(chars), actual_noise_level


def split_dataset_stratified(
    dataset: list[dict],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave room for a non-negative test split")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in dataset:
        grouped[record["template_id"]].append(record)

    train_data: list[dict] = []
    val_data: list[dict] = []
    test_data: list[dict] = []

    for template_id in sorted(grouped):
        records = grouped[template_id]
        random.shuffle(records)

        total = len(records)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_data.extend(records[:train_count])
        val_data.extend(records[train_count:train_count + val_count])
        test_data.extend(records[train_count + val_count:])

    target_train = round(len(dataset) * train_ratio)
    target_val = round(len(dataset) * val_ratio)
    target_test = len(dataset) - target_train - target_val

    def rebalance(source: list[dict], target: list[dict], expected_source: int, expected_target: int) -> None:
        while len(source) > expected_source and len(target) < expected_target:
            target.append(source.pop())

    rebalance(test_data, val_data, target_test, target_val)
    rebalance(test_data, train_data, target_test, target_train)
    rebalance(val_data, train_data, target_val, target_train)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    return train_data, val_data, test_data


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in records:
            f.write(json.dumps(entry) + "\n")


def main(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> None:
    dataset = []
    print(f"Generating {num_samples} noisy samples...")

    for _ in tqdm(range(num_samples)):
        gross = round(random.uniform(2000, 10000), 2)
        tax = round(gross * random.uniform(0.1, 0.25), 2)
        deductions = round(random.uniform(50, 500), 2)
        net = round(gross - tax - deductions, 2)

        data = PayStub(
            employee_name=fake.name(),
            gross_pay=gross,
            tax=tax,
            deductions=deductions,
            net_pay=net,
            pay_period=f"{fake.month_name()} {random.randint(2023, 2026)}",
            invoice_number=str(random.randint(10000, 99999)),
        )

        template_id, clean_text = generate_paystub_text(data)
        noisy_text, noise_level = add_noise(clean_text)

        dataset.append(
            {
                "instruction": EXTRACTION_INSTRUCTION,
                "input": noisy_text,
                "output": json.dumps(asdict(data)),
                "template_id": template_id,
                "noise_level": noise_level,
            }
        )

    train_data, val_data, test_data = split_dataset_stratified(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    os.makedirs("data", exist_ok=True)
    write_jsonl("data/train.jsonl", train_data)
    write_jsonl("data/val.jsonl", val_data)
    write_jsonl("data/test.jsonl", test_data)

    print(
        f"Done. Train: {len(train_data)} samples | "
        f"Val: {len(val_data)} samples | Test: {len(test_data)} samples"
    )
    print("Saved to data/train.jsonl, data/val.jsonl and data/test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic train/val/test datasets.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    args = parser.parse_args()
    main(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
