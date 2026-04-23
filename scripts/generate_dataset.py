import json
import random
import os
from faker import Faker
from pydantic import BaseModel
from tqdm import tqdm

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


class PayStub(BaseModel):
    employee_name: str
    gross_pay: float
    tax: float
    deductions: float
    net_pay: float
    pay_period: str
    invoice_number: str


def generate_paystub_text(data: PayStub) -> str:
    templates = [
        # Template 1: structured key-value
        (
            f"Employee: {data.employee_name}\n"
            f"Invoice #: {data.invoice_number}\n"
            f"Period: {data.pay_period}\n"
            f"Gross: ${data.gross_pay:.2f}\n"
            f"Tax Amount: ${data.tax:.2f}\n"
            f"Deductions: ${data.deductions:.2f}\n"
            f"Total Net: ${data.net_pay:.2f}"
        ),
        # Template 2: abbreviated labels
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
        # Template 3: prose / narrative
        (
            f"Earnings Statement for {data.employee_name}. "
            f"Invoice {data.invoice_number} for period {data.pay_period}. "
            f"Your gross pay was {data.gross_pay:.2f} with taxes of {data.tax:.2f} "
            f"and deductions of {data.deductions:.2f}. "
            f"Resulting net: {data.net_pay:.2f}."
        ),
        # Template 4: table-like format  [NEW]
        (
            f"STATEMENT OF EARNINGS\n"
            f"{data.employee_name} | Ref: {data.invoice_number}\n"
            f"{data.pay_period}\n"
            f"| Gross       | Tax        | Deductions | Net        |\n"
            f"| {data.gross_pay:.2f} | {data.tax:.2f} | {data.deductions:.2f} | {data.net_pay:.2f} |"
        ),
        # Template 5: indented pay summary  [NEW]
        (
            f"Pay Summary\n"
            f"To: {data.employee_name}\n"
            f"Doc: {data.invoice_number} [{data.pay_period}]\n"
            f"Base Salary:          {data.gross_pay:.2f}\n"
            f"  (-) Tax withheld:   {data.tax:.2f}\n"
            f"  (-) Other deductions: {data.deductions:.2f}\n"
            f"  (=) Amount due:     {data.net_pay:.2f}"
        ),
    ]
    return random.choice(templates)


def add_noise(text: str) -> str:
    """
    Simulates OCR noise on the document text.

    Rules:
    - Only corrupts alphabetic and punctuation characters — digits are NEVER
      touched. This keeps invoice_number and monetary values numerically intact
      while still degrading labels/words (the realistic OCR failure mode).
    - Corrupts ~2 % of eligible characters.
    - Randomly inserts a spurious line-break 50 % of the time.
    """
    noise_chars = "!@#$%^&*()_+"
    chars = list(text)

    # Only positions that are NOT digits, spaces, or newlines are eligible
    eligible = [i for i, c in enumerate(chars) if c not in "0123456789\n "]
    noise_count = max(1, int(len(eligible) * 0.02))

    indices_to_corrupt = random.sample(eligible, min(noise_count, len(eligible)))
    for idx in indices_to_corrupt:
        chars[idx] = random.choice(noise_chars)

    # Random spurious line-break (simulates OCR mis-segmentation)
    if random.random() > 0.5:
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, "\n  ")

    return "".join(chars)


def main(num_samples: int = 1000, val_ratio: float = 0.1) -> None:
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

        noisy_text = add_noise(generate_paystub_text(data))

        dataset.append({
            "instruction": (
                "Extract the following fields from the document text into a JSON format: "
                "employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number."
            ),
            "input": noisy_text,
            "output": data.model_dump_json(),
        })

    # --- Train / Val split (shuffled) ---
    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - val_ratio))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    os.makedirs("data", exist_ok=True)

    with open("data/train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    with open("data/val.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Train: {len(train_data)} samples | Val: {len(val_data)} samples")
    print("Saved to data/train.jsonl and data/val.jsonl")


if __name__ == "__main__":
    main()
