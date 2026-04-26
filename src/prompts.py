"""
Single source of truth for DocTune prompts.

Why this module exists: the Alpaca-style instruction and prompt format were duplicated
across src/api/main.py, scripts/evaluate.py, and scripts/generate_dataset.py.
Any divergence between training-time and inference-time prompts silently degrades accuracy.
PROMPT_VERSION lets MLflow, the model card, and results/*.json record exactly which
prompt variant a given model was trained and evaluated with.
"""

PROMPT_VERSION = "v1"

EXTRACTION_INSTRUCTION = (
    "Extract the following fields from the document text into a JSON format: "
    "employee_name, gross_pay, tax, deductions, net_pay, pay_period, invoice_number."
)


def build_alpaca_prompt(instruction: str, input_text: str, response: str = "") -> str:
    """Return an Alpaca-style prompt string.

    Leave ``response`` empty (default) to produce an open-ended prompt ready
    for model generation. Pass a non-empty ``response`` to include a full
    training example (used by the dataset formatter and few-shot prefixes).
    """
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{response}"
    )
