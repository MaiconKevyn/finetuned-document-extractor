from src.postprocess import normalize_extraction, normalize_invoice, parse_money


class TestParseMoney:
    def test_parses_us_currency(self):
        assert parse_money("$5,000.25") == 5000.25

    def test_parses_br_currency(self):
        assert parse_money("R$ 5.000,25") == 5000.25

    def test_parses_eu_currency(self):
        assert parse_money("5.000,25 EUR") == 5000.25

    def test_missing_money_returns_none(self):
        assert parse_money("N/A") is None


class TestNormalizeInvoice:
    def test_strips_reference_prefix(self):
        assert normalize_invoice("Ref# 39109") == "39109"

    def test_missing_invoice_returns_none(self):
        assert normalize_invoice("N/A") is None


class TestNormalizeExtraction:
    def test_normalizes_model_strings(self):
        raw = {
            "employee_name": " Jane Doe ",
            "gross_pay": "R$ 5.000,00",
            "tax": "R$ 750,00",
            "deductions": "R$ 200,00",
            "net_pay": "R$ 4.050,00",
            "pay_period": " Marco 2025 ",
            "invoice_number": "Ref# 84201",
        }
        assert normalize_extraction(raw) == {
            "employee_name": "Jane Doe",
            "gross_pay": 5000.0,
            "tax": 750.0,
            "deductions": 200.0,
            "net_pay": 4050.0,
            "pay_period": "Marco 2025",
            "invoice_number": "84201",
        }

    def test_infers_missing_deductions_from_business_rule(self):
        raw = {
            "employee_name": "Jane Doe",
            "gross_pay": "5000.00",
            "tax": "750.00",
            "deductions": None,
            "net_pay": "4050.00",
            "pay_period": "March 2025",
            "invoice_number": "84201",
        }
        assert normalize_extraction(raw)["deductions"] == 200.0

    def test_none_prediction_stays_none(self):
        assert normalize_extraction(None) is None
