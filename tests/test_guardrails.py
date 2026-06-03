"""
Unit tests for GuardrailsManager — the anti-hallucination layer.

These tests are fully deterministic and do not require a live Gemini API key.
All Gemini model calls are mocked via unittest.mock.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.guardrails import GuardrailsManager, SAFE_REFUSAL, STAFF_COMMAND_DENIED, PRICE_REGEX, URL_REGEX
from app.chunking import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def payment_chunk():
    """A chunk whose content explicitly mentions the no-payment-link policy."""
    return [
        {
            "chunk": Chunk(
                id="kb_payment_methods",
                text=(
                    "Category: POLICIES\nTitle: Accepted Payment Methods and Link Policy\n"
                    "Content: We accept Visa, MasterCard and UPI at the front desk. "
                    "We do NOT generate online payment links."
                ),
                metadata={
                    "title": "Accepted Payment Methods and Link Policy",
                    "content": (
                        "We accept Visa, MasterCard and UPI at the front desk. "
                        "We do NOT generate online payment links."
                    )
                }
            ),
            "distance": 0.05,
            "similarity": 0.97
        }
    ]


@pytest.fixture
def contact_chunk():
    """A chunk that includes a known, grounded email address."""
    return [
        {
            "chunk": Chunk(
                id="kb_contact",
                text=(
                    "Category: CONTACTS\nTitle: Contact Info\n"
                    "Content: Email us at support@grandhorizonresort.com."
                ),
                metadata={
                    "title": "Contact Info",
                    "content": "Email us at support@grandhorizonresort.com."
                }
            ),
            "distance": 0.1,
            "similarity": 0.95
        }
    ]


# ---------------------------------------------------------------------------
# PRICE_REGEX Unit Tests
# ---------------------------------------------------------------------------
def test_price_regex_detects_dollar_sign():
    assert PRICE_REGEX.search("The room costs $150 per night.")


def test_price_regex_detects_inr_prefix():
    assert PRICE_REGEX.search("The fee is INR 500.")


def test_price_regex_detects_rupees_symbol():
    assert PRICE_REGEX.search("You owe ₹2500.")


def test_price_regex_detects_rs_prefix():
    assert PRICE_REGEX.search("Charge is Rs. 800.")


def test_price_regex_does_not_match_plain_numbers():
    # A plain number like "3:00 PM" or "150 rooms" should NOT trigger the price regex
    assert not PRICE_REGEX.search("There are 150 rooms in the hotel.")
    assert not PRICE_REGEX.search("Check-in is at 3:00 PM.")


# ---------------------------------------------------------------------------
# Post-Generation Scan Tests (Regex layer — no API key required)
# ---------------------------------------------------------------------------
@patch("google.generativeai.GenerativeModel")
def test_price_hallucination_blocked(mock_gen_model, payment_chunk):
    """Any invented price not in the source chunks must be blocked."""
    mock_gen_model.return_value.generate_content.return_value.text = "PASS"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    hallucinated = "The Deluxe Ocean View Room costs $220 per night."
    response, passed = manager.run_post_generation_scan(hallucinated, payment_chunk)

    assert response == SAFE_REFUSAL
    assert not passed


@patch("google.generativeai.GenerativeModel")
def test_url_hallucination_blocked(mock_gen_model, payment_chunk):
    """Invented payment URLs must be blocked by the regex scanner."""
    mock_gen_model.return_value.generate_content.return_value.text = "PASS"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    hallucinated = "Pay here: http://pay.grandhorizonresort.com/invoice/12345"
    response, passed = manager.run_post_generation_scan(hallucinated, payment_chunk)

    assert response == SAFE_REFUSAL
    assert not passed


@patch("google.generativeai.GenerativeModel")
def test_unknown_email_hallucination_blocked(mock_gen_model, payment_chunk):
    """An email address not present in the source chunk text must be blocked."""
    mock_gen_model.return_value.generate_content.return_value.text = "PASS"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    hallucinated = "Contact billing@fakeaddress.com to pay your bill."
    response, passed = manager.run_post_generation_scan(hallucinated, payment_chunk)

    assert response == SAFE_REFUSAL
    assert not passed


@patch("google.generativeai.GenerativeModel")
def test_grounded_email_in_chunk_passes(mock_gen_model, contact_chunk):
    """An email that appears verbatim in the source chunk text must NOT be blocked."""
    mock_gen_model.return_value.generate_content.return_value.text = "PASS"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    # This email is literally in the contact_chunk fixture above
    grounded_response = "For help, email us at support@grandhorizonresort.com."
    response, passed = manager.run_post_generation_scan(grounded_response, contact_chunk)

    assert response == grounded_response
    assert passed


@patch("google.generativeai.GenerativeModel")
def test_clean_response_passes(mock_gen_model, payment_chunk):
    """A response with no hallucinated prices, URLs, or emails must pass all scans."""
    mock_gen_model.return_value.generate_content.return_value.text = "PASS"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    clean = "We accept Visa and Mastercard at the front desk. No online payment links are issued."
    response, passed = manager.run_post_generation_scan(clean, payment_chunk)

    assert response == clean
    assert passed


@patch("google.generativeai.GenerativeModel")
def test_llm_grounding_check_fail_blocks_response(mock_gen_model, payment_chunk):
    """If the LLM grounding check returns FAIL, the response must be blocked."""
    mock_gen_model.return_value.generate_content.return_value.text = "FAIL"
    manager = GuardrailsManager(api_key="fake-key-for-testing")

    # No regex-detectable hallucination, but LLM disagrees
    unsupported = "Our cancellation policy allows full refunds up to 1 hour before arrival."
    response, passed = manager.run_post_generation_scan(unsupported, payment_chunk)

    assert response == SAFE_REFUSAL
    assert not passed


# ---------------------------------------------------------------------------
# Guardrail Constants
# ---------------------------------------------------------------------------
def test_safe_refusal_string_contains_expected_message():
    """Ensure the safe refusal constant is a meaningful non-empty string."""
    assert len(SAFE_REFUSAL) > 20
    assert "I apologize" in SAFE_REFUSAL or "don't have" in SAFE_REFUSAL


def test_staff_command_denied_string_is_correct():
    """Ensure the staff command denial string is present and correct."""
    assert "Access Denied" in STAFF_COMMAND_DENIED
    assert "staff credentials" in STAFF_COMMAND_DENIED
