import re
import json
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils import get_logger, retry_on_quota_limit

logger = get_logger(__name__)

SAFE_REFUSAL = (
    "I apologize, but I don't have that specific information in my knowledge base. "
    "I can connect you with a member of our front desk team to assist you further."
)

STAFF_COMMAND_DENIED = "Access Denied: Operational commands require secure staff credentials."

# Regex to detect currency and prices (e.g. $100, Rs. 500, 200 INR, 5000 rupees, 100 USD)
PRICE_REGEX = re.compile(
    r'(?:(?:\b(?:Rs\.?|INR|USD|EUR|GBP|Rupees|dollars|inr|usd)\b|[\$€£₹])\s*\d+(?:[\.,]\d+)?\b)|'
    r'(?:\b\d+(?:[\.,]\d+)?\s*(?:\b(?:Rs\.?|INR|USD|EUR|GBP|Rupees|dollars|inr|usd)\b|[\$€£₹]))',
    re.IGNORECASE
)

# Regex to detect URLs (links)
URL_REGEX = re.compile(
    r'\b(?:https?://|www\.)\S+\b',
    re.IGNORECASE
)

# Regex to detect email addresses
EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

class GuardrailsManager:
    """Manages the verification of retrieval context and generated responses to enforce zero-hallucination policies."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name

    def verify_context_sufficiency(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> bool:
        """Uses Gemini to evaluate if the retrieved chunks contain enough details to answer the query."""
        if not retrieved_chunks:
            logger.warning("No chunks retrieved. Context is insufficient.")
            return False
            
        # Combine all chunk texts
        context = "\n\n".join([f"--- Chunk {idx+1} ---\n{res['chunk'].text}" for idx, res in enumerate(retrieved_chunks)])
        
        prompt = f"""You are a strict compliance auditor for a hotel chatbot (The Grand Horizon Resort).
Your job is to read the retrieved context chunks and decide if they contain sufficient, direct facts to answer the user's query.

Rule:
- If the retrieved context contains explicit information that directly addresses the query, answer YES.
- If the context lacks the answer, or if the answer is only partially present, or if answering would require making assumptions/inventing details (e.g., pricing, links, or specifics not written in the chunks), answer NO.

User Query:
"{query}"

Retrieved Context:
{context}

Response (Only YES or NO):
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = retry_on_quota_limit(
                model.generate_content,
                prompt,
                generation_config={"temperature": 0.0} # deterministic
            )
            decision = response.text.strip().upper()
            logger.info(f"Sufficiency check decision: '{decision}' for query: '{query[:40]}'")
            return "YES" in decision
        except Exception as e:
            logger.error(f"Error checking context sufficiency: {e}. Defaulting to True (will run post-validation).")
            return True

    def run_post_generation_scan(self, response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """Programmatically scans response text for price/URL hallucinations.
        
        If a price, URL, or email is generated that does not exist in the retrieved chunks,
        we block the response and return the safe refusal.
        """
        # Extract source texts for substring verification
        source_context_lower = " ".join([res["chunk"].text.lower() for res in retrieved_chunks])
        
        # 1. Scan for pricing patterns
        prices_in_response = PRICE_REGEX.findall(response_text)
        if prices_in_response:
            logger.warning(f"Price pattern detected in response: {prices_in_response}")
            for price in prices_in_response:
                # Check if this price string is in the context
                if price.lower() not in source_context_lower:
                    logger.error(f"Blocked response due to ungrounded price hallucination: '{price}'")
                    return SAFE_REFUSAL, False
                    
        # 2. Scan for URLs / Links
        urls_in_response = URL_REGEX.findall(response_text)
        if urls_in_response:
            logger.warning(f"URL pattern detected in response: {urls_in_response}")
            for url in urls_in_response:
                if url.lower() not in source_context_lower:
                    logger.error(f"Blocked response due to ungrounded URL/Link hallucination: '{url}'")
                    return SAFE_REFUSAL, False

        # 3. Scan for Email addresses
        emails_in_response = EMAIL_REGEX.findall(response_text)
        if emails_in_response:
            logger.warning(f"Email pattern detected in response: {emails_in_response}")
            for email in emails_in_response:
                if email.lower() not in source_context_lower:
                    logger.error(f"Blocked response due to ungrounded Email hallucination: '{email}'")
                    return SAFE_REFUSAL, False

        # 4. LLM-based Grounding Double Check
        # Ensure LLM didn't invent some other specific claim (e.g. availability, policies)
        if retrieved_chunks:
            context = "\n\n".join([res['chunk'].text for res in retrieved_chunks])
            grounding_prompt = f"""You are a strict compliance auditor for The Grand Horizon Resort.
Compare the generated assistant response with the retrieved context. Verify if the assistant response makes any ungrounded factual claims, rules, or specific policies that contradict or go beyond the retrieved context.

Rules:
- Generalizations and synonyms are allowed (e.g. classifying a 'cat' or 'dog' as a 'pet' matches a 'no-pet policy', and referring to the resort's pool as 'swimming pool' matches the context).
- If the response invents any prices, rates, online links, booking codes, or specific rules not stated in the context, output FAIL.
- If the response contains statements that cannot be verified by the context, output FAIL.
- Otherwise, output PASS.

Retrieved Context:
{context}

Assistant Response:
{response_text}

Output (PASS or FAIL):
"""
            try:
                model = genai.GenerativeModel(self.model_name)
                val_res = retry_on_quota_limit(
                    model.generate_content,
                    grounding_prompt,
                    generation_config={"temperature": 0.0}
                )
                decision = val_res.text.strip().upper()
                logger.info(f"LLM grounding check result: {decision}")
                if "FAIL" in decision:
                    logger.error("Blocked response: LLM grounding check failed.")
                    return SAFE_REFUSAL, False
            except Exception as e:
                logger.error(f"LLM grounding check failed to run: {e}")

        logger.info("Response passed all post-generation scans.")
        return response_text, True
