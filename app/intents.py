import json
from typing import Dict, Any
import google.generativeai as genai
from pydantic import BaseModel, Field

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils import get_logger, retry_on_quota_limit

logger = get_logger(__name__)

class IntentClassification(BaseModel):
    intent: str = Field(..., description="The predicted intent. Must be exactly one of: 'booking inquiry', 'amenity question', 'complaint', 'staff command', 'other'.")
    explanation: str = Field(..., description="A short explanation justifying the chosen intent class.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0.")

class IntentClassifier:
    """Classifies user queries into hotel-specific intent categories using Gemini."""
    
    VALID_INTENTS = ["booking inquiry", "amenity question", "complaint", "staff command", "other"]

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name

    def classify(self, query: str) -> IntentClassification:
        """Classifies the given query and returns a structured classification object."""
        logger.info(f"Classifying intent for query: '{query}'")
        
        prompt = f"""You are an intent classification system for a luxury hotel chatbot (The Grand Horizon Resort).
Classify the user's query into exactly one of these five categories:

1. 'booking inquiry': Questions about booking rooms, room specs, room capacity, check-in/check-out rules, cancellation policies, or room pricing.
2. 'amenity question': Questions about parking, guest Wi-Fi, pool/gym/spa timings, hotel location, restaurants, or airport shuttle transfers.
3. 'complaint': Feedback or statements indicating an issue, poor service, dirty rooms, noise, billing errors, or demanding escalations.
4. 'staff command': Restricted operational commands (e.g. system override, get guest list, set room temperature, check logs), or any command starting with "/staff" or "/system".
5. 'other': General chit-chat, greetings (hi, hello), thanks, or questions unrelated to the hotel.

Examples:
- "Is there a pool?" -> 'amenity question'
- "Can I book a room?" -> 'booking inquiry'
- "How much is the Deluxe Room?" -> 'booking inquiry'
- "Mujhe room booking karni hai, rates kya hain?" -> 'booking inquiry'
- "Room is very dirty, fix this now!" -> 'complaint'
- "/staff print guest list" -> 'staff command'
- "Set AC temperature to 18" -> 'staff command'
- "Hello, good morning!" -> 'other'

Query to classify:
"{query}"

Return only a valid JSON object matching this schema:
{{
  "intent": "booking inquiry" | "amenity question" | "complaint" | "staff command" | "other",
  "explanation": "Why this category was chosen",
  "confidence": 0.95
}}
Do not include markdown blocks or any text outside of the JSON.
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = retry_on_quota_limit(
                model.generate_content,
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            data = json.loads(response.text.strip())
            intent = data.get("intent", "other").lower().strip()
            
            # Validation
            if intent not in self.VALID_INTENTS:
                logger.warning(f"Invalid intent returned: '{intent}'. Defaulting to 'other'.")
                intent = "other"
                
            return IntentClassification(
                intent=intent,
                explanation=data.get("explanation", "Standard classification"),
                confidence=float(data.get("confidence", 1.0))
            )
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}. Falling back to 'other'.")
            return IntentClassification(
                intent="other",
                explanation=f"Fallback due to error: {e}",
                confidence=0.5
            )
