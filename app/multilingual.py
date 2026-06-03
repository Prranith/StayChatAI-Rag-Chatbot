from typing import Optional
import json
import google.generativeai as genai
from pydantic import BaseModel, Field

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils import get_logger, retry_on_quota_limit

logger = get_logger(__name__)

class LanguageDetails(BaseModel):
    detected_language: str = Field(..., description="The language of the query. Must be one of: 'english', 'hindi', 'hinglish'.")
    translated_query: str = Field(..., description="The English translation of the user's query. If already English, return the original query.")

class MultilingualHandler:
    """Handles query language detection and English translation to optimize RAG retrieval."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name

    def detect_and_translate(self, query: str) -> LanguageDetails:
        """Detects the language of the query (English/Hindi/Hinglish) and translates it to English."""
        logger.info(f"Detecting language and translating query: '{query}'")
        
        prompt = f"""You are a language analyzer for a hotel guest assistant bot. 
Analyze the following user query:
"{query}"

Determine if the query is in English, Hindi (in Devanagari script), or Hinglish (Hindi words written in Latin script / English letters).
Then, translate the query into clean, standard English to be used for search.

You must return the result as a raw JSON object with the following schema:
{{
  "detected_language": "english" | "hindi" | "hinglish",
  "translated_query": "English translation here"
}}
Return only valid JSON. Do not include markdown formatting or backticks around the JSON.
"""
        
        try:
            model = genai.GenerativeModel(self.model_name)
            # Use generation config to enforce JSON if possible, or clean the output
            response = retry_on_quota_limit(
                model.generate_content,
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text.strip())
            
            # Basic validation
            lang = data.get("detected_language", "english").lower()
            if lang not in ["english", "hindi", "hinglish"]:
                lang = "english"
                
            details = LanguageDetails(
                detected_language=lang,
                translated_query=data.get("translated_query", query)
            )
            logger.info(f"Detected language: {details.detected_language}, English Query: '{details.translated_query}'")
            return details
            
        except Exception as e:
            logger.error(f"Error in language detection and translation: {e}. Falling back to default.")
            # Fallback to defaults on error
            return LanguageDetails(
                detected_language="english",
                translated_query=query
            )
