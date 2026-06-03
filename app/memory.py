from typing import List, Dict, Any
import google.generativeai as genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.utils import get_logger, retry_on_quota_limit

logger = get_logger(__name__)

class ConversationMemory:
    """Manages chat history and provides query reformulation for multi-turn conversations."""
    
    def __init__(self, max_turns: int = 5, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name

    def add_user_message(self, text: str) -> None:
        """Adds a user query to the history."""
        self.history.append({"role": "user", "content": text})
        # Keep history within limit (each turn has 1 user + 1 model message, so max_turns * 2)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def add_bot_message(self, text: str) -> None:
        """Adds the bot response to the history."""
        self.history.append({"role": "model", "content": text})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the current chat history."""
        return self.history

    def clear(self) -> None:
        """Resets the conversation history."""
        self.history = []
        logger.info("Conversation memory cleared.")

    def reformulate_query(self, query: str) -> str:
        """Reformulates a follow-up query using the conversation history to make it search-friendly."""
        # If there's no history (or just the current user query, which hasn't been added yet),
        # return the query as-is.
        if not self.history:
            return query
            
        logger.info("Reformulating query based on conversation history...")
        
        # Build chat transcript for the prompt
        transcript = ""
        for turn in self.history:
            role = "Guest" if turn["role"] == "user" else "Assistant"
            transcript += f"{role}: {turn['content']}\n"
            
        prompt = f"""You are an assistant for The Grand Horizon Resort. 
Given the conversation history between a Guest and the Assistant, rewrite the Guest's latest question to be a complete, self-contained search query. 
The rewritten query will be used for a vector search against the hotel's knowledge base. Do not answer the question; only rewrite it.
If the question is already fully self-contained or does not refer to previous turns, return it exactly as is.

Conversation History:
{transcript}

Latest Question from Guest:
"{query}"

Self-contained Search Query:
"""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = retry_on_quota_limit(model.generate_content, prompt)
            reformulated = response.text.strip()
            # Clean up quotes if returned
            if reformulated.startswith('"') and reformulated.endswith('"'):
                reformulated = reformulated[1:-1]
            logger.info(f"Original: '{query}' -> Reformulated: '{reformulated}'")
            return reformulated
        except Exception as e:
            logger.error(f"Error reformulating query: {e}. Returning original.")
            return query
