import sys
import io
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path to allow running as script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag import HotelRAGBot
from app.utils import get_logger

logger = get_logger("cli_interface")

def run_cli() -> None:
    """Runs an interactive command-line interface for chatting with the Hotel RAG bot."""
    print("="*80)
    print("      THE GRAND HORIZON RESORT - AI GUEST ASSISTANT (RAG + FAISS)")
    print("="*80)
    print("Loading AI Assistant, please wait...")
    
    try:
        bot = HotelRAGBot()
    except Exception as e:
        print(f"\nERROR: Could not load the assistant: {e}")
        print("Please verify that:")
        print("1. Your GEMINI_API_KEY is set in a .env file in the project root.")
        print("2. You have run the ingestion pipeline: 'python app/ingest.py'")
        sys.exit(1)

    print("\nAssistant loaded successfully!")
    print("Type 'exit' to quit, 'clear' to reset conversation memory.")
    print("Supported Languages: English, Hindi, Hinglish")
    print("="*80)

    while True:
        try:
            query = input("\nGuest: ").strip()
            if not query:
                continue
                
            if query.lower() == "exit":
                print("Thank you for chatting with The Grand Horizon Resort assistant. Goodbye!")
                break
                
            if query.lower() == "clear":
                bot.memory.clear()
                print("System: Conversation memory has been cleared.")
                continue

            # Process query
            result = bot.process_query(query)
            
            # Print response
            print(f"\nAssistant: {result['response']}")
            
            # Print metadata debug panel
            print("\n" + "-"*40 + " DEBUG PANEL " + "-"*40)
            print(f"  Detected Language : {result['language'].upper()}")
            print(f"  Predicted Intent  : {result['intent'].upper()} (Reason: {result['intent_explanation']})")
            print(f"  Grounded / Safe   : {'YES' if result['grounded'] else 'NO'}")
            print(f"  Refused           : {'YES' if result['refused'] else 'NO'}")
            
            if result['retrieved_chunks']:
                print("  Retrieved Chunks  :")
                for idx, res in enumerate(result['retrieved_chunks']):
                    chunk = res["chunk"]
                    print(f"    {idx+1}. Similarity: {res['similarity']:.4f} | Title: '{chunk.metadata.get('title')}'")
                    print(f"       Text Snippet: \"{chunk.metadata.get('content')[:120]}...\"")
            else:
                print("  Retrieved Chunks  : None (or scores below threshold)")
            print("-"*93)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nSystem Error occurred: {e}")

if __name__ == "__main__":
    run_cli()
