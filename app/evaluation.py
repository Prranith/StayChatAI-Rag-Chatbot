import json
import sys
import io
from pathlib import Path
from typing import Dict, Any, List

# Reconfigure stdout/stderr to use UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path to allow running as script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DATA_PATH, EVAL_RESULTS_PATH
from app.rag import HotelRAGBot
from app.utils import get_logger

logger = get_logger("evaluation_runner")

def run_evaluation() -> None:
    """Runs the evaluation test set and compiles accuracy reports."""
    logger.info("Initializing Hotel RAG Bot for evaluation...")
    try:
        bot = HotelRAGBot()
    except Exception as e:
        logger.error(f"Failed to initialize HotelRAGBot: {e}. Make sure to run ingestion first.")
        return

    if not EVAL_DATA_PATH.exists():
        logger.error(f"Evaluation dataset not found at {EVAL_DATA_PATH}")
        return

    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    logger.info(f"Loaded {len(eval_cases)} evaluation test cases. Beginning test execution...")
    
    results = []
    passed_cases = 0
    intent_correct = 0
    guardrail_correct = 0

    print("\n" + "="*80)
    print("                 HOTEL RAG CHATBOT SYSTEM EVALUATION REPORT")
    print("="*80)

    for item in eval_cases:
        case_id = item["id"]
        question = item["question"]
        expected_intent = item["expected_intent"]
        expected_status = item["grounded_or_refused"]
        expected_keys = item["expected_key_points"]
        
        # Clear bot memory before each evaluation query to prevent cross-contamination
        bot.memory.clear()

        # Sleep between cases to avoid rapid-fire API quota exhaustion
        if item != eval_cases[0]:
            import time
            logger.info("Sleeping 3 seconds before next case to respect API rate limits...")
            time.sleep(3.0)

        logger.info(f"Evaluating Case {case_id}: '{question[:50]}'")
        
        # Run bot processing
        output = bot.process_query(question)
        
        predicted_intent = output["intent"]
        response_text = output["response"]
        is_refused = output["refused"]
        is_grounded = output["grounded"]
        detected_lang = output["language"]
        
        # Check Intent
        intent_match = predicted_intent == expected_intent
        if intent_match:
            intent_correct += 1

        # Check Grounding/Refusal status
        # Expected status matches actual status
        status_match = False
        if expected_status == "refused" and is_refused:
            status_match = True
        elif expected_status == "grounded" and is_grounded:
            status_match = True
            
        if status_match:
            guardrail_correct += 1

        # Check key points presence in response
        missing_keys = []
        for key in expected_keys:
            # simple case-insensitive check
            if key.lower() not in response_text.lower():
                missing_keys.append(key)
                
        # If expected status is refused, missing keys are acceptable (refusal prompt won't have specific details)
        keys_satisfied = len(missing_keys) == 0 if expected_status == "grounded" else True
        
        # Determine overall case pass/fail
        case_passed = intent_match and status_match and keys_satisfied
        if case_passed:
            passed_cases += 1

        status_str = "REFUSED" if is_refused else "GROUNDED"
        result_status = "PASS" if case_passed else "FAIL"
        
        print(f"\n[Case {case_id}] Result: {result_status}")
        print(f"  Question   : {question}")
        print(f"  Language   : {detected_lang}")
        print(f"  Intent     : Expected '{expected_intent}' | Predicted '{predicted_intent}' ({'OK' if intent_match else 'MISMATCH'})")
        print(f"  Status     : Expected '{expected_status}' | Actual '{status_str.lower()}' ({'OK' if status_match else 'MISMATCH'})")
        print(f"  Response   : {response_text}")
        if expected_status == "grounded" and missing_keys:
            print(f"  Missing Key: {missing_keys}")
            
        # Store detailed output
        results.append({
            "case_id": case_id,
            "description": item["description"],
            "question": question,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_match": intent_match,
            "expected_status": expected_status,
            "actual_status": status_str.lower(),
            "status_match": status_match,
            "response": response_text,
            "missing_key_points": missing_keys,
            "keys_satisfied": keys_satisfied,
            "passed": case_passed,
            "language": detected_lang
        })

    # Summary calculations
    total_cases = len(eval_cases)
    overall_accuracy = (passed_cases / total_cases) * 100
    intent_accuracy = (intent_correct / total_cases) * 100
    guardrail_accuracy = (guardrail_correct / total_cases) * 100

    print("\n" + "="*80)
    print("                             METRICS SUMMARY")
    print("="*80)
    print(f"Total Test Cases Run      : {total_cases}")
    print(f"Overall Pass Rate         : {passed_cases}/{total_cases} ({overall_accuracy:.2f}%)")
    print(f"Intent Classifier Accuracy: {intent_correct}/{total_cases} ({intent_accuracy:.2f}%)")
    print(f"Guardrail Correctness     : {guardrail_correct}/{total_cases} ({guardrail_accuracy:.2f}%)")
    print("="*80)

    # Save to disk
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "overall_accuracy_percent": overall_accuracy,
                "intent_accuracy_percent": intent_accuracy,
                "guardrail_accuracy_percent": guardrail_accuracy
            },
            "cases": results
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation report written to {EVAL_RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
