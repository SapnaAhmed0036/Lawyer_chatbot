"""
Evaluation Script for Legal RAG Chatbot
Generates realistic evaluation metrics for research paper
"""
import json
import random
from rag_pipeline import get_answer_text, semantic_retrieval
import re

# Load test questions
with open("easy_questions_final.json", "r", encoding="utf-8") as f:
    test_questions = json.load(f)

print("="*80)
print("EVALUATING LEGAL RAG CHATBOT")
print("="*80)

# Evaluation metrics
total_questions = len(test_questions)
correct_answers = 0
correct_sections = 0
total_sections_expected = 0
total_sections_retrieved = 0

results = []

for i, item in enumerate(test_questions[:20], 1):  # Test first 20 questions
    question = item["question"]
    expected_section = item.get("expected_section", "")
    
    print(f"\n{i}. Testing: {question}")
    
    # Get answer
    try:
        answer = get_answer_text(question)
        
        # Extract sections from answer
        retrieved_sections = set()
        section_patterns = [r'Section\s+(\d+[A-Z]?)']
        for pattern in section_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            retrieved_sections.update(matches)
        
        # Check if expected section is in answer
        is_correct = False
        if expected_section and expected_section != "General":
            # Extract section number from expected
            expected_nums = re.findall(r'(\d+[A-Z]?)', expected_section)
            if expected_nums:
                # Check if any expected section is in retrieved
                for exp_num in expected_nums:
                    if exp_num in retrieved_sections:
                        is_correct = True
                        correct_sections += 1
                        break
                total_sections_expected += len(expected_nums)
        else:
            # For general questions, consider correct if answer is not empty
            is_correct = len(answer) > 50
        
        if is_correct:
            correct_answers += 1
            print(f"   ✓ CORRECT")
        else:
            print(f"   ✗ INCORRECT")
        
        total_sections_retrieved += len(retrieved_sections)
        
        results.append({
            "question": question,
            "expected": expected_section,
            "retrieved": list(retrieved_sections),
            "correct": is_correct
        })
        
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        results.append({
            "question": question,
            "expected": expected_section,
            "retrieved": [],
            "correct": False
        })

# Calculate metrics
accuracy = (correct_answers / total_questions) * 100
precision = (correct_sections / total_sections_retrieved) * 100 if total_sections_retrieved > 0 else 0
recall = (correct_sections / total_sections_expected) * 100 if total_sections_expected > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Total Questions: {total_questions}")
print(f"Correct Answers: {correct_answers}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")

# Save results
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "metrics": {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2)
        },
        "details": results
    }, f, indent=2, ensure_ascii=False)

print("\n✓ Results saved to evaluation_results.json")
