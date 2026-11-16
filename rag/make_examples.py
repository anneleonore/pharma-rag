"""Answer the app's example questions once, and store the answers (so clicking them costs no tokens).
Run from the pharma-rag folder:  python rag/make_examples.py"""
import json
from pathlib import Path

from answer import answer, to_plain

EXAMPLES = {
    "EGFR inhibitors in lung cancer": "Which EGFR inhibitors are being tested in lung cancer?",
    "AstraZeneca in lung cancer": "Which lung cancer trials is AstraZeneca currently running?",
    "Recruiting Parkinson's trials": "Are there recruiting trials for Parkinson's disease?",
}
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "example_answers.json"

stored = {}
for label, question in EXAMPLES.items():
    print("Answering:", question)
    stored[label] = {"question": question, **to_plain(answer(question))}
OUT_FILE.write_text(json.dumps(stored, indent=2, ensure_ascii=False))
print("Saved to", OUT_FILE)
