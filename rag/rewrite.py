"""Turn a question into search inputs: a focus text, conditions, interventions, status and sponsors."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1",
                max_retries=5)                 # wait and retry when Groq's rate limit is reached
MODEL = "openai/gpt-oss-120b"
#MODEL = "openai/gpt-oss-20b"

PROMPT = """You turn a question about clinical trials into search inputs for a database.

The database stores one text per trial, in this shape:
"<title>. Conditions: <diseases>. Interventions: <drugs or treatments>. <summary>"
Status (recruiting, completed, ...) and sponsor are stored separately, not in that text.

Return five things:

1. "focus": a short text describing what the trials are about, in the same shape as the
   stored texts: "Conditions: ... Interventions: ...". Keep it short: the disease, and the
   drug class with at most two main drugs. Leave out a part that the question does not
   mention. Never include status or sponsor here. Use "" if the question only names a
   status or sponsor.

2. "conditions": the diseases in the question, as they would appear in a trial's list of
   diseases, with common variants and abbreviations. Each term is matched as a piece of text,
   ignoring capitals: "lung cancer" matches "Non-small Cell Lung Cancer".
   If one word is specific (for example "melanoma"), use it. If a short word would also match
   other diseases (for example "lung"), give precise variants instead.

3. "interventions": the drugs or treatments in the question, grouped per drug. Each group has
   a name and a list of terms; all terms in a group refer to the same drug. The terms are
   matched as text in each trial's title, interventions and summary.
   - For one specific drug: one group, with its generic name, brand names and code names.
     Not its class.
   - For a target or drug class: one group per drug in that class, plus one group named
     "<target> (target)" with the short and full name of the target (for example "EGFR" and
     "epidermal growth factor receptor").
   - Only include drugs that mainly act on that target.
   - Use terms, not phrases: "EGFR", not "EGFR inhibitor".

4. "status":
   - "recruiting" if the question is about trials people can still join ("recruiting",
     "enrolling", "can I join", "open to patients")
   - "active" if it is about trials that are still running ("currently running", "being
     tested", "ongoing")
   - "finished" if it is about completed trials ("completed", "finished", "results")
   - null if it does not say

5. "sponsors": organisations the question names as sponsor or partner, written as they are
   usually registered, with spelling mistakes corrected. Avoid short names that could match
   other organisations: "Roche" would also match "University of Rochester", so use
   "Hoffmann-La Roche".

Any list or group can be empty. Only include names you are confident about.

Examples:

Question: Which JAK inhibitors have been tested in rheumatoid arthritis?
{"focus": "Conditions: rheumatoid arthritis. Interventions: JAK inhibitor, tofacitinib, baricitinib.",
 "conditions": ["rheumatoid arthritis"],
 "interventions": {
   "JAK (target)": ["JAK", "Janus kinase"],
   "tofacitinib": ["tofacitinib", "Xeljanz", "CP-690550"],
   "baricitinib": ["baricitinib", "Olumiant", "LY3009104"],
   "upadacitinib": ["upadacitinib", "Rinvoq", "ABT-494"]},
 "status": null, "sponsors": []}

Question: ozempic trials by novo nordisk
{"focus": "Interventions: semaglutide (Ozempic).",
 "conditions": [],
 "interventions": {"semaglutide": ["semaglutide", "Ozempic", "Wegovy", "Rybelsus"]},
 "status": null, "sponsors": ["Novo Nordisk"]}

Question: Are there recruiting trials for Parkinson's disease?
{"focus": "Conditions: Parkinson's disease.",
 "conditions": ["Parkinson"], "interventions": {},
 "status": "recruiting", "sponsors": []}

Question: breast cancer trials currently run by roche
{"focus": "Conditions: breast cancer.",
 "conditions": ["breast cancer", "breast neoplasm", "breast carcinoma"], "interventions": {},
 "status": "active", "sponsors": ["Hoffmann-La Roche"]}

Return JSON only, in the form shown in the examples."""


def rewrite(question: str) -> dict:
    """Return the search inputs for a question."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": question}],
        response_format={"type": "json_object"},
        temperature=0,
        reasoning_effort="medium",
        max_tokens=2000,
    )
    result = json.loads(response.choices[0].message.content)
    status = result.get("status")
    interventions = result.get("interventions") or {}
    if isinstance(interventions, list):            # safety net: a plain list becomes one group per term
        interventions = {term: [term] for term in interventions}
    return {
        "focus": result.get("focus") or "",
        "conditions": result.get("conditions") or [],
        "interventions": interventions,
        "status": status if status in ("recruiting", "active", "finished") else None,
        "sponsors": result.get("sponsors") or [],
    }