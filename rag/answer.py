"""Answer a question about clinical trials: rewrite, search, count, answer, then check the answer."""
import re

import pandas as pd

from rewrite import MODEL, client, rewrite
from search import count_drugs, expand_terms, search

ANSWER_PROMPT = """You answer questions about clinical trials with at least one site in Australia,
using data from ClinicalTrials.gov.

## What you are given
1. Selection: which trials were selected (status and sponsor, if the question named them).
2. Counts: every drug that appears in ALL selected trials, with the number of trials.
   This includes drugs that do not fit the question (for example chemotherapy given
   alongside, or another drug tested in the same patients).
3. Trials: the most relevant selected trials, with details. This is a sample, not all trials.

## How to use it
- Base every statement on the information given. Use your own knowledge only to recognise
  that two names refer to the same drug.
- Decide which drugs in the counts fit the question; mention only those.
- For numbers, use the counts and the total, never the sample.
- For examples, use the trials: at most two per drug.
- Leave out trials that do not fit the question.

## How to answer
- Choose one format:
  - Drug format: only if the question asks which drugs (for example "which EGFR inhibitors").
  - Trial format: for all other questions, including questions about one named drug.
- Start with the opening sentence of that format. N is "Total selected trials". Leave out the
  status word if the selection has no status.
- List each drug or trial once. Cite trial IDs in square brackets, for example [NCT01234567].
- If the information does not answer the question, say so. Do not give medical advice.

## Drug format
<N> <recruiting/active/finished> trials <describe the topic>. They test these drugs:

- **<drug name>** (<other names>): <K> trials. <role> in [<trial ID>]; <role> in [<trial ID>].

K is the drug's count. Roles: tested drug, combination partner or comparison.
Example opening: 40 active trials test EGFR inhibitors in lung cancer. They test these drugs:

## Trial format
<N> <recruiting/active/finished> trials <describe the topic>. The most relevant are listed below.

- **<short trial description>** [<trial ID>]: <main drugs tested>.

Name the main drugs, not chemotherapy given alongside. For a question about one drug, say what
each trial combines it with. Do not write phase, status or sponsor: they are added automatically.
Example opening: 20 recruiting trials are for Parkinson's disease. The most relevant are listed below.

End after the last line of the list: no closing summary."""


CHECK_PROMPT = """You are a strict fact-checker. You check a draft answer about clinical trials
against its sources: the counts, and the trials the draft cites. Assume the draft contains
mistakes: your job is to find them.

## How to check
Go through the draft line by line. For each claim, find the source text it depends on (the
cited trial, or the counts) and check that the source states it. A claim is supported
only if the source says it. It is not enough that it sounds plausible, or that you believe it.

Pay special attention to:
- Drugs: is each drug named in the cited trial?
- Roles: does the trial say the drug is tested alone, combined, or used as a comparison?
- Targets: if the draft says what a drug acts on, does the trial say so?
- Counts: does each drug count match the counts?
- Duplicates: is one drug listed twice under different names?

## What to do with a mistake
Correct it if the sources give the right information; otherwise remove the claim. Do not add
new information. Keep the opening sentence, the draft's structure and its citations.

## Output
First the corrected answer. Then a line with only "CHANGES:", followed by one line per change,
each starting with "- " and naming the trial ID. If you found no mistakes, write "- none"."""


def format_trials(trials: pd.DataFrame) -> str:
    """Turn the search results into text the LLM can read, one block per trial."""
    blocks = []
    for r in trials.itertuples():
        summary = r.summary[:300] if isinstance(r.summary, str) else ""
        start = int(r.start_year) if pd.notna(r.start_year) else "unknown"
        partners = f" | Collaborators: {r.collaborators}" if isinstance(r.collaborators, str) else ""
        blocks.append(
            f"[{r.trial_id}] {r.title}\n"
            f"Phase: {r.phase} | Status: {r.status} | Sponsor: {r.sponsor}{partners} | Start: {start}\n"
            f"Conditions: {r.conditions}\n"
            f"Interventions: {r.interventions}\n"
            f"Summary: {summary}"
        )
    return "\n\n".join(blocks)


def _clean(text: str) -> str:
    """Use square brackets for citations (the model sometimes uses 【】)."""
    return text.replace("【", "[").replace("】", "]")


TRIAL_LINE = re.compile(r"^(\s*[-•*]\s*\*\*.+?\*\*\s*\[(NCT\d{8})\]\s*:.*?)\s*$")
CITATION = re.compile(r"\[(NCT\d{8})\]")


def trial_facts(r) -> str:
    """Phase, status and sponsors of one trial, written from the data (not by the LLM)."""
    parts = [str(r.phase) if isinstance(r.phase, str) else "phase not applicable",
             str(r.status).replace("_", " ").lower(),
             f"lead sponsor: {r.sponsor}"]
    if isinstance(r.collaborators, str):
        parts.append(f"collaborators: {r.collaborators}")
    return " | ".join(parts)


def add_trial_facts(answer_text: str, trials: pd.DataFrame) -> str:
    """Add the facts from the data to each line of the trial format."""
    facts = {r.trial_id: trial_facts(r) for r in trials.itertuples()}
    lines = []
    for line in answer_text.splitlines():
        m = TRIAL_LINE.match(line)
        if m and m.group(2) in facts:
            line = f"{m.group(1).rstrip('. ')}. *{facts[m.group(2)]}*"
        lines.append(line)
    return "\n".join(lines)


def unknown_citations(answer_text: str, trials: pd.DataFrame) -> list[str]:
    """Trial IDs cited in the answer that are not among the trials the LLM was given."""
    return sorted(set(CITATION.findall(answer_text)) - set(trials["trial_id"]))


def finalize(draft: str, checked: dict, trials: pd.DataFrame) -> tuple[str, list[str]]:
    """Steps done in code after the check: keep the draft's opening sentence, add the facts
    from the data to each trial line, and report cited trials that were not in the sources."""
    opening = draft.strip().split("\n", 1)[0]
    lines = checked["answer"].strip().split("\n", 1)
    text = opening + ("\n" + lines[1] if len(lines) > 1 else "")
    changes = [c for c in checked["changes"] if not c.lower().startswith("none")]
    changes += [f"cited trial not in the sources: {t}" for t in unknown_citations(text, trials)]
    return add_trial_facts(text, trials), changes


def check(question: str, context: str, draft: str) -> dict:
    """Second LLM pass: check each claim in the draft against the sources."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": CHECK_PROMPT},
                      {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"
                                                  f"\n\nDraft answer:\n{draft}"}],
            temperature=0,
            reasoning_effort="medium",
            max_tokens=6000,
        )
        text = response.choices[0].message.content or ""
    except Exception as error:                       # if the check fails, keep the draft
        return {"answer": draft, "changes": [f"check failed ({type(error).__name__}); showing the draft"]}

    answer_part, _, changes_part = text.partition("CHANGES:")
    changes = [line.strip()[2:].strip() for line in changes_part.splitlines()
               if line.strip().startswith("- ")]
    changes = [c for c in changes if c.lower() != "none"]
    return {"answer": _clean(answer_part.strip() or draft), "changes": changes}


def build_context(terms: dict, trials: pd.DataFrame, total: int, counts) -> str:
    """Write the sources for the LLM: the selection, the counts and the trials."""
    status_text = {"recruiting": "only recruiting trials (recruiting or not yet recruiting)",
                   "active": "only active trials (recruiting or ongoing)",
                   "finished": "only finished trials (completed or terminated)"}.get(terms["status"], "all statuses")
    sponsor_text = ", ".join(terms["sponsors"]) if terms["sponsors"] else "any sponsor"
    context = (f"Selection: {status_text}; sponsor or collaborator: {sponsor_text}.\n"
               f"Total selected trials: {total}.\n\n")
    if counts is not None and len(counts):
        context += ("Counts (drugs in ALL selected trials, with the number of trials):\n"
                    + "\n".join(f"- {r.drug}" + (f" ({r.other_names})" if isinstance(r.other_names, str) else "")
                                 + f": {r.trials}" for r in counts.itertuples())
                    + "\n\n")
    context += f"Trials (the {len(trials)} most relevant):\n\n" + format_trials(trials)
    return context


def check_sources(terms: dict, trials: pd.DataFrame, total: int, counts, draft: str) -> str:
    """The sources for the check: the counts, and only the trials the draft cites."""
    cited = set(CITATION.findall(draft))
    return build_context(terms, trials[trials["trial_id"].isin(cited)], total, counts)


def answer(question: str, k: int = 5) -> dict:
    """Run the full pipeline for one question."""
    terms = rewrite(question)                                          # 1. rewrite
    terms["interventions"] = expand_terms(terms["interventions"])     #    + all names from the data
    selection = dict(conditions=terms["conditions"], interventions=terms["interventions"],
                     status_group=terms["status"], sponsors=terms["sponsors"])

    trials = search(focus=terms["focus"], k=k, **selection)            # 2. search
    total = int(trials["total_matches"].iloc[0]) if len(trials) else 0
    counts = count_drugs(**selection)                                  # 3. count

    context = build_context(terms, trials, total, counts)              # 4. augment

    response = client.chat.completions.create(                         # 5. generate a draft
        model=MODEL,
        messages=[{"role": "system", "content": ANSWER_PROMPT},
                  {"role": "user", "content": f"Question: {question}\n\n{context}"}],
        temperature=0,
        reasoning_effort="medium",
        max_tokens=6000,
    )
    draft = _clean(response.choices[0].message.content)

    sources = check_sources(terms, trials, total, counts, draft)        # 6. check the draft,
    checked = check(question, sources, draft)                          #    against the cited trials
    final, changes = finalize(draft, checked, trials)                  # 7. steps in code

    return dict(answer=final, draft=draft, changes=changes,
                terms=terms, total=total, counts=counts, trials=trials)



def to_plain(result: dict) -> dict:
    """The result as plain lists and dicts, so it can be stored as JSON or shown in the app."""
    plain = {k: v for k, v in result.items() if k not in ("counts", "trials")}
    counts, trials = result.get("counts"), result.get("trials")
    plain["counts"] = (counts[["drug", "trials"]].to_dict("records")
                       if counts is not None and len(counts) else [])
    columns = ["trial_id", "title", "phase", "status", "sponsor", "url"]
    plain["trials"] = (trials[columns].astype(object).where(trials[columns].notna(), None)
                       .to_dict("records") if trials is not None and len(trials) else [])
    plain["counts"] = [{"drug": c["drug"], "trials": int(c["trials"])} for c in plain["counts"]]
    plain["total"] = int(plain.get("total", 0))
    return plain