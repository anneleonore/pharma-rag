"""Search the clinical trials index (data/pharma.duckdb): find trials, and count them."""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "pharma.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"   # must be the same model that built the index
DIM = 384

INTERVENTION_COLUMNS = ["title", "interventions", "summary", "drug_terms"]
CONDITION_COLUMNS = ["conditions", "condition_terms"]
SPONSOR_COLUMNS = ["sponsor", "collaborators"]

# Trial statuses grouped by what a question asks for
STATUS_GROUPS = {
    "recruiting": ["RECRUITING", "NOT_YET_RECRUITING"],
    "active": ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"],
    "finished": ["COMPLETED", "TERMINATED"],
}

_con = None
_model = None


def get_con():
    """Open the database once, read-only."""
    global _con
    if _con is None:
        _con = duckdb.connect(str(DB_FILE), read_only=True)
    return _con


def get_model():
    """Load the embedding model once."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _any_word(words, columns):
    """SQL condition: at least one of the words appears in at least one of the columns."""
    words = [words] if isinstance(words, str) else list(words)
    parts, params = [], []
    for w in words:
        parts.append("(" + " OR ".join(f"{c} ILIKE ?" for c in columns) + ")")
        params += [f"%{w}%"] * len(columns)
    return "(" + " OR ".join(parts) + ")", params


def all_terms(interventions):
    """All intervention terms in one list (from groups, or a plain list)."""
    if isinstance(interventions, dict):
        return [t for terms in interventions.values() for t in terms]
    return list(interventions or [])


def _selection(conditions=None, interventions=None, status_group=None, sponsors=None):
    """The SQL checks a trial must pass, and their values."""
    where, params = [], []
    interventions = all_terms(interventions)
    for words, columns in [(interventions, INTERVENTION_COLUMNS),
                           (conditions, CONDITION_COLUMNS),
                           (sponsors, SPONSOR_COLUMNS)]:
        if words:
            clause, p = _any_word(words, columns)
            where.append(clause); params += p
    if status_group:
        statuses = STATUS_GROUPS[status_group]
        where.append(f"status IN ({', '.join('?' * len(statuses))})"); params += statuses
    return where, params


def search(focus=None, conditions=None, interventions=None, status_group=None, sponsors=None,
           k=15, min_similarity=0.35) -> pd.DataFrame:
    """Find the most relevant trials.
    focus:         text describing what the trials are about; compared by meaning (embedding)
    conditions:    disease terms; the trial's diseases must contain at least one
    interventions: drug groups {name: [terms]} or a list of terms; the trial must mention at least one
    status_group:  "recruiting", "active" or "finished" (None = all)
    sponsors:      organisation names; the sponsor or a collaborator must contain at least one"""
    params = []

    # 1. Meaning: turn the focus text into an embedding
    if focus:
        params.append(get_model().encode(focus, normalize_embeddings=True).tolist())
        sim_sql = f"array_cosine_similarity(embedding, ?::FLOAT[{DIM}])"
    else:
        sim_sql = "NULL::FLOAT"

    # 2. Selection: words and filters
    where, p = _selection(conditions, interventions, status_group, sponsors)
    params += p

    # 3. Without any word-based selection, drop trials with a very low similarity
    selected_by_words = bool(conditions or all_terms(interventions) or sponsors)
    threshold = "WHERE similarity >= ?" if (focus and not selected_by_words) else ""
    if threshold:
        params.append(min_similarity)

    # 4. Run: select, score, sort, return the top k
    sql = f"""
        WITH scored AS (
            SELECT trial_id, title, conditions, interventions, phase, status, sponsor,
                   collaborators, start_year, summary, url, {sim_sql} AS similarity
            FROM trials
            {"WHERE " + " AND ".join(where) if where else ""}
        )
        SELECT *, count(*) OVER () AS total_matches
        FROM scored
        {threshold}
        ORDER BY similarity DESC NULLS LAST, start_year DESC
        LIMIT {k}
    """
    return get_con().execute(sql, params).df()


def count_by_intervention(interventions, conditions=None, status_group=None, sponsors=None) -> pd.DataFrame:
    """For each drug group, count ALL selected trials that mention any of its terms."""
    if not isinstance(interventions, dict):
        interventions = {term: [term] for term in interventions}
    where, params = _selection(conditions, None, status_group, sponsors)
    extra = "".join(f" AND {w}" for w in where)
    rows = []
    for name, terms in interventions.items():
        if not terms:
            continue
        clause, p = _any_word(terms, INTERVENTION_COLUMNS)
        n = get_con().execute(f"SELECT count(*) FROM trials WHERE {clause}{extra}",
                              p + params).fetchone()[0]
        rows.append({"drug": name, "terms": ", ".join(terms), "trials": n})
    counts = pd.DataFrame(rows, columns=["drug", "terms", "trials"])
    counts = counts[counts["trials"] > 0]          # leave out drugs no selected trial mentions
    return counts.sort_values("trials", ascending=False, ignore_index=True)


def expand_terms(interventions: dict) -> dict:
    """Add every name the drug dictionary knows for each term (from the drug_names table)."""
    expanded = {}
    for group, terms in interventions.items():
        names = {t.lower() for t in terms}
        found = get_con().execute(
            f"""SELECT name FROM drug_names
                WHERE drug IN (SELECT drug FROM drug_names WHERE name IN ({", ".join("?" * len(names))}))""",
            list(names)).fetchall() if names else []
        extra = {n for (n,) in found if len(n) >= 4}          # skip very short names (match inside words)
        expanded[group] = list(terms) + sorted(extra - names)
    return expanded


def count_drugs(conditions=None, interventions=None, status_group=None, sponsors=None, top=15) -> pd.DataFrame:
    """Count every drug that appears in the selected trials (from the trial_drugs table)."""
    where, params = _selection(conditions, interventions, status_group, sponsors)
    sql = f"""
        SELECT d.drug,
               (SELECT string_agg(name, ', ') FROM (SELECT name FROM drug_names n
                    WHERE n.drug = d.drug AND n.name <> d.drug LIMIT 4)) AS other_names,
               count(DISTINCT d.trial_id) AS trials
        FROM trial_drugs d
        JOIN (SELECT trial_id FROM trials {"WHERE " + " AND ".join(where) if where else ""}) s
          USING (trial_id)
        GROUP BY d.drug
        ORDER BY trials DESC, d.drug
        LIMIT {top}
    """
    return get_con().execute(sql, params).df()