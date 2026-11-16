"""Build a drug dictionary from the raw trial data, and store it in data/pharma.duckdb.

Creates two tables:
- drug_names:  one row per name   (name, drug)   -> which drug each name belongs to
- trial_drugs: one row per trial per drug (trial_id, drug)
Run once from the pharma-rag folder:  python rag/build_drugs.py
(Close notebooks that have the database open first.)"""
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "trials_australia_raw.json.gz"
DB_FILE = ROOT / "data" / "pharma.duckdb"

DRUG_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}
# Names that are not a specific drug
NOT_A_DRUG = {"placebo", "matching placebo", "placebos", "chemotherapy", "standard of care",
              "standard chemotherapy", "investigator's choice", "best supportive care", "saline",
              "normal saline", "vehicle", "control", "standard treatment", "usual care", "sham",
              "drug", "oral", "trade name", "iupac", "intrathecal", "or equivalent", "sub-study"}
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|mg/m2|mg/kg|iu|units?|%)\b.*$", re.I)
NOT_A_NAME = re.compile(r"\b(therapy|radiation|radiotherapy|chemotherapy|regimen|arm|cohort|"
                        r"inhibitor|antibody|study|standard|choice|combination|biosimilar|"
                        r"surgery|dissection|resection|keynote|or generic)\b", re.I)
SPLIT_COMPONENTS = re.compile(r"\s*(?:/|\+|,|;|\bplus\b|\band\b|\bwith\b)\s*", re.I)


def clean(name: str) -> str:
    """Lower-case a name and remove doses, symbols and extra spaces."""
    name = name.replace("®", "").replace("™", "")
    name = DOSE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" -.:")
    return name.lower()


def usable(name: str) -> bool:
    """Keep names that look like one specific drug."""
    return (3 <= len(name) <= 40 and len(name.split()) <= 4 and name not in NOT_A_DRUG
            and "placebo" not in name)


def synonyms(part: str) -> list[str]:
    """A name with brackets gives two synonyms: 'Pembrolizumab (KEYTRUDA)' -> pembrolizumab, keytruda."""
    inside = re.findall(r"\(([^)]*)\)", part)
    outside = re.sub(r"\([^)]*\)", " ", part)
    return [n for n in (clean(x) for x in [outside, *inside]) if usable(n)]


def main():
    with gzip.open(RAW_FILE, "rt") as f:
        raw = json.load(f)
    studies = raw["studies"] if isinstance(raw, dict) else raw

    self_count = Counter()        # how often a name is registered as an intervention's own name
    pair_count = Counter()        # how often a name is listed as an "other name" of a registered name
    trial_names = defaultdict(set)

    for s in studies:
        trial_id = s["protocolSection"]["identificationModule"]["nctId"]
        interventions = s["protocolSection"].get("armsInterventionsModule", {}).get("interventions", [])
        for i in interventions:
            if i.get("type") not in DRUG_TYPES:
                continue
            name = DOSE.sub("", i.get("name", ""))           # remove doses first: "80 mg/40 mg" is not two drugs
            components = [c for c in SPLIT_COMPONENTS.split(name) if c.strip()]
            for c in components:
                names = synonyms(c)
                if not names:
                    continue
                anchor = names[0]                            # the registered name
                self_count[anchor] += 1
                trial_names[trial_id].add(anchor)
                for n in names[1:]:                          # a name in brackets: "pembrolizumab (keytruda)"
                    pair_count[(n, anchor)] += 2             # strong evidence: counts as two
            if len(components) == 1:                         # other names only belong to a single drug
                names = synonyms(components[0])
                if names:
                    for o in i.get("otherNames", []) or []:
                        for n in synonyms(o):
                            pair_count[(n, names[0])] += 1

    # Each name belongs to the registered name it is linked to most often (its "anchor").
    # This is a vote, not a chain: a list of many drugs under one name (for example a regimen
    # like "CHOP") does not merge those drugs, because each is mostly registered as itself.
    votes = defaultdict(Counter)
    for name, n in self_count.items():
        votes[name][name] += n
    for (name, anchor), n in pair_count.items():
        # A name seen only once is accepted if it is never registered as its own drug
        # and does not look like a description ("radiation therapy", "anti-EGFR antibody").
        if n >= 2 or (self_count[name] == 0 and not NOT_A_NAME.search(name)):
            votes[name][anchor] += n
    anchor_of = {name: v.most_common(1)[0][0] for name, v in votes.items()}

    def final(name, hops=0):
        """Follow the anchors a few steps: azd9291 -> osimertinib -> osimertinib."""
        nxt = anchor_of.get(name, name)
        return name if nxt == name or hops > 3 else final(nxt, hops + 1)

    root_of = {name: final(name) for name in anchor_of}

    # Label each drug with its most used name, preferring a name without digits
    # (so "zipalertinib" rather than the code name "tas6417").
    members = defaultdict(list)
    for name, root in root_of.items():
        members[root].append(name)
    used = Counter(self_count)
    for (name, _), n in pair_count.items():
        used[name] += n
    label_of = {root: max(names, key=lambda n: (not any(ch.isdigit() for ch in n), used[n]))
                for root, names in members.items()}
    drug_of = {name: label_of[root] for name, root in root_of.items()}
    trial_names = {t: {drug_of.get(n, n) for n in ns} for t, ns in trial_names.items()}

    drug_names = pd.DataFrame(sorted(drug_of.items()), columns=["name", "drug"])
    trial_drugs = pd.DataFrame(sorted({(t, d) for t, ds in trial_names.items() for d in ds}),
                               columns=["trial_id", "drug"])

    con = duckdb.connect(str(DB_FILE))
    con.register("drug_names_df", drug_names)
    con.register("trial_drugs_df", trial_drugs)
    con.execute("CREATE OR REPLACE TABLE drug_names AS SELECT * FROM drug_names_df")
    con.execute("CREATE OR REPLACE TABLE trial_drugs AS SELECT * FROM trial_drugs_df")
    con.close()

    sizes = drug_names.groupby("drug").size().sort_values(ascending=False)
    print(f"Names: {len(drug_names)} | drugs: {drug_names['drug'].nunique()} | "
          f"trial-drug pairs: {len(trial_drugs)}")
    print("\nDrugs with the most names (check these look right):")
    print("(names: how many names each drug has)")
    for drug in sizes.head(10).index:
        names = drug_names.loc[drug_names.drug == drug, "name"]
        print(f"- {drug} ({len(names)} names): {', '.join(names.head(12))}")
    for drug in ["osimertinib", "erlotinib", "carboplatin", "pembrolizumab"]:
        names = drug_names.loc[drug_names.drug == drug, "name"]
        print(f"\n{drug}: {', '.join(names)}")


if __name__ == "__main__":
    main()