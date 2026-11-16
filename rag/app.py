"""Streamlit app: an AI research assistant for clinical trials in Australia.
Run from the pharma-rag folder:  streamlit run rag/app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clinical Trials Assistant", page_icon="🔬", layout="centered")

EXAMPLES_FILE = Path(__file__).resolve().parent.parent / "data" / "example_answers.json"
MAX_QUESTIONS = 3            # live questions per visitor (the free LLM plan has a daily limit)
RED = "#C62828"

st.markdown(f"""
<style>
.title-row {{ display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:4px; }}
.title-icon {{ width:52px; height:52px; border-radius:50%; background:#F5F5F5; display:flex;
              align-items:center; justify-content:center; }}
.title-icon span {{ font-family:'Material Symbols Rounded'; font-size:30px; color:{RED}; }}
.title-row h1 {{ margin:0; padding:0; font-size:2rem; }}
.subtitle {{ text-align:center; color:#666; margin-bottom:1.6rem; }}
.label {{ font-size:0.75rem; letter-spacing:0.08em; color:#888; margin:0.4rem 0 0.4rem; }}
div.stButton > button {{ border-radius:999px; border:1px solid #ddd; background:white; }}
div.stButton > button:hover {{ border-color:{RED}; color:{RED}; }}
div[data-testid="stChatInput"] textarea:focus {{ border-color:#bbb !important; box-shadow:none !important; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded" rel="stylesheet">
<div class="title-row"><div class="title-icon"><span>science</span></div><h1>Clinical Trials Assistant</h1></div>
<div class="subtitle">Ask about 12,000+ clinical trials running in Australia.<br>
Answers cite the trials they are based on.</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_examples() -> dict:
    return json.loads(EXAMPLES_FILE.read_text()) if EXAMPLES_FILE.exists() else {}


@st.cache_resource(show_spinner="Loading the trial database…")
def load_pipeline():
    """Import the pipeline once per server, and load the database and embedding model."""
    import answer as pipeline
    import search
    search.get_con()
    search.get_model()
    return pipeline


def show_result(result: dict):
    """The answer, plus a fold-out with how it was found."""
    st.markdown(result["answer"])
    with st.expander("How this answer was found"):
        terms = result.get("terms", {})
        drugs = terms.get("interventions") or {}
        st.markdown(
            f"**Search terms**  \n"
            f"Conditions: {', '.join(terms.get('conditions') or []) or '—'}  \n"
            f"Drugs: {', '.join(drugs) or '—'}  \n"
            f"Status: {terms.get('status') or 'any'} · Sponsors: {', '.join(terms.get('sponsors') or []) or 'any'}")
        st.markdown(f"**{result.get('total', 0)} matching trials.** The most relevant:")
        trials = pd.DataFrame(result.get("trials", []))
        if len(trials):
            st.dataframe(trials, hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("link", display_text="open")})
        if result.get("counts"):
            st.markdown("**Drugs across all matching trials**")
            st.dataframe(pd.DataFrame(result["counts"]), hide_index=True, width="stretch")
        if result.get("changes"):
            st.markdown("**Corrected by the check:**  \n" + "  \n".join(f"- {c}" for c in result["changes"]))


if "history" not in st.session_state:
    st.session_state.history = []          # list of (question, result)
    st.session_state.live_questions = 0

examples = load_examples()
if examples:
    st.markdown('<div class="label">TRY AN EXAMPLE QUESTION</div>', unsafe_allow_html=True)
    cols = st.columns(len(examples))
    for col, (label, stored) in zip(cols, examples.items()):
        if col.button(label, width="stretch"):
            st.session_state.history.append((stored["question"], stored))

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/science:"):
        show_result(result)

left = MAX_QUESTIONS - st.session_state.live_questions
prompt = st.chat_input("Ask about a drug, a disease or a sponsor…" if left > 0
                       else "You have used your questions. The examples still work.",
                       disabled=left <= 0)
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/science:"):
        with st.spinner("Searching the trials and writing an answer…"):
            try:
                pipeline = load_pipeline()
                result = pipeline.to_plain(pipeline.answer(prompt))
            except Exception as error:
                name = type(error).__name__
                result = None
                if "RateLimit" in name:
                    st.warning("The free language model limit has been reached. Please try again later, "
                               "or click one of the example questions.")
                else:
                    st.error(f"Something went wrong ({name}). Please try another question.")
        if result:
            show_result(result)
            st.session_state.history.append((prompt, result))
            st.session_state.live_questions += 1
            st.rerun()                     # redraw, so the question counter and input box update

st.caption(f"Questions left: {MAX_QUESTIONS - st.session_state.live_questions} of {MAX_QUESTIONS} · "
           "Data: ClinicalTrials.gov, trials with at least one site in Australia. Not medical advice.")
