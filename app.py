"""Previous Year Question Paper Analyzer - PS37.

Upload one or more previous-year question papers (PDF), and the app extracts
the questions, asks an LLM to tag each question with its syllabus topic as
JSON, counts the topics with pandas, and shows a bar chart of the topics that
came up most often.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import analysis, extract, tagger  # noqa: E402
from core.llm_client import (  # noqa: E402
    DEFAULT_MODEL,
    LLMError,
    load_api_key,
    save_api_key,
)

SAMPLES_DIR = ROOT / "samples"

st.set_page_config(
    page_title="Question Paper Analyzer",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------------------ styles ---

st.markdown(
    """
<style>
.block-container {padding-top: 3.5rem; max-width: 1200px;}
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    color: #fff; border-radius: 18px; padding: 28px 32px; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, .25);
}
.hero h1 {color: #fff; font-size: 2.05rem; margin: 0 0 .35rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.9); font-size: 1.05rem; margin: 0;}
.hero .tag {
    display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
    padding: 3px 12px; font-size: .78rem; letter-spacing: .04em; margin-bottom: .7rem;
}
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: .4rem 0 1.2rem;}
.step {
    background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 16px 18px;
}
.step .n {
    width: 28px; height: 28px; border-radius: 50%; background: #EEF2FF; color: #4F46E5;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 8px;
}
.step b {display: block; margin-bottom: 2px; color: #111827;}
.step span {color: #6B7280; font-size: .88rem;}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: .6rem 0 1rem;}
.kpi {
    background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 16px 18px;
    border-top: 4px solid #4F46E5;
}
.kpi .v {font-size: 1.9rem; font-weight: 700; color: #111827; line-height: 1.1;}
.kpi .l {color: #6B7280; font-size: .85rem; margin-top: 4px;}
.top-topic {
    background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 14px;
    padding: 16px 20px; margin-bottom: 1rem; color: #065F46;
}
.top-topic b {font-size: 1.15rem; color: #047857;}
.rank {display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #F3F4F6;}
.rank:last-child {border-bottom: none;}
.rank .badge {
    min-width: 30px; height: 30px; border-radius: 8px; background: #4F46E5; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
}
.rank .t {font-weight: 600; color: #111827;}
.rank .s {color: #6B7280; font-size: .85rem;}
@media (max-width: 800px) {
    .steps, .kpis {grid-template-columns: repeat(2, 1fr);}
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- helpers ---


def init_state() -> None:
    """Set up session state on first run."""
    defaults = {
        "df": pd.DataFrame(),
        "topics": [],
        "analysis_done": False,
        "error": None,
        "raw_json": [],
        "api_key": load_api_key(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results() -> None:
    """Reset the analysis so a new run starts from scratch."""
    st.session_state["df"] = pd.DataFrame()
    st.session_state["topics"] = []
    st.session_state["analysis_done"] = False
    st.session_state["error"] = None
    st.session_state["raw_json"] = []


def run_analysis(files: dict[str, bytes], api_key: str, model: str, temperature: float) -> None:
    """Extract, tag and aggregate the papers."""
    clear_results()
    status = st.status("Analysing papers...", expanded=True)

    status.write("**Step 1/3** · Extracting questions from the PDFs")
    papers_text: dict[str, str] = {}
    all_questions: list = []
    for name, data in files.items():
        result = extract.extract_text_from_pdf(data, name)
        if result.error:
            status.write(f"⚠️ {name}: {result.error}")
            continue
        papers_text[name] = result.full_text
        all_questions.extend(result.questions)
        status.write(f"✅ {name}: {result.pages} page(s), {len(result.questions)} question(s)")

    if not all_questions:
        status.update(label="No questions found", state="error")
        st.session_state["error"] = (
            "No questions could be found in the PDFs. Check that they contain selectable "
            "text and use numbered questions. Scanned papers need OCR first."
        )
        return

    try:
        status.write("**Step 2/3** · Asking Gemini for a shared topic list (JSON)")
        topics = tagger.build_vocabulary(papers_text, api_key, model, temperature)
        if not topics:
            raise LLMError("The model did not return a usable topic list.")
        status.write(f"✅ {len(topics)} topics: {', '.join(topics[:6])}" + (" ..." if len(topics) > 6 else ""))

        status.write(f"**Step 3/3** · Tagging {len(all_questions)} questions with a topic (JSON)")
        raw_json: list = []
        tagged = tagger.tag_questions(
            all_questions, topics, api_key, model, temperature, raw_log=raw_json
        )
    except LLMError as exc:
        status.update(label="Gemini request failed", state="error")
        st.session_state["error"] = friendly_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        status.update(label="Something went wrong", state="error")
        st.session_state["error"] = f"Unexpected error: {exc}"
        return

    st.session_state["df"] = analysis.to_dataframe(tagged)
    st.session_state["topics"] = topics
    st.session_state["raw_json"] = raw_json
    st.session_state["analysis_done"] = True
    status.update(label="Analysis complete", state="complete", expanded=False)


def friendly_error(message: str) -> str:
    """Turn raw API errors into advice a student can act on."""
    if "429" in message:
        return (
            "Gemini's free quota is used up for now (HTTP 429). Wait a minute and try "
            "again, or set the model to `gemini-3.5-flash-lite` under Advanced settings."
        )
    if "503" in message:
        return "Gemini is overloaded right now (HTTP 503). Please try again in a few minutes."
    if "400" in message or "401" in message or "403" in message:
        return (
            "Gemini rejected the API key. Use **Advanced settings → Replace API key** in the sidebar and paste "
            "a key from Google AI Studio."
        )
    return message


def topic_bar_chart(counts: pd.DataFrame) -> alt.Chart:
    """Horizontal bar chart with full topic names and count labels."""
    data = counts.copy()
    base = alt.Chart(data).encode(
        y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
        x=alt.X("Questions:Q", title="Number of questions", axis=alt.Axis(tickMinStep=1)),
        tooltip=["Topic", "Questions", "% of Questions", "Total Marks", "Papers Covered"],
    )
    bars = base.mark_bar(cornerRadiusEnd=6, height=22).encode(
        color=alt.Color("Questions:Q", scale=alt.Scale(scheme="purples"), legend=None)
    )
    labels = base.mark_text(align="left", dx=6, fontWeight="bold", color="#374151").encode(
        text="Questions:Q"
    )
    return (bars + labels).properties(height=max(220, 38 * len(data)))


def topic_paper_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Topic × paper heatmap showing where each topic was asked."""
    data = df.groupby(["Topic", "Paper"]).size().reset_index(name="Questions")
    base = alt.Chart(data).encode(
        x=alt.X("Paper:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Topic:N", title=None, axis=alt.Axis(labelLimit=320)),
    )
    rect = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Questions:Q", scale=alt.Scale(scheme="purples"), legend=None),
        tooltip=["Topic", "Paper", "Questions"],
    )
    text = base.mark_text(fontWeight="bold").encode(
        text="Questions:Q",
        color=alt.condition("datum.Questions > 1", alt.value("white"), alt.value("#1F2937")),
    )
    return (rect + text).properties(height=max(220, 34 * data["Topic"].nunique()))


# ----------------------------------------------------------------- sidebar ---

init_state()

with st.sidebar:
    # The key section only appears until a key is saved. After that, nothing
    # about the key is shown; it can be replaced from Advanced settings.
    if not st.session_state["api_key"]:
        st.markdown("### 🔑 Gemini API key")
        st.caption("Enter it once. Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).")
        with st.form("key_form", border=False):
            new_key = st.text_input("API key", type="password", placeholder="Paste your key here")
            remember = st.checkbox("Remember on this computer", value=True,
                                   help="Saved to the project's .env file, which is never uploaded to GitHub.")
            if st.form_submit_button("Save key", type="primary", width="stretch"):
                if new_key.strip():
                    if remember:
                        save_api_key(new_key)
                    st.session_state["api_key"] = new_key.strip()
                    st.rerun()
                else:
                    st.error("Please paste a key first.")
        st.divider()

    st.markdown("### ⚙️ Settings")
    with st.expander("Advanced settings"):
        model = st.text_input("Model", value=DEFAULT_MODEL,
                              help="Fallback models are tried automatically if this one is busy.")
        temperature = st.slider("Temperature", 0.0, 0.7, 0.0, 0.1, help="0.0 gives the most consistent tags.")
        if st.session_state["api_key"]:
            st.caption("An API key is saved on this computer. To replace it, paste a new one below; "
                       "the saved key is kept until you do.")
            with st.form("replace_key_form", border=False, clear_on_submit=True):
                replacement = st.text_input("New API key", type="password", placeholder="Paste new key")
                if st.form_submit_button("Replace API key") and replacement.strip():
                    save_api_key(replacement)
                    st.session_state["api_key"] = replacement.strip()
                    st.rerun()

    if st.session_state["analysis_done"] and st.button("🗑️ Clear results", width="stretch"):
        clear_results()
        st.rerun()

    st.caption("PS37 · LLM Mini Project")

api_key = st.session_state["api_key"]

# -------------------------------------------------------------------- main ---

st.markdown(
    """
<div class="hero">
  <div class="tag">PS37 · GEN AI MINI PROJECT</div>
  <h1>📄 Previous Year Question Paper Analyzer</h1>
  <p>Upload past exam papers and see which topics are asked most often, so you know what to revise first.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state["analysis_done"]:
    st.markdown(
        """
<div class="steps">
  <div class="step"><div class="n">1</div><b>Extract</b><span>pdfplumber reads each paper and splits it into questions</span></div>
  <div class="step"><div class="n">2</div><b>Tag as JSON</b><span>Gemini labels every question with one syllabus topic</span></div>
  <div class="step"><div class="n">3</div><b>Count</b><span>pandas counts questions and marks per topic</span></div>
  <div class="step"><div class="n">4</div><b>Visualise</b><span>A bar chart shows the most frequent topics</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

if not api_key:
    st.info("👈 Add your Gemini API key in the sidebar. You only need to do this once.")

with st.container(border=True):
    st.markdown("#### 📥 Question papers")
    source = st.radio(
        "Source",
        ["Upload my PDFs", f"Use the {len(list(SAMPLES_DIR.glob('*.pdf')))} sample papers"],
        horizontal=True,
        label_visibility="collapsed",
    )

    files: dict[str, bytes] = {}
    if source.startswith("Upload"):
        uploaded = st.file_uploader(
            "Drop PDF question papers here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Text-based PDFs work best. Scanned papers need OCR first.",
        )
        for index, file in enumerate(uploaded or [], start=1):
            name = file.name if file.name not in files else f"{Path(file.name).stem} ({index}).pdf"
            files[name] = file.getvalue()
    else:
        for path in sorted(SAMPLES_DIR.glob("*.pdf")):
            files[path.name] = path.read_bytes()
        st.caption("Sample DBMS papers: " + ", ".join(files))

    left, right = st.columns([3, 1])
    left.caption(f"**{len(files)}** paper(s) selected" if files else "No papers selected yet.")
    clicked = right.button(
        "🔍 Analyse papers",
        type="primary",
        width="stretch",
        disabled=not files or not api_key,
    )

if clicked:
    run_analysis(files, api_key, model, temperature)

if st.session_state["error"]:
    st.error(st.session_state["error"])

# ----------------------------------------------------------------- results ---

if st.session_state["analysis_done"]:
    df: pd.DataFrame = st.session_state["df"]
    topics: list[str] = st.session_state["topics"]
    counts = analysis.topic_counts(df)
    total_marks = int(df["Marks"].sum()) if df["Marks"].notna().any() else 0

    st.markdown(
        f"""
<div class="kpis">
  <div class="kpi"><div class="v">{df['Paper'].nunique()}</div><div class="l">Papers analysed</div></div>
  <div class="kpi"><div class="v">{len(df)}</div><div class="l">Questions tagged</div></div>
  <div class="kpi"><div class="v">{df['Topic'].nunique()}</div><div class="l">Distinct topics</div></div>
  <div class="kpi"><div class="v">{total_marks}</div><div class="l">Total marks</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    if counts.empty:
        st.warning("No topic counts could be built from the results.")
        st.stop()

    top = counts.iloc[0]
    st.markdown(
        f"""<div class="top-topic">🎯 Revise first: <b>{top['Topic']}</b> &nbsp;·&nbsp;
        {int(top['Questions'])} question(s), {top['% of Questions']}% of all questions,
        asked in {int(top['Papers Covered'])} paper(s)</div>""",
        unsafe_allow_html=True,
    )

    paper_count = df["Paper"].nunique()

    # --- Bar chart of top topics (always visible) ---------------------------
    with st.container(border=True):
        chart_col, list_col = st.columns([3, 2], gap="large")
        with chart_col:
            st.markdown("#### 📊 Most frequently asked topics")
            st.altair_chart(topic_bar_chart(counts.head(12)), width="stretch")
        with list_col:
            st.markdown("#### 🏆 Top 5 topics")
            rows = "".join(
                f"""<div class="rank"><div class="badge">{rank}</div><div>
                <div class="t">{row['Topic']}</div>
                <div class="s">{int(row['Questions'])} question(s) · {int(row['Total Marks'])} marks ·
                {int(row['Papers Covered'])} of {paper_count} paper(s)</div></div></div>"""
                for rank, (_, row) in enumerate(counts.head(5).iterrows(), start=1)
            )
            st.markdown(rows, unsafe_allow_html=True)

    # --- Frequently asked topics with their questions (always visible) -----
    st.markdown("### ⭐ Frequently asked topics")
    st.caption(
        "Every topic, ranked by how often it is asked. Open a topic to see the actual "
        "questions from the papers, so you know what to prepare."
    )
    for rank, (_, row) in enumerate(counts.iterrows(), start=1):
        topic = row["Topic"]
        papers_in = int(row["Papers Covered"])
        share = papers_in / paper_count if paper_count else 0
        if share >= 0.8:
            level = "🔥 Very frequent"
        elif share >= 0.5:
            level = "⭐ Frequent"
        else:
            level = "• Occasional"
        label = (
            f"{rank}. {topic}  —  {level}  ·  asked {int(row['Questions'])} time(s) "
            f"in {papers_in} of {paper_count} paper(s)"
        )
        with st.expander(label, expanded=rank == 1):
            topic_rows = df[df["Topic"] == topic].sort_values(["Paper", "Q No"])
            for _, q in topic_rows.iterrows():
                marks = f" · {int(q['Marks'])} marks" if pd.notna(q["Marks"]) else ""
                st.markdown(f"- {q['Question']}  \n  <small>📄 {q['Paper']} · Q{q['Q No']}{marks}</small>",
                            unsafe_allow_html=True)

    st.markdown("### 🔎 Details")
    tab_overview, tab_papers, tab_questions, tab_json, tab_export = st.tabs(
        ["📋 Topic table", "🗂️ By paper", "📝 All questions", "🧾 LLM JSON", "⬇️ Export"]
    )

    with tab_overview:
        st.markdown("##### Topic frequency table")
        st.dataframe(
            counts,
            hide_index=True,
            width="stretch",
            column_config={
                "Questions": st.column_config.ProgressColumn(
                    "Questions", format="%d", min_value=0, max_value=int(counts["Questions"].max())
                ),
                "% of Questions": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    with tab_papers:
        st.markdown("##### Where each topic was asked")
        st.altair_chart(topic_paper_heatmap(df), width="stretch")
        st.markdown("##### Per-paper summary")
        st.dataframe(analysis.papers_covered(df), hide_index=True, width="stretch")

    with tab_questions:
        chosen = st.multiselect("Filter by topic", counts["Topic"].tolist(), placeholder="All topics")
        view = df if not chosen else df[df["Topic"].isin(chosen)]
        st.caption(f"Showing {len(view)} of {len(df)} questions")
        st.dataframe(
            view[["Paper", "Q No", "Marks", "Topic", "Question"]],
            hide_index=True,
            width="stretch",
            column_config={"Question": st.column_config.TextColumn(width="large")},
        )

    with tab_json:
        st.caption(
            "Exactly what Gemini returned: stage 1 is the shared topic list, "
            "stage 2 is the topic assigned to each question id."
        )
        st.markdown("**Topic vocabulary:** " + " ".join(f"`{t}`" for t in topics))
        left_json, right_json = st.columns([2, 3], gap="large")
        with left_json:
            st.markdown("**Stage 1 · shared topic list**")
            st.code(json.dumps(topics, indent=2, ensure_ascii=False), language="json")
        with right_json:
            st.markdown("**Stage 2 · topic for each question id**")
            for batch in st.session_state["raw_json"]:
                lines = ",\n".join(
                    "  " + json.dumps(item, ensure_ascii=False) for item in batch["response"]
                )
                st.caption(f"Batch {batch['batch']}")
                st.code("[\n" + lines + "\n]", language="json")

    with tab_export:
        st.caption("Download the results to include in your report.")
        col_a, col_b, col_c = st.columns(3)
        col_a.download_button(
            "📈 Topic counts (CSV)",
            counts.to_csv(index=False).encode("utf-8"),
            file_name="topic_counts.csv",
            mime="text/csv",
            width="stretch",
        )
        col_b.download_button(
            "📝 Tagged questions (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="tagged_questions.csv",
            mime="text/csv",
            width="stretch",
        )
        col_c.download_button(
            "🧾 Tagged questions (JSON)",
            json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="tagged_questions.json",
            mime="application/json",
            width="stretch",
        )
