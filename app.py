import json
import os
import time

import streamlit as st


st.set_page_config(
    page_title="Legal Intelligence Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    defaults = {
        "step": 1,
        "raw_text": "",
        "pipeline_result": None,
        "selected_source": "No document loaded",
        "analysis_runtime": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_case() -> None:
    st.session_state.step = 1
    st.session_state.raw_text = ""
    st.session_state.pipeline_result = None
    st.session_state.selected_source = "No document loaded"
    st.session_state.analysis_runtime = None


def _build_metrics(raw_text: str, result: dict, runtime_sec: float | None) -> list[tuple[str, str]]:
    return [
        ("Input size", f"{len(raw_text.split()):,} words"),
        ("Segmentation", result.get("segmentation_mode", "unknown").title()),
        ("Ranked chunks", str(len(result.get("top_chunks", [])))),
        ("Runtime", f"{runtime_sec:.2f}s" if runtime_sec is not None else "Pending"),
    ]


def _build_financial_rows(result: dict) -> list[dict]:
    rows = []
    financials = result.get("financials", {})
    for category in ["fine", "compensation", "penalty", "costs"]:
        for entry in financials.get(category, []):
            rows.append(
                {
                    "Type": category.capitalize(),
                    "Amount": entry["amount"],
                    "Context": entry["context"].replace("\n", " ").strip(),
                }
            )
    return rows


def _build_export_payload(raw_text: str, result: dict, runtime_sec: float | None) -> dict:
    return {
        "input_text": raw_text,
        "input_word_count": len(raw_text.split()),
        "runtime_sec": runtime_sec,
        "summary": result.get("summary", ""),
        "mapped_summary": result.get("mapped_summary", ""),
        "segmentation_mode": result.get("segmentation_mode", ""),
        "top_chunk_count": len(result.get("top_chunks", [])),
        "top_chunks": result.get("top_chunks", []),
        "statute_changes": result.get("statute_changes", []),
        "financials": result.get("financials", {}),
    }


def _build_text_report(raw_text: str, result: dict, runtime_sec: float | None) -> str:
    lines = [
        "LEGAL INTELLIGENCE ENGINE REPORT",
        "=" * 48,
        "",
        "EXECUTIVE SUMMARY",
        result.get("mapped_summary", ""),
        "",
        "CASE METRICS",
        f"- Input size: {len(raw_text.split())} words",
        f"- Segmentation mode: {result.get('segmentation_mode', 'unknown')}",
        f"- Ranked chunks: {len(result.get('top_chunks', []))}",
        f"- Runtime: {runtime_sec:.2f}s" if runtime_sec is not None else "- Runtime: Pending",
        "",
        "STATUTE MAPPINGS",
    ]

    statute_changes = result.get("statute_changes", [])
    if statute_changes:
        for item in statute_changes:
            lines.append(
                f"- {item['ipc_section']} -> {item['bns_section']} ({item['description']})"
            )
    else:
        lines.append("- No outdated statutes detected.")

    lines.extend(["", "FINANCIAL IMPLICATIONS"])
    financial_rows = _build_financial_rows(result)
    if financial_rows:
        for row in financial_rows:
            lines.append(f"- {row['Type']}: {row['Amount']} | {row['Context']}")
    else:
        lines.append("- No monetary implications found.")

    lines.extend(["", "SOURCE JUDGMENT", raw_text.strip()])
    return "\n".join(lines)


def _card_marker(marker_class: str) -> None:
    st.markdown(f'<div class="card-marker {marker_class}"></div>', unsafe_allow_html=True)


def _build_metric_grid(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="metric-card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in items
    )
    return f'<div class="metrics-grid">{cards}</div>'


_init_state()

with st.spinner("Initializing AI Models..."):
    try:
        from main import run_pipeline
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        st.stop()


SAMPLES = {
    "Select a pre-loaded Case...": "",
    "State vs. Sharma (Fraud)": """IN THE HIGH COURT OF DELHI
CRIMINAL APPEAL NO. 456 OF 2023
State vs. Rajesh Sharma

FACTS: The respondent was charged with cheating and fraud amounting to Rs. 50,00,000 under Section 420 IPC. He allegedly siphoned funds using dummy company accounts over two years.

ARGUMENTS: The defense argued lack of direct evidence and purely circumstantial transactions. The prosecution presented the forged bank guarantees alongside witness testimony.

JUDGMENT: The court finds the digital footprint overwhelming. The circumstantial evidence forms a complete chain pointing squarely to the respondent's guilt.

ORDERED: The respondent is convicted under Section 420 IPC and sentenced to 5 years rigorous imprisonment. A fine of Rs. 5,00,000 is imposed.""",
    "Jadhav Murder Appeal": """IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION
CRIMINAL APPEAL NO. 993 OF 2021
Sunil Devidas Patil vs. State of Maharashtra

FACTS OF THE CASE:
The appellant, Sunil, along with co-accused Santosh, was charged under Section 302 and 392 IPC for the murder and robbery of the deceased goldsmith on 15-08-2015. The Session Court convicted both based on last-seen theory and recovery of stolen ornaments.

ARGUMENTS:
Appellant argued that recovery at his behest was planted and there were discrepancies in eyewitness accounts. The State maintained the recovery under Section 27 of Evidence Act was solid.

JUDGMENT:
Upon careful perusal of the trial record, we find that the High Court rightly affirmed the Session Court's judgment. The recovery of the murder weapon and stolen gold directly links the appellant to the crime. Minor discrepancies in witness testimony are natural.

ORDERED:
We find no merit in this appeal. The conviction and life sentence under Section 302 IPC is upheld. Additionally, compensation of Rs. 2,00,000 must be paid to the victim's family.""",
}


st.markdown(
    """
    <section class="modern-hero">
        <div class="hero-copy">
            <span class="eyebrow">Legal Intelligence Engine</span>
            <h1>Transform raw rulings into strategic intelligence</h1>
            <p>
                Load a judgment, run the automated analysis, and review the intelligence dashboard. 
                Our engine automatically flags actionable components and penal clauses.
            </p>
        </div>
        <div class="hero-actions">
            <div class="hero-panel">
                <span class="hero-label">Workflow</span>
                <strong>Input</strong>
                <strong>Analyze</strong>
                <strong>Review</strong>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

selected_text = ""
uploaded_file = None
choice = "Select a pre-loaded Case..."
pasted_text = ""

progress_value = {1: 34, 2: 68, 3: 100}.get(st.session_state.step, 34)
with st.container():
    _card_marker("shell-card-marker")
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <span class="section-kicker">Review stages</span>
                <h2>Guided case workflow</h2>
            </div>
            <div class="progress-note">Current stage: Step {st.session_state.step} of 3</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(progress_value / 100)
    progress_cols = st.columns(3)
    progress_labels = [
        ("01", "Upload document", st.session_state.step >= 1),
        ("02", "Document analysis", st.session_state.step >= 2),
        ("03", "Review findings", st.session_state.step >= 3),
    ]
    for column, (number, label, active) in zip(progress_cols, progress_labels):
        tone = "progress-chip active" if active else "progress-chip"
        column.markdown(
            f'<div class="{tone}"><span>{number}</span><strong>{label}</strong></div>',
            unsafe_allow_html=True,
        )

with st.container():
    _card_marker("shell-card-marker")
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <span class="section-kicker">Step 01</span>
                <h2>Load a judgment</h2>
            </div>
            <div class="progress-note">Choose a sample case, upload a file, or paste the text directly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, detail_col = st.columns([1.4, 1])

    with input_col:
        input_tab1, input_tab2 = st.tabs(["Quick Start", "Paste Judgment"])

        with input_tab1:
            sample_col, upload_col = st.columns(2)
            with sample_col:
                choice = st.selectbox(
                    "Choose a sample judgment",
                    list(SAMPLES.keys()),
                    index=0,
                    key="choice_select",
                )
            with upload_col:
                uploaded_file = st.file_uploader(
                    "Upload judgment (.txt or .pdf)",
                    type=["txt", "pdf"],
                )

        with input_tab2:
            pasted_text = st.text_area(
                "Paste judgment text",
                height=260,
                placeholder="Paste a judgment here to generate an executive legal briefing.",
                key="paste_area",
            )

    with detail_col:
        st.markdown("##### What the analysis returns")
        st.markdown(
            """
            - A focused executive summary built from ranked legal sections
            - A multi-pass abstractive summary of **Facts**, **Arguments**, and the **Verdict**
            - Extraction of structured **fines & penalties**
            - Statutory migration from **IPC → BNS**
            - A technical trace showing how the AI model prioritized the judgment text
            """
        )

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        selected_text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)
    else:
        selected_text = str(uploaded_file.read(), "utf-8")
    st.session_state.selected_source = f"Uploaded file: {uploaded_file.name}"
elif pasted_text.strip():
    selected_text = pasted_text
    st.session_state.selected_source = "Pasted judgment text"
elif choice != "Select a pre-loaded Case...":
    selected_text = SAMPLES[choice]
    st.session_state.selected_source = f"Sample case: {choice}"

if selected_text:
    st.session_state.raw_text = selected_text

preview_col, action_col = st.columns([1.6, 1])
with preview_col:
    if st.session_state.raw_text:
        st.markdown("##### Document preview")
        st.caption(st.session_state.selected_source)
        st.code(st.session_state.raw_text[:1200] + ("..." if len(st.session_state.raw_text) > 1200 else ""))
    else:
        st.markdown('<div class="empty-state-card">Load a document to preview the judgment here.</div>', unsafe_allow_html=True)

with action_col:
    word_count = len(st.session_state.raw_text.split()) if st.session_state.raw_text else 0
    st.markdown("##### Case snapshot")
    st.metric("Word count", f"{word_count:,}")
    st.metric("Source", "Ready" if st.session_state.raw_text else "Waiting")
    if st.button(
        "Load Document Into Workspace",
        type="primary",
        use_container_width=True,
        disabled=not bool(st.session_state.raw_text),
    ):
        st.session_state.step = 2
        st.rerun()
    if st.button("Reset Workspace", use_container_width=True):
        _reset_case()
        st.rerun()

if st.session_state.step >= 2:
    with st.container():
        _card_marker("shell-card-marker")
        st.markdown(
            """
            <div class="section-heading">
                <div>
                    <span class="section-kicker">Step 02</span>
                    <h2>Process judgment document</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_col, run_col = st.columns([1.6, 1])
        with metrics_col:
            st.markdown(
                _build_metric_grid(
                    [
                        ("Selected source", st.session_state.selected_source),
                        ("Input size", f"{len(st.session_state.raw_text.split()):,} words"),
                        (
                            "Result status",
                            "Ready to analyze" if st.session_state.pipeline_result is None else "Analysis available",
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )

        with run_col:
            st.markdown("##### Analysis controls")
            if st.button("Analyze Judgement", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                with st.status("Analyzing judgment...", expanded=True) as status:
                    start_time = time.time()
                    st.write("Segmenting the judgment into legal sections.")
                    progress_bar.progress(0.2)
                    time.sleep(0.2)
                    st.write("Extracting statutes and financial references.")
                    progress_bar.progress(0.45)
                    time.sleep(0.2)
                    st.write("Ranking semantically important legal chunks.")
                    progress_bar.progress(0.7)
                    result = run_pipeline(st.session_state.raw_text)
                    runtime_sec = time.time() - start_time
                    progress_bar.progress(1.0)
                    status.update(
                        label=f"Analysis complete in {runtime_sec:.2f}s",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.pipeline_result = result
                st.session_state.analysis_runtime = runtime_sec
                st.session_state.step = 3
                st.rerun()

            if st.session_state.pipeline_result is not None:
                st.success("A report is available below and ready to export.")

if st.session_state.step >= 3 and st.session_state.pipeline_result is not None:
    result = st.session_state.pipeline_result
    runtime_sec = st.session_state.analysis_runtime
    financial_rows = _build_financial_rows(result)
    export_payload = _build_export_payload(st.session_state.raw_text, result, runtime_sec)
    text_report = _build_text_report(st.session_state.raw_text, result, runtime_sec)

    with st.container():
        _card_marker("shell-card-marker")
        st.markdown(
            """
            <div class="section-heading">
                <div>
                    <span class="section-kicker">Step 03</span>
                    <h2>Review the intelligence report</h2>
                </div>
                <div class="progress-note">Executive summary first, technical evidence second.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_columns = st.columns(4)
        for column, (label, value) in zip(
            metric_columns,
            _build_metrics(st.session_state.raw_text, result, runtime_sec),
        ):
            column.markdown(
                f'<div class="metric-card metric-card-compact"><span>{label}</span><strong>{value}</strong></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <span class="section-kicker">Executive summary</span>
                <h3>Outcome-aware case briefing</h3>

{result["mapped_summary"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        insight_col, export_col = st.columns([1.7, 1])
        with insight_col:
            statute_tab, finance_tab = st.tabs(["Statute updates", "Financial implications"])

        with statute_tab:
            if result.get("statute_changes"):
                st.markdown("##### IPC to BNS migrations")
                statute_html = (
                    '<table class="fancy-table"><thead><tr><th>Original</th><th>Modern</th><th>Description</th></tr></thead><tbody>'
                )
                for item in result["statute_changes"]:
                    statute_html += (
                        f"<tr><td>{item['ipc_section']}</td>"
                        f"<td><span class='statute-highlight'>{item['bns_section']}</span></td>"
                        f"<td>{item['description']}</td></tr>"
                    )
                statute_html += "</tbody></table>"
                st.markdown(statute_html, unsafe_allow_html=True)
            else:
                st.info("No outdated IPC references were detected in this judgment.")

        with finance_tab:
            if financial_rows:
                finance_html = (
                    '<table class="fancy-table"><thead><tr><th>Category</th><th>Amount</th><th>Context</th></tr></thead><tbody>'
                )
                for item in financial_rows:
                    finance_html += (
                        f"<tr><td>{item['Type']}</td>"
                        f"<td><span class='statute-highlight'>{item['Amount']}</span></td>"
                        f"<td>{item['Context']}</td></tr>"
                    )
                finance_html += "</tbody></table>"
                st.markdown(finance_html, unsafe_allow_html=True)
            else:
                st.info("No financial implications were extracted from the judgment.")

        with export_col:
            st.markdown("##### Export the report")
            st.caption("JSON export is fully structured. A text briefing is available for quick sharing.")
            st.download_button(
                "Download JSON Report",
                data=json.dumps(export_payload, indent=2),
                file_name="legal_intelligence_report.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Download Case Brief (.txt)",
                data=text_report,
                file_name="legal_intelligence_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
            if st.button("Start New Case Analysis", type="primary", use_container_width=True):
                _reset_case()
                st.rerun()

        st.markdown("##### Side-by-side review")
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            st.markdown("###### Original judgment")
            st.caption(st.session_state.selected_source)
            st.text_area(
                "Source judgment",
                st.session_state.raw_text,
                height=360,
                disabled=True,
                key="source_judgment_view",
                label_visibility="collapsed",
            )

        with compare_col2:
            st.markdown("###### Extracted intelligence")
            st.text_area(
                "Generated legal briefing",
                result["mapped_summary"],
                height=180,
                disabled=True,
                key="generated_summary_view",
                label_visibility="collapsed",
            )
            if financial_rows:
                st.markdown("**Financial highlights**")
                for row in financial_rows[:4]:
                    st.markdown(f"- {row['Type']}: {row['Amount']}")
            if result.get("statute_changes"):
                st.markdown("**Statute highlights**")
                for item in result["statute_changes"][:4]:
                    st.markdown(f"- {item['ipc_section']} -> {item['bns_section']}")

        with st.expander("Technical analysis", expanded=False):
            st.markdown(
                f"**Segmentation mode:** {result['segmentation_mode'].upper()}  \n"
                f"**Top chunks selected:** {len(result['top_chunks'])}"
            )
            for chunk, score, section in result["top_chunks"][:5]:
                st.markdown(f"**{section.upper()}** (Score: {score})")
                st.caption(chunk[:280] + "...")
