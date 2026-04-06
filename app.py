import re
import time
from collections import Counter

import streamlit as st
import pandas as pd

# Set UI styling and layout
st.set_page_config(
    page_title="Legal Intelligence Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Dark Legal Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #d2a84d !important;
        font-family: 'Garamond', serif !important;
        letter-spacing: 0.5px;
    }
    .summary-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 24px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #d2a84d;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(22, 27, 34, 0.6);
        border: 1px solid #444c56;
        padding: 10px;
        border-radius: 5px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import backend pipeline
with st.spinner("Initializing AI Models (GPU)..."):
    try:
        from main import run_pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")

# Sample Cases
SAMPLES = {
    "Select a Case...": "",
    "Medium: State vs. Sharma (Fraud)": """IN THE HIGH COURT OF DELHI
CRIMINAL APPEAL NO. 456 OF 2023
State vs. Rajesh Sharma

FACTS: The respondent was charged with cheating and fraud amounting to Rs. 50,00,000 under Section 420 IPC. He allegedly siphoned funds using dummy company accounts over two years.

ARGUMENTS: The defense argued lack of direct evidence and purely circumstantial transactions. The prosecution presented the forged bank guarantees alongside witness testimony.

JUDGMENT: The court finds the digital footprint overwhelming. The circumstantial evidence forms a complete chain pointing squarely to the respondent's guilt.

ORDERED: The respondent is convicted under Section 420 IPC and sentenced to 5 years rigorous imprisonment. A fine of Rs. 5,00,000 is imposed.""",
    "Long: Jadhav Murder Appeal": """IN THE SUPREME COURT OF INDIA
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


def clean_legal_text(text: str) -> str:
    """Wipes out repetitive PDF noise and headers."""
    lines = text.split("\n")
    line_counts = Counter(line.strip() for line in lines if line.strip())
    global_noise = {line for line, count in line_counts.items() if count > 3}

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in global_noise:
            continue
        if re.search(r"(?i)www\.manupatra\.com|manupatra", stripped):
            continue
        if re.search(r"(?i)page\s+\d+(\s+of\s+\d+)?", stripped):
            continue
        cleaned.append(line)

    return re.sub(r"\n\s*\n", "\n\n", "\n".join(cleaned)).strip()


def read_input_file(file):
    """Read uploaded .txt or .pdf file."""
    if file.name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages)
        return clean_legal_text(text)
    return str(file.read(), "utf-8")


# ─── UI ───
st.title("⚖️ Legal Intelligence Engine")
st.markdown("*An intelligent, context-aware abstractive summarization pipeline for Indian Legal Judgments.*")

with st.sidebar:
    st.header("Upload or Select Case")
    st.markdown("Use a pre-loaded case or upload a raw unstructured judgment text.")
    choice = st.selectbox("Pre-loaded Cases", list(SAMPLES.keys()))
    uploaded_file = st.file_uploader("Upload Judgment (.txt or .pdf)", type=["txt", "pdf"])
    st.divider()
    st.markdown("### ⚙️ Pipeline Components")
    st.markdown("""
    1. **Context Segmenter**
    2. **BNS & Financial Extractor**
    3. **Semantic Ranker** (`all-MiniLM-L6`)
    4. **Abstractive Generator** (GPU)
    """)

# Determine working text
raw_text = ""
if uploaded_file is not None:
    raw_text = read_input_file(uploaded_file)
elif choice != "Select a Case...":
    raw_text = SAMPLES[choice]

if raw_text:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📄 Raw Document")
        st.text_area("Judgment Text", raw_text, height=500, disabled=True)
        st.caption(f"Input Length: {len(raw_text.split())} words")

    with col2:
        st.subheader("✨ Intelligent Output")

        if st.button("Process Document", type="primary", use_container_width=True):
            with st.status("Analyzing Judgment Pipeline...", expanded=True) as status:
                start_t = time.time()
                st.write("🔍 Context Segmentation...")
                time.sleep(0.5)
                st.write("⚖️ Mapping IPC to BNS...")
                time.sleep(0.5)
                st.write("📊 Ranking Semantic Chunks...")
                result = run_pipeline(raw_text)
                total_t = time.time() - start_t
                status.update(label=f"Analysis Complete in {total_t:.2f}s!", state="complete", expanded=False)

            # Results
            st.markdown("### ✨ Case Analysis Brief")
            st.markdown(
                f'<div class="summary-card" style="min-height: 400px; white-space: pre-wrap;">'
                f'{result["mapped_summary"]}</div>',
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("### 🏛️ Extracted Legal Metadata")
            mc1, mc2 = st.columns(2)

            with mc1:
                st.markdown("#### IPC -> BNS Migrations")
                bns_changes = result.get("statute_changes", [])
                if bns_changes:
                    st.dataframe(pd.DataFrame(bns_changes), use_container_width=True, hide_index=True)
                else:
                    st.info("No outdated IPC codes detected.")

            with mc2:
                st.markdown("#### ✅ Verified Penalties")
                fin = result.get("financials", {})
                f_list = [
                    {"Type": cat.capitalize(), "Amount": item["amount"], "Reason": item["context"]}
                    for cat in ("fine", "compensation", "penalty", "costs")
                    for item in fin.get(cat, [])
                ]
                if f_list:
                    st.dataframe(pd.DataFrame(f_list), use_container_width=True, hide_index=True)
                else:
                    st.info("No verified penalties detected.")

                with st.expander("🔍 View All Monetary Mentions (including unverified)"):
                    raw = fin.get("raw_mentions", [])
                    if raw:
                        st.table(pd.DataFrame(raw))
                    else:
                        st.write("No mentions found.")
else:
    st.info("👈 Please select a case from the sidebar to begin processing.")
