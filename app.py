import streamlit as st
import pandas as pd
import time
import os

# Set UI styling and layout to wide
st.set_page_config(
    page_title="Legal Intelligence Engine", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Legal Theme
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    /* Headers */
    h1, h2, h3 {
        color: #d2a84d !important; /* Authentic Gold */
        font-family: 'Garamond', serif !important;
        letter-spacing: 0.5px;
    }
    /* Cards for Outputs */
    .summary-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 24px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-left: 5px solid #d2a84d;
    }
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: rgba(22, 27, 34, 0.6);
        border: 1px solid #444c56;
        padding: 10px;
        border-radius: 5px;
    }
    /* Hide streamer elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import the backend pipeline quietly
with st.spinner("Initializing AI Models (GPU)..."):
    try:
        from main import run_pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")

# Sample Data
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
We find no merit in this appeal. The conviction and life sentence under Section 302 IPC is upheld. Additionally, compensation of Rs. 2,00,000 must be paid to the victim's family."""
}

# ----------------- UI STRUCTURE -----------------

st.title("⚖️ Legal Intelligence Engine")
st.markdown("*An intelligent, context-aware abstractive summarization pipeline for Indian Legal Judgments.*")

# Sidebar
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

# Helper to read files
def read_input_file(file):
    if file.name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return str(file.read(), "utf-8")

# Determine the working text
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
                # Run the actual pipeline
                result = run_pipeline(raw_text)
                
                total_t = time.time() - start_t
                status.update(label=f"Analysis Complete in {total_t:.2f}s!", state="complete", expanded=False)
            
            # --- Results Presentation ---
            st.markdown(f'<div class="summary-card"><h4>AI Abstractive Summary</h4>{result["mapped_summary"]}</div>', unsafe_allow_html=True)
            
            # Financials & BNS
            st.divider()
            st.markdown("### 🏛️ Extracted Legal Metadata")
            
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("#### IPC -> BNS Migrations")
                bns_changes = result.get("statute_changes", [])
                if bns_changes:
                    # Convert to dataframe for clean table
                    df_bns = pd.DataFrame(bns_changes)
                    st.dataframe(df_bns, use_container_width=True, hide_index=True)
                else:
                    st.info("No outdated IPC codes detected.")
                    
            with mc2:
                st.markdown("#### Monetary Extraction")
                fin = result.get("financials", {})
                if any(v for k,v in fin.items()):
                    f_list = []
                    for k,v in fin.items():
                        if k == "raw_mentions": continue
                        if v: 
                            for entry in v:
                                amt = entry["amount"]
                                ctx = entry["context"].replace("\n", " ")
                                f_list.append({"Type": k.capitalize(), "Amount": amt, "Context (Reason)": ctx})
                    st.dataframe(pd.DataFrame(f_list), use_container_width=True, hide_index=True)
                else:
                    st.info("No financial penalties detected.")
                    
else:
    st.info("👈 Please select a case from the sidebar to begin processing.")
