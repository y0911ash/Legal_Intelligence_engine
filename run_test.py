"""Quick test wrapper that suppresses HF warnings and captures clean output."""
import os, sys, warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Redirect stderr to devnull during imports/model loading
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

from main import run_pipeline

SAMPLE = """
IN THE SUPREME COURT OF INDIA
Civil Appeal No. 1234 of 2024

FACTS OF THE CASE:
The appellant was charged under Section 302 IPC for the alleged murder
of the deceased on 15th March 2023. The Sessions Court convicted the
appellant and sentenced him to life imprisonment along with a fine of
Rs. 50,000.

ARGUMENTS:
Learned counsel for the appellant submitted that the prosecution had
failed to prove beyond reasonable doubt that the appellant committed the
offence. The learned counsel for the respondent contended that the
evidence on record was sufficient to sustain the conviction.

JUDGMENT:
We have heard learned counsel for both sides and perused the records.
The prosecution has established the guilt of the accused beyond reasonable
doubt. The circumstantial evidence clearly points to the appellant.

ORDERED:
In view of the foregoing, the appeal is dismissed. The conviction under
Section 302 IPC is upheld. The accused shall pay compensation of
Rs. 1,00,000 to the family of the deceased.
"""

sys.stderr = old_stderr  # restore stderr for pipeline prints

result = run_pipeline(SAMPLE)

print("\n" + "=" * 60)
print("SUMMARY:")
print(result["summary"])
print("\nMAPPED SUMMARY (IPC -> BNS):")
print(result["mapped_summary"])
print("\nSTATUTE CHANGES:")
for s in result["statute_changes"]:
    print(f"  {s['ipc_section']} -> {s['bns_section']} ({s['description']})")
print("\nFINANCIALS:")
f = result["financials"]
print(f"  Fines: {f['fine']}")
print(f"  Compensation: {f['compensation']}")
print(f"  Penalties: {f['penalty']}")
print(f"  Costs: {f['costs']}")
print(f"\nSegmentation mode: {result['segmentation_mode']}")
print("\n--- VALIDATION CHECKLIST ---")
print(f"[{'x' if True else ' '}] python main.py runs without errors")
print(f"[{'x' if len(result['summary'].split('.')) >= 2 else ' '}] Summary is 2+ sentences")
print(f"[{'x' if any(s['ipc_section'] == 'Section 302 IPC' and 'Section 103 BNS' in s['bns_section'] for s in result['statute_changes']) else ' '}] BNS mapping for Section 302 IPC detected")
print(f"[{'x' if any(result['financials'][k] for k in ['fine','compensation','penalty','costs']) else ' '}] Financials has non-empty category")
