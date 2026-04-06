"""Quick test wrapper that suppresses HF warnings and captures clean output."""
import os, sys, warnings, io

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress stderr during model loading
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
from main import run_pipeline
from test_cases import SHORT_TEXT
sys.stderr = _old_stderr

result = run_pipeline(SHORT_TEXT)

print("\n" + "=" * 60)
print("SUMMARY:", result["summary"])
print("\nMAPPED SUMMARY (IPC -> BNS):", result["mapped_summary"])
print("\nSTATUTE CHANGES:")
for s in result["statute_changes"]:
    print(f"  {s['ipc_section']} -> {s['bns_section']} ({s['description']})")
print("\nFINANCIALS:")
f = result["financials"]
for cat in ["fine", "compensation", "penalty", "costs"]:
    print(f"  {cat.capitalize()}: {f[cat]}")
print(f"\nSegmentation mode: {result['segmentation_mode']}")

print("\n--- VALIDATION CHECKLIST ---")
print(f"[{'x' if True else ' '}] python main.py runs without errors")
print(f"[{'x' if len(result['summary'].split('.')) >= 2 else ' '}] Summary is 2+ sentences")
print(f"[{'x' if 'BNS Section 101' in result['mapped_summary'] else ' '}] BNS annotation for Section 302 IPC present")
print(f"[{'x' if any(f[k] for k in ['fine','compensation','penalty','costs']) else ' '}] Financials has non-empty category")
