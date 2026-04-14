
import sys
import os
import re

# Add current directory to path so we can import the pipeline
sys.path.append(os.getcwd())

from pipeline.financial_extractor import extract_financials

test_cases = [
    {
        "name": "Standard Case (Sharma)",
        "text": """
        ORDERED: The respondent is convicted and sentenced. A fine of Rs. 5,00,000 is imposed.
        """
    },
    {
        "name": "Suffix Case (Lakhs)",
        "text": """
        The appellant sold a flat for Rs. 45,00,000. 
        The complainant is awarded compensation of Rs. 5 lakhs.
        """
    },
    {
        "name": "Abbreviation Context (Rs. and No.)",
        "text": """
        In Case No. 123, the court ordered a penalty of Rs. 10,000/-.
        """
    },
    {
        "name": "Forbidden Prefix Test (Section)",
        "text": """
        The respondent was charged under Section 420 IPC and fined Rs. 500.
        """
    }
]

for case in test_cases:
    print(f"\n--- Testing: {case['name']} ---")
    result = extract_financials(case['text'])
    
    found = False
    for cat in ["fine", "compensation", "penalty", "costs"]:
        if result[cat]:
            found = True
            print(f"{cat.capitalize()}:")
            for entry in result[cat]:
                print(f"  - {entry['amount']} (Context: {entry['context']})")
    
    if not found:
        print("No financial implications extracted.")
    
    print("Raw Mentions:")
    for entry in result["raw_mentions"]:
        print(f"  - {entry['amount']} ({entry['category']})")
