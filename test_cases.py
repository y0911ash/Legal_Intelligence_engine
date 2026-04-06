"""
Shared test cases for evaluation scripts.
Centralizes judgment text and reference summaries used across
run_test.py, run_rouge.py, and run_comparison.py.
"""

SHORT_TEXT = """
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

SHORT_REF = (
    "The appeal was dismissed. The conviction under Section 302 IPC "
    "for murder was upheld. The appellant was sentenced to life imprisonment "
    "with a fine of Rs. 50,000 and ordered to pay compensation of Rs. 1,00,000 "
    "to the family of the deceased."
)

MEDIUM_TEXT = """
IN THE HIGH COURT OF DELHI
Criminal Appeal No. 456 of 2023

FACTS OF THE CASE:
The appellant Ramesh Kumar was convicted by the Sessions Court under
Section 420 IPC for cheating and dishonestly inducing delivery of property.
The complainant Suresh Sharma alleged that the appellant sold him a
residential flat in Dwarka for Rs. 45,00,000 by producing forged documents.
The complainant paid Rs. 20,00,000 as advance but later discovered that the
appellant had no title to the property. An FIR was registered and after
investigation the charge sheet was filed. The trial court examined 8
prosecution witnesses and 3 defence witnesses over 14 hearing dates.

ARGUMENTS:
Learned counsel for the appellant argued that the prosecution failed to
establish mens rea. The sale agreement was genuine and the title dispute
was a civil matter. The counsel cited State of Haryana v. Bhajan Lal
to argue that criminal proceedings should not be used for civil disputes.
Learned counsel for the respondent State contended that the forged
documents clearly established criminal intent. The appellant had previously
been involved in two similar property fraud cases.

JUDGMENT:
This Court has carefully considered the submissions of both sides and
perused the evidence on record. The prosecution has established beyond
reasonable doubt that the appellant used forged documents to induce the
complainant to part with his money. The testimony of PW-1, PW-3 and PW-5
is consistent and credible. The defence of civil dispute is rejected as
the evidence clearly shows criminal intent. The trial court's assessment
of evidence is sound and does not warrant interference.

ORDER:
The appeal is dismissed. The conviction under Section 420 IPC is upheld.
The sentence of 3 years rigorous imprisonment and fine of Rs. 1,00,000
is confirmed. The appellant shall surrender within 30 days. The complainant
is awarded compensation of Rs. 5,00,000 from the fine amount.
"""

MEDIUM_REF = (
    "The High Court dismissed the criminal appeal and upheld the conviction "
    "under Section 420 IPC for cheating. The appellant Ramesh Kumar had sold "
    "a flat using forged documents and cheated the complainant of Rs. 20,00,000. "
    "The sentence of 3 years rigorous imprisonment and fine of Rs. 1,00,000 was "
    "confirmed. Compensation of Rs. 5,00,000 was awarded to the complainant."
)

LONG_TEXT = """
IN THE HIGH COURT OF JUDICATURE AT BOMBAY
CRIMINAL APPELLATE JURISDICTION
Criminal Appeal No. 1178 of 2022

HON'BLE JUSTICE S.M. MODAK AND HON'BLE JUSTICE R.N. LADDHA

FACTS OF THE CASE:
The appellant Sunil Devidas Patil was tried and convicted by the Sessions
Court, Pune in Sessions Case No. 234 of 2020 for offences punishable under
Sections 302, 392 and 34 of the Indian Penal Code. The prosecution case
was that on the night of 23rd August 2019 the appellant along with two
co-accused entered the residence of the deceased Govind Shankar Kulkarni
aged 72 years. The deceased was a retired bank officer living alone in
his bungalow at Koregaon Park, Pune. The appellant and co-accused gained
entry by breaking the rear window. They were armed with an iron rod and
a knife. The deceased who was asleep was woken up and was assaulted with
the iron rod on his head causing severe head injuries. The co-accused
ransacked the house and took away gold ornaments weighing approximately
250 grams valued at Rs. 12,50,000, cash amounting to Rs. 3,75,000 and
a laptop computer. The deceased succumbed to his injuries in Sassoon
Hospital the next morning. The post-mortem report revealed that death
was caused by severe cranial trauma and subdural hemorrhage. An FIR
bearing No. 567 of 2019 was registered at Koregaon Park Police Station.

INVESTIGATION:
During the investigation, the police recovered the stolen gold ornaments
from the possession of co-accused Santosh Jadhav pursuant to his
disclosure statement. The iron rod used as a weapon was also recovered
from a drain near the crime scene. CCTV footage from a nearby shop showed
three persons moving towards the deceased's house at approximately 11:45 PM
and leaving at 12:30 AM. The fingerprint bureau matched the fingerprints
found on the broken window with those of the appellant. Cell tower location
data placed the appellant's mobile phone in the vicinity of the crime scene
during the relevant period. The investigating officer examined 14
prosecution witnesses including the forensic expert, the fingerprint expert,
the medical officer who conducted the post-mortem, and two neighbours who
heard sounds from the deceased's house. The charge sheet was filed on
15th November 2019.

ARGUMENTS:
Learned counsel for the appellant Shri A.K. Deshmukh submitted the
following contentions. First, the identification of the appellant by
CCTV footage was unreliable as the quality of the footage was poor and
the faces were not clearly visible. Second, the fingerprint evidence
was contaminated as the scene of crime was not properly secured before
the fingerprint expert arrived. Third, the cell tower evidence was
circumstantial and could not conclusively place the appellant at the
exact location. Fourth, no independent witness corroborated the
prosecution theory. Fifth, the disclosure statement of the co-accused
could not be used against the appellant. Learned counsel for the
respondent State Smt. P.R. Bhonsle argued that the chain of circumstantial
evidence was complete and unbroken. The CCTV footage, fingerprint evidence,
cell tower data, and recovery of stolen property together established the
guilt of the appellant beyond reasonable doubt. The medical evidence clearly
established the cause of death and the brutal nature of the assault.

JUDGMENT:
After careful consideration of the entire evidence on record and the
submissions of both learned counsel, this Court is of the considered view
that the prosecution has successfully established the guilt of the appellant
beyond reasonable doubt. The fingerprint evidence is reliable as
demonstrated by the expert testimony of PW-9. The CCTV footage, while not
of the highest quality, clearly shows three individuals matching the
description of the accused persons. The cell tower evidence corroborates
the presence of the appellant near the scene. The recovery of stolen
property from the co-accused further strengthens the prosecution case.
The defence has failed to provide any credible explanation for the
overwhelming circumstantial evidence against the appellant.

ORDER:
In view of the foregoing analysis, the appeal is dismissed. The conviction
of the appellant under Sections 302, 392 and 34 IPC is hereby upheld. The
sentence of life imprisonment for the offence under Section 302 IPC and
seven years rigorous imprisonment for the offence under Section 392 IPC
with a fine of Rs. 2,00,000 is confirmed. The sentences shall run
concurrently. The appellant is directed to pay compensation of Rs. 10,00,000
to the family of the deceased under Section 357A CrPC. In default of
payment of fine, the appellant shall undergo further imprisonment of one
year. The bail bonds of the appellant stand cancelled. The appellant shall
surrender before the trial court within four weeks from the date of this
order. A copy of this judgment shall be sent to the Sessions Court, Pune
for compliance. Costs of Rs. 25,000 awarded to the respondent.
"""

LONG_REF = (
    "The Bombay High Court dismissed the criminal appeal and upheld the "
    "conviction under Sections 302, 392 and 34 IPC. The appellant Sunil Patil "
    "was convicted for the murder and robbery of a retired bank officer. The court "
    "found the chain of circumstantial evidence including fingerprints, CCTV "
    "footage and cell tower data to be complete and reliable. The sentence of life "
    "imprisonment and seven years rigorous imprisonment with fine of Rs. 2,00,000 "
    "was confirmed. Compensation of Rs. 10,00,000 was ordered to be paid to the "
    "family of the deceased."
)
