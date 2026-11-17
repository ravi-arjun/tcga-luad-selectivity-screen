import os
import pandas as pd

clinical_dir = os.path.join("data", "clinical.cart.2026-01-17")
print("Files in clinical directory:", os.listdir(clinical_dir))

clinical_path = os.path.join(clinical_dir, "clinical.tsv")
clin = pd.read_csv(clinical_path, sep="\t")
print("Clinical table shape:", clin.shape)

# Find all survival, demographic, and ID columns
keywords = ["submitter", "case", "vital", "days_to", "stage", "age", "gender", "tobacco", "smok"]
matched_cols = [c for c in clin.columns if any(k in c.lower() for k in keywords)]
print("\nMatched Clinical Columns:")
for col in matched_cols:
    print(f"  - {col}: {clin[col].dropna().unique()[:4]}")

# Check format of your expression matrix sample columns
de_path = os.path.join("results", "LUAD_DESeq2_annotated.csv")
if os.path.exists(de_path):
    de_df = pd.read_csv(de_path, index_col=0)
    print("\nAnnotated DE table loaded successfully! Shape:", de_df.shape)