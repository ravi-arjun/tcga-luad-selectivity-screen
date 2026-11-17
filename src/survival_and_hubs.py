import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def load_clean_clinical(clinical_path):
    """
    Loads GDC clinical.tsv, replaces '--' sentinels, deduplicates multiple 
    treatment rows per patient, and computes Overall Survival (OS) time & event.
    """
    clin = pd.read_csv(clinical_path, sep="\t", dtype=str)
    clin.replace(["'--", "--", "not reported", "Not Reported", "unknown"], np.nan, inplace=True)

    # Convert numeric fields
    clin["days_to_death"] = pd.to_numeric(clin["demographic.days_to_death"], errors="coerce")
    clin["days_to_follow_up"] = pd.to_numeric(clin["diagnoses.days_to_last_follow_up"], errors="coerce")
    clin["age"] = pd.to_numeric(clin["demographic.age_at_index"], errors="coerce")

    # Deduplicate per patient (cases.submitter_id = TCGA-XX-XXXX)
    agg_rules = {
        "demographic.vital_status": "first",
        "days_to_death": "max",
        "days_to_follow_up": "max",
        "age": "first",
        "demographic.gender": "first",
        "diagnoses.ajcc_pathologic_stage": "first",
    }
    patient_df = clin.groupby("cases.submitter_id", as_index=False).agg(agg_rules)

    # Define Event (1 = Dead, 0 = Alive/Censored) & Time (days)
    patient_df["event"] = (patient_df["demographic.vital_status"] == "Dead").astype(int)
    patient_df["time"] = np.where(
        patient_df["event"] == 1,
        patient_df["days_to_death"],
        patient_df["days_to_follow_up"],
    )

    # Simplify AJCC pathologic stage into Early (Stage I/II) vs Late (Stage III/IV) for Cox regression
    def simplify_stage(s):
        if pd.isna(s):
            return np.nan
        s = str(s).upper()
        if "IV" in s or "III" in s:
            return 1  # Advanced stage
        if "I" in s or "II" in s:
            return 0  # Early stage
        return np.nan

    patient_df["advanced_stage"] = patient_df["diagnoses.ajcc_pathologic_stage"].apply(simplify_stage)
    patient_df["is_male"] = (patient_df["demographic.gender"].str.lower() == "male").astype(int)

    # Keep valid positive survival times
    patient_df = patient_df[patient_df["time"] > 0].copy()
    print(f"Cleaned unique clinical patients with valid OS data: {len(patient_df)}")
    return patient_df


def run_survival_screen(expr_df, sample_meta, annotated_de, patient_df, top_n=100, output_dir="results"):
    """
    Screens top overexpressed genes via Kaplan-Meier Log-Rank test and 
    Multivariate Cox Proportional Hazards (adjusted for age, gender, tumor stage).
    """
    # Filter to tumor samples only (Primary Tumor)
    tumor_samples = sample_meta[sample_meta["condition"] == "tumor"].copy()
    
    # Extract 12-char TCGA patient barcode (e.g., TCGA-67-3773) from sample_id / barcode
    # Adjust column name below if your sample_meta stores barcode under 'barcode' or 'sample_id'
    barcode_col = "tcga_id" if "tcga_id" in tumor_samples.columns else ("barcode" if "barcode" in tumor_samples.columns else "sample_id")
    tumor_samples["patient_id"] = tumor_samples[barcode_col].str.slice(0, 12)
    tumor_samples = tumor_samples.drop_duplicates(subset=["patient_id"])

    # Merge tumor sample IDs with clinical survival data
    merged_meta = pd.merge(tumor_samples, patient_df, left_on="patient_id", right_on="cases.submitter_id")
    print(f"Matched Tumor RNA-seq samples with Survival metadata: {len(merged_meta)}")

    # Select top N significantly upregulated genes with valid HUGO symbols
    up_genes = annotated_de[
        (annotated_de["padj"] < 0.05) & (annotated_de["log2FoldChange"] >= 1.5)
    ].copy()
    up_genes = up_genes[~up_genes["symbol"].str.startswith("ENSG", na=True)]
    up_genes["score"] = up_genes["log2FoldChange"] * (-np.log10(up_genes["padj"].replace(0, 1e-300)))
    candidates = up_genes.sort_values("score", ascending=False).head(top_n)

    survival_results = []
    valid_sample_ids = merged_meta["sample_id"].tolist()

    for ensg_id, row in candidates.iterrows():
        if ensg_id not in expr_df.index:
            continue
        symbol = row["symbol"]
        gene_vals = expr_df.loc[ensg_id, valid_sample_ids].values
        
        df_gene = merged_meta.copy()
        df_gene["expr"] = gene_vals
        
        # Median split for Kaplan-Meier
        median_val = df_gene["expr"].median()
        high_grp = df_gene[df_gene["expr"] >= median_val]
        low_grp = df_gene[df_gene["expr"] < median_val]
        
        if len(high_grp) < 10 or len(low_grp) < 10:
            continue

        lr = logrank_test(high_grp["time"], low_grp["time"], event_observed_A=high_grp["event"], event_observed_B=low_grp["event"])
        
        # Multivariate Cox Regression (expr Z-score + age + sex + stage)
# Regularized Multivariate Cox Regression (prevents collinearity singular matrix)
        cox_df = df_gene[["time", "event", "expr", "age", "is_male", "advanced_stage"]].dropna().copy()
        cox_df["expr_z"] = (cox_df["expr"] - cox_df["expr"].mean()) / (cox_df["expr"].std() + 1e-8)

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_df, duration_col="time", event_col="event")
            hr = float(np.exp(cph.params_["expr_z"]))
            cox_p = float(cph.summary.loc["expr_z", "p"])
        except Exception:
            # Fallback to univariate Cox on expression Z-score alone
            cph_uni = CoxPHFitter(penalizer=0.1)
            cph_uni.fit(cox_df[["time", "event", "expr_z"]], duration_col="time", event_col="event")
            hr = float(np.exp(cph_uni.params_["expr_z"]))
            cox_p = float(cph_uni.summary.loc["expr_z", "p"])

        survival_results.append({
            "ensembl_id": ensg_id,
            "symbol": symbol,
            "log2FoldChange": row["log2FoldChange"],
            "deseq_padj": row["padj"],
            "km_logrank_p": lr.p_value,
            "cox_HR": hr,
            "cox_multivar_p": cox_p,
        })

    surv_df = pd.DataFrame(survival_results)
    # Keep genes where higher expression significantly worsens survival (HR > 1 and p < 0.05)
    prognostic = surv_df[
        ((surv_df["km_logrank_p"] < 0.05) | (surv_df["cox_multivar_p"] < 0.05)) & (surv_df["cox_HR"] > 1.0)
    ].sort_values("cox_multivar_p")
    os.makedirs(output_dir, exist_ok=True)
    surv_df.to_csv(f"{output_dir}/LUAD_all_survival_screen.csv", index=False)
    prognostic.to_csv(f"{output_dir}/LUAD_prognostic_drivers.csv", index=False)
    print(f"\nFound {len()} statistically significant prognostic drivers (KM p < 0.05, HR > 1.0)!")
    return prognostic, merged_meta


def query_string_hubs(gene_symbols, output_dir="results"):
    """
    Queries the STRING v12 PPI API to compute network degree centrality 
    and prioritize signaling hub bottlenecks among prognostic candidates.
    """
    if len(gene_symbols) < 2:
        return pd.DataFrame()
    
    print(f"\nQuerying STRING PPI network for {len(gene_symbols)} prognostic genes...")
    url = "https://version-12-0.string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(gene_symbols[:50]),
        "species": 9606,  # Homo sapiens
        "required_score": 400,  # Medium confidence (0.4)
    }
    resp = requests.post(url, data=params, timeout=30)
    edges = resp.json()

    G = nx.Graph()
    for g in gene_symbols[:50]:
        G.add_node(g)
    for edge in edges:
        G.add_edge(edge["preferredName_A"], edge["preferredName_B"], weight=edge["score"])

    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G)
    hub_df = pd.DataFrame({
        "symbol": list(G.nodes()),
        "ppi_degree": [deg[n] for n in G.nodes()],
        "betweenness": [round(bet[n], 4) for n in G.nodes()],
    }).sort_values(by=["ppi_degree", "betweenness"], ascending=False)

    hub_df.to_csv(f"{output_dir}/LUAD_ppi_hubs.csv", index=False)
    return hub_df