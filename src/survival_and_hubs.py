import os
import glob
import pickle
import requests
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from statsmodels.stats.multitest import multipletests


def load_clean_clinical(clin_pattern="**/clinical*.tsv"):
    """Load GDC clinical metadata without coercing missing vital status or stage into valid categories."""
    clin_files = glob.glob(clin_pattern, recursive=True)
    if not clin_files:
        raise FileNotFoundError(f"No clinical TSV matched pattern: {clin_pattern}")

    raw_clin = pd.read_csv(clin_files[0], sep="\t", na_values=["'--", "not reported", "Not Reported", "NA", "nan", ""])
    id_col = "cases.submitter_id" if "cases.submitter_id" in raw_clin.columns else "case_submitter_id"
    raw_clin["_pid"] = raw_clin[id_col].astype(str).str[:12]
    raw_clin = raw_clin[raw_clin["_pid"].str.startswith("TCGA-")].drop_duplicates(subset=["_pid"]).copy()

    clin = pd.DataFrame()
    clin["patient_id"] = raw_clin["_pid"]

    death_col = "demographic.days_to_death" if "demographic.days_to_death" in raw_clin.columns else "days_to_death"
    fu_col = "diagnoses.days_to_last_follow_up" if "diagnoses.days_to_last_follow_up" in raw_clin.columns else "days_to_last_follow_up"
    days_death = pd.to_numeric(raw_clin[death_col], errors="coerce")
    days_fu = pd.to_numeric(raw_clin[fu_col], errors="coerce")
    clin["time"] = days_death.fillna(days_fu)

    # Preserve missing vital_status as NaN rather than assuming alive
    vital_col = "demographic.vital_status" if "demographic.vital_status" in raw_clin.columns else "vital_status"
    vital_clean = raw_clin[vital_col].astype(str).str.strip().str.lower()
    clin["event"] = np.where(
        vital_clean == "dead", 1.0,
        np.where(vital_clean == "alive", 0.0, np.nan)
    )

    age_col = "demographic.age_at_index" if "demographic.age_at_index" in raw_clin.columns else "age_at_index"
    clin["age"] = pd.to_numeric(raw_clin[age_col], errors="coerce")

    gender_col = "demographic.gender" if "demographic.gender" in raw_clin.columns else "gender"
    gender_clean = raw_clin[gender_col].astype(str).str.strip().str.lower()
    clin["is_male"] = np.where(
        gender_clean == "male", 1.0,
        np.where(gender_clean == "female", 0.0, np.nan)
    )

    # Preserve missing pathological stage as NaN rather than assuming early-stage
    stage_col = "diagnoses.ajcc_pathologic_stage" if "diagnoses.ajcc_pathologic_stage" in raw_clin.columns else "ajcc_pathologic_stage"
    stage_raw = raw_clin[stage_col]
    def encode_stage(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if any(tok in s for tok in ["STAGE III", "STAGE IV", "III", "IV"]):
            return 1.0
        if any(tok in s for tok in ["STAGE I", "STAGE II", "I", "II"]):
            return 0.0
        return np.nan

    clin["advanced_stage"] = stage_raw.apply(encode_stage)
    return clin


def query_string_hubs(gene_symbols, species=9606, score_threshold=700, out_csv="results/LUAD_ppi_hubs.csv"):
    """Query STRING v12 functional association network (combined score >= 0.70) and compute degree centrality."""
    genes = [str(g).strip() for g in gene_symbols if pd.notna(g) and str(g).strip()]
    if len(genes) < 2:
        return pd.DataFrame()

    url = "https://string-db.org/api/json/network"
    payload = {
        "identifiers": "\r".join(genes),
        "species": species,
        "required_score": score_threshold,
        "caller_identity": "tcga_luad_prioritization_v1_2",
    }
    resp = requests.post(url, data=payload, timeout=30)
    resp.raise_for_status()
    edges = resp.json()

    degree_map = {g: 0 for g in genes}
    seen_pairs = set()
    for e in edges:
        a = e.get("preferredName_A", "")
        b = e.get("preferredName_B", "")
        if a in degree_map and b in degree_map and a != b:
            pair = tuple(sorted([a, b]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                degree_map[a] += 1
                degree_map[b] += 1

    hub_df = pd.DataFrame([
        {"symbol": g, "network_degree": deg, "string_min_score": score_threshold / 1000.0}
        for g, deg in degree_map.items()
    ]).sort_values(["network_degree", "symbol"], ascending=[False, True])
    hub_df.to_csv(out_csv, index=False)
    return hub_df


def run_survival_screen(
    cache_pkl="results/stage1_cache.pkl",
    out_all_csv="results/LUAD_all_survival_screen.csv",
    out_cand_csv="results/LUAD_prognostic_candidates.csv",
):
    with open(cache_pkl, "rb") as f:
        cache = pickle.load(f)

    de_df = cache["annotated_de"].copy()
    expr_mat = cache["filtered_expression"].copy()
    tumor_meta = cache["tcga_tumor"].copy()

    if float(expr_mat.max().max()) > 50:
        expr_log = np.log2(expr_mat.div(expr_mat.sum(axis=0), axis=1) * 1e6 + 1.0)
    else:
        expr_log = expr_mat

    expr_log.index = expr_log.index.astype(str).str.split(".").str[0]
    expr_log = expr_log[~expr_log.index.duplicated(keep="first")]
    de_df["ensembl_id"] = de_df.index.astype(str).str.split(".").str[0]

    # Explicit Primary Tumor filter (-01) and deterministic vial selection (-01A before -01B)
    tumor_meta = tumor_meta[tumor_meta["tcga_id"].astype(str).str.contains("-01", na=False)].copy()
    tumor_meta["patient_id"] = tumor_meta["tcga_id"].astype(str).str[:12]
    tumor_meta = tumor_meta.sort_values(["patient_id", "tcga_id", "sample_id"]).drop_duplicates(subset=["patient_id"], keep="first")
    patient_to_sample = dict(zip(tumor_meta["patient_id"], tumor_meta["sample_id"]))

    clin = load_clean_clinical()
    clin["sample_id"] = clin["patient_id"].map(patient_to_sample)
    clin = clin.dropna(subset=["sample_id"]).copy()

    up_genes = de_df[(de_df["log2FoldChange"] > 1.5) & (de_df["padj"] < 0.01)].copy()
    records = []

    for _, row in up_genes.iterrows():
        ens = row["ensembl_id"]
        sym = row["symbol"]
        if pd.isna(sym) or ens not in expr_log.index:
            continue

        df_g = clin.copy()
        df_g["expr"] = df_g["sample_id"].map(expr_log.loc[ens])
        df_g = df_g.dropna(subset=["time", "event", "expr"]).copy()
        df_g = df_g[df_g["time"] > 0]
        if len(df_g) < 40:
            continue

        # 1. Exploratory median-split Kaplan-Meier log-rank test
        med = df_g["expr"].median()
        high = df_g[df_g["expr"] >= med]
        low = df_g[df_g["expr"] < med]
        if len(high) == 0 or len(low) == 0:
            continue

        lr = logrank_test(high["time"], low["time"], event_observed_A=high["event"], event_observed_B=low["event"])
        km_p = float(lr.p_value) if np.isfinite(lr.p_value) else np.nan

        # 2. Complete-case multivariable Cox PH model (adjusted for age, sex, pathological stage)
        cox_mv = df_g[["time", "event", "age", "is_male", "advanced_stage"]].dropna().copy()
        n_cox = len(cox_mv)
        cox_hr = np.nan
        cox_mv_p = np.nan
        cox_uv_p = np.nan
        ph_p = np.nan
        ph_status = "Not assessed"
        model_used = "multivariate_cox"

        if n_cox >= 40 and df_g.loc[cox_mv.index, "expr"].std() > 1e-8:
            cox_mv["expr_z"] = (
                df_g.loc[cox_mv.index, "expr"] - df_g.loc[cox_mv.index, "expr"].mean()
            ) / df_g.loc[cox_mv.index, "expr"].std()
            try:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cox_mv, duration_col="time", event_col="event")
                cox_hr = float(np.exp(cph.params_["expr_z"]))
                cox_mv_p = float(cph.summary.loc["expr_z", "p"])

                ph_res = proportional_hazard_test(cph, cox_mv, time_transform="rank")
                ph_p = float(ph_res.summary.loc["expr_z", "p"])
                ph_status = "Pass" if ph_p >= 0.05 else "Fail"
            except Exception:
                model_used = "univariate_fallback"

        if np.isnan(cox_hr) and df_g["expr"].std() > 1e-8:
            model_used = "univariate_fallback"
            cox_uv = df_g[["time", "event"]].copy()
            cox_uv["expr_z"] = (df_g["expr"] - df_g["expr"].mean()) / df_g["expr"].std()
            cph_u = CoxPHFitter(penalizer=0.1)
            cph_u.fit(cox_uv, duration_col="time", event_col="event")
            cox_hr = float(np.exp(cph_u.params_["expr_z"]))
            cox_uv_p = float(cph_u.summary.loc["expr_z", "p"])

        records.append({
            "ensembl_id": ens,
            "symbol": sym,
            "n_km": int(len(df_g)),
            "n_cox_complete_case": int(n_cox),
            "log2FoldChange": round(float(row["log2FoldChange"]), 4),
            "deseq_padj": float(row["padj"]),
            "km_logrank_p": km_p,
            "cox_HR": round(cox_hr, 4) if np.isfinite(cox_hr) else np.nan,
            "cox_multivar_p": cox_mv_p,
            "cox_univar_p": cox_uv_p,
            "schoenfeld_ph_p": ph_p,
            "ph_assumption_status": ph_status,
            "cox_model": model_used,
        })

    res_df = pd.DataFrame(records)

    # CRITICAL FIX (#1): Filter non-NaN p-values BEFORE Benjamini-Hochberg correction
    res_df["km_logrank_padj"] = np.nan
    valid_km = res_df["km_logrank_p"].notna()
    if valid_km.any():
        res_df.loc[valid_km, "km_logrank_padj"] = multipletests(
            res_df.loc[valid_km, "km_logrank_p"], method="fdr_bh"
        )[1]

    res_df["cox_multivar_padj"] = np.nan
    valid_mv = res_df["cox_multivar_p"].notna()
    if valid_mv.any():
        res_df.loc[valid_mv, "cox_multivar_padj"] = multipletests(
            res_df.loc[valid_mv, "cox_multivar_p"], method="fdr_bh"
        )[1]

    res_df = res_df.sort_values(["km_logrank_padj", "km_logrank_p"], na_position="last")
    res_df.to_csv(out_all_csv, index=False)

    # Exploratory KM FDR candidates (q < 0.10, HR > 1.0)
    cand_df = res_df[(res_df["km_logrank_padj"] < 0.10) & (res_df["cox_HR"] > 1.0)].copy()
    cand_df.to_csv(out_cand_csv, index=False)

    nom_count = int((res_df["km_logrank_p"] < 0.05).sum())
    mv_sig_count = int((res_df["cox_multivar_p"] < 0.05).sum())
    print(f"Tested {len(res_df)} upregulated protein-coding genes (valid KM p-values: {int(valid_km.sum())})")
    print(f"Nominal KM p < 0.05: {nom_count} | Exploratory KM FDR q < 0.10 (HR > 1.0): {len(cand_df)}")
    print(f"Multivariable Cox p < 0.05 (complete-case adjusted): {mv_sig_count}")
    return res_df, cand_df


# Backward-compatible alias
run_corrected_survival_screen = run_survival_screen

if __name__ == "__main__":
    run_survival_screen()