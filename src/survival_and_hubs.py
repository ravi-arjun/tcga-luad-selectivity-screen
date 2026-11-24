import os
import glob
import pickle
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from statsmodels.stats.multitest import multipletests


def run_corrected_survival_screen(
    cache_pkl="results/stage1_cache.pkl",
    out_all_csv="results/LUAD_all_survival_screen.csv",
    out_cand_csv="results/LUAD_prognostic_candidates.csv",
):
    with open(cache_pkl, "rb") as f:
        cache = pickle.load(f)

    de_df = cache["annotated_de"].copy()
    expr_mat = cache["filtered_expression"].copy()
    tumor_meta = cache["tcga_tumor"].copy()

    # Convert to log2(CPM + 1) if raw counts are stored
    if float(expr_mat.max().max()) > 50:
        expr_log = np.log2(expr_mat.div(expr_mat.sum(axis=0), axis=1) * 1e6 + 1.0)
    else:
        expr_log = expr_mat

    expr_log.index = expr_log.index.astype(str).str.split(".").str[0]
    expr_log = expr_log[~expr_log.index.duplicated(keep="first")]
    de_df["ensembl_id"] = de_df.index.astype(str).str.split(".").str[0]

    # Map TCGA-XX-XXXX patient_id directly from tcga_tumor['tcga_id'] -> 'sample_id'
    tumor_meta["patient_id"] = tumor_meta["tcga_id"].astype(str).str[:12]
    tumor_meta = tumor_meta.drop_duplicates(subset=["patient_id"])
    patient_to_sample = dict(zip(tumor_meta["patient_id"], tumor_meta["sample_id"]))

    # Load dotted GDC clinical metadata
    clin_files = glob.glob("**/clinical*.tsv", recursive=True)
    raw_clin = pd.read_csv(clin_files[0], sep="\t", na_values=["'--", "not reported", "NA", ""])
    raw_clin["_pid"] = raw_clin["cases.submitter_id"].astype(str).str[:12]
    raw_clin = raw_clin[raw_clin["_pid"].str.startswith("TCGA-")].drop_duplicates(subset=["_pid"])

    clin = pd.DataFrame()
    clin["patient_id"] = raw_clin["_pid"]
    days_death = pd.to_numeric(raw_clin["demographic.days_to_death"], errors="coerce")
    days_fu = pd.to_numeric(raw_clin["diagnoses.days_to_last_follow_up"], errors="coerce")
    clin["time"] = days_death.fillna(days_fu)
    clin["event"] = (raw_clin["demographic.vital_status"].astype(str).str.lower() == "dead").astype(int)
    clin["age"] = pd.to_numeric(raw_clin["demographic.age_at_index"], errors="coerce")
    clin["is_male"] = (raw_clin["demographic.gender"].astype(str).str.lower() == "male").astype(int)
    stage_str = raw_clin["diagnoses.ajcc_pathologic_stage"].astype(str)
    clin["advanced_stage"] = stage_str.str.contains("III|IV", case=False, na=False).astype(int)

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

        # 1. Prespecified primary test: Median-split Kaplan-Meier log-rank test
        med = df_g["expr"].median()
        high = df_g[df_g["expr"] >= med]
        low = df_g[df_g["expr"] < med]
        lr = logrank_test(high["time"], low["time"], event_observed_A=high["event"], event_observed_B=low["event"])
        km_p = float(lr.p_value)

        # 2. Secondary Cox PH model with single standardized expression predictor (expr_z)
        cox_mv = df_g[["time", "event", "age", "is_male", "advanced_stage"]].dropna().copy()
        cox_mv["expr_z"] = (
            df_g.loc[cox_mv.index, "expr"] - df_g.loc[cox_mv.index, "expr"].mean()
        ) / (df_g.loc[cox_mv.index, "expr"].std() + 1e-8)

        cox_hr = np.nan
        cox_mv_p = np.nan
        cox_uv_p = np.nan
        ph_p = np.nan
        ph_status = "Not assessed"
        model_used = "multivariate_cox"

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
            cox_uv = df_g[["time", "event"]].copy()
            cox_uv["expr_z"] = (df_g["expr"] - df_g["expr"].mean()) / (df_g["expr"].std() + 1e-8)
            cph_u = CoxPHFitter(penalizer=0.1)
            cph_u.fit(cox_uv, duration_col="time", event_col="event")
            cox_hr = float(np.exp(cph_u.params_["expr_z"]))
            cox_uv_p = float(cph_u.summary.loc["expr_z", "p"])

        records.append({
            "ensembl_id": ens,
            "symbol": sym,
            "log2FoldChange": round(float(row["log2FoldChange"]), 4),
            "deseq_padj": float(row["padj"]),
            "km_logrank_p": km_p,
            "cox_HR": round(cox_hr, 4),
            "cox_multivar_p": cox_mv_p,
            "cox_univar_p": cox_uv_p,
            "schoenfeld_ph_p": ph_p,
            "ph_assumption_status": ph_status,
            "cox_model": model_used,
        })

    res_df = pd.DataFrame(records)
    res_df["km_logrank_padj"] = multipletests(res_df["km_logrank_p"], method="fdr_bh")[1]
    valid_mv = res_df["cox_multivar_p"].notna()
    res_df["cox_multivar_padj"] = np.nan
    if valid_mv.any():
        res_df.loc[valid_mv, "cox_multivar_padj"] = multipletests(res_df.loc[valid_mv, "cox_multivar_p"], method="fdr_bh")[1]

    res_df = res_df.sort_values(["km_logrank_padj", "km_logrank_p"])
    res_df.to_csv(out_all_csv, index=False)

    cand_df = res_df[(res_df["km_logrank_padj"] < 0.10) & (res_df["cox_HR"] > 1.0)].copy()
    cand_df.to_csv(out_cand_csv, index=False)
    print(f"Saved {len(res_df)} tested genes -> '{out_all_csv}'")
    print(f"Saved {len(cand_df)} KM FDR-adjusted candidates (q < 0.10, HR > 1.0) -> '{out_cand_csv}'")
    return res_df, cand_df


if __name__ == "__main__":
    run_corrected_survival_screen()