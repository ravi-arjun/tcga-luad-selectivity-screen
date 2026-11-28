from glob import glob
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mygene
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from survival_and_hubs import load_clean_clinical, run_survival_screen, query_string_hubs


def sample_cleaner(name, path):
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[["gene_id", "unstranded"]]
    df = df[df["gene_id"].str.startswith("ENSG")].copy()
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    df = df.drop_duplicates(subset=["gene_id"]).set_index("gene_id")
    df.columns = [name]
    return df


def build_expression_matrix(raw_path):
    files = sorted(glob(f"{raw_path}/**/*.rna_seq.augmented_star_gene_counts.tsv", recursive=True))
    if not files:
        raise FileNotFoundError(f"No STAR count TSVs found under '{raw_path}'.")
    dfs = [sample_cleaner(f"sample_{i}", f) for i, f in enumerate(files, start=1)]
    expression_matrix = pd.concat(dfs, axis=1)
    return expression_matrix, files


def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path, sep="\t", comment="#")
    metadata = metadata[["samples.submitter_id", "samples.sample_type"]].copy()

    # Explicitly classify Primary Tumor vs. Solid Tissue Normal; exclude other sample types
    # NOTE: np.where(..., "str", np.nan) coerces NaN to the *string* "nan", which dropna() does not remove,
    # so a dict map (unmapped -> real NaN) is used instead.
    sample_type_clean = metadata["samples.sample_type"].astype(str).str.strip()
    metadata["condition"] = sample_type_clean.map(
        {"Primary Tumor": "tumor", "Solid Tissue Normal": "normal"}
    )
    metadata = metadata.dropna(subset=["condition"]).copy()
    metadata = metadata[["samples.submitter_id", "condition"]].rename(
        columns={"samples.submitter_id": "tcga_id"}
    )
    return metadata


def build_file_lookup(metadata_json):
    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    lookup = []
    for record in metadata:
        file_id = record["file_id"]
        tcga_id = record["associated_entities"][0]["entity_submitter_id"]
        tcga_id = "-".join(tcga_id.split("-")[:4])
        lookup.append({"file_id": file_id, "tcga_id": tcga_id})
    return pd.DataFrame(lookup)


def build_sample_map(expression_matrix, files):
    sample_map = pd.DataFrame({
        "sample_id": expression_matrix.columns,
        "file_path": files,
    })
    sample_map["file_id"] = sample_map["file_path"].apply(
        lambda x: os.path.normpath(x).split(os.sep)[-2]
    )
    return sample_map


def validate_sample_mapping(sample_map, tcga_tumor):
    """
    Checks whether multiple expression files map to the same
    biological TCGA sample before differential expression analysis.
    """

    # Check whether a GDC file has been mapped more than once
    duplicate_files = sample_map[
        sample_map.duplicated(subset=["file_id"], keep=False)
    ]

    if not duplicate_files.empty:
        duplicate_files.to_csv(
            "results/duplicate_gdc_file_ids.csv", index=False
        )
        raise ValueError(
            "Duplicate GDC file IDs detected. "
            "Check results/duplicate_gdc_file_ids.csv"
        )

    # Check whether multiple files represent the same TCGA sample
    duplicate_samples = tcga_tumor[
        tcga_tumor.duplicated(subset=["tcga_id"], keep=False)
    ].copy()

    if not duplicate_samples.empty:
        duplicate_samples = duplicate_samples.sort_values("tcga_id")

        duplicate_samples.to_csv(
            "results/duplicate_tcga_samples.csv", index=False
        )

        print("\nDuplicate biological samples detected:")
        print(
            duplicate_samples[
                ["tcga_id", "sample_id", "file_id", "condition"]
            ].to_string(index=False)
        )

        raise ValueError(
            f"{duplicate_samples['tcga_id'].nunique()} TCGA samples "
            "have multiple expression files. Resolve these before DESeq2."
        )

    print("\nSample mapping validation passed.")
    print(f"Unique biological samples: {tcga_tumor['tcga_id'].nunique()}")



def run_pipeline(cancer_type, raw_path, metadata_path, metadata_json):
    print(f"Running {cancer_type} Stage 1 ingestion...")
    expression_matrix, files = build_expression_matrix(raw_path)
    expression_matrix.to_csv(f"{cancer_type}_raw_counts.csv")

    metadata = load_metadata(metadata_path)
    file_lookup = build_file_lookup(metadata_json)
    sample_map = build_sample_map(expression_matrix, files)
    sample_map = sample_map.merge(file_lookup, on="file_id", how="left")
    
    tcga_tumor = sample_map.merge(metadata,on="tcga_id", how="inner",validate="many_to_one")

    # Validate biological sample independence before DESeq2
    validate_sample_mapping(sample_map, tcga_tumor)

    # Retain only verified samples
    expression_matrix = expression_matrix[
    tcga_tumor["sample_id"]
    ].copy()

    return expression_matrix, metadata, sample_map, tcga_tumor


def normalize_counts(expression_matrix):
    library_sizes = expression_matrix.sum(axis=0)
    cpm = expression_matrix.div(library_sizes, axis=1) * 1_000_000
    return np.log2(cpm + 1)


def filter_low_expression(raw_counts, min_cpm=1.0, min_samples=10):
    """Filter low-abundance genes by CPM threshold while returning original integer raw counts."""
    log_cpm = normalize_counts(raw_counts)
    expressed = (log_cpm > np.log2(min_cpm + 1.0)).sum(axis=1)
    keep_genes = expressed[expressed >= min_samples].index
    return raw_counts.loc[keep_genes].copy(), log_cpm.loc[keep_genes].copy()


def differential_expression(filtered_raw_counts, tcga_tumor):
    """Performs negative binomial differential expression analysis via PyDESeq2 on integer raw counts."""
    valid_samples = [s for s in tcga_tumor["sample_id"] if s in filtered_raw_counts.columns]
    counts_df = filtered_raw_counts[valid_samples].T.round().astype(int)
    metadata_df = tcga_tumor.set_index("sample_id").loc[valid_samples]

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata_df,
        design_factors="condition",
        ref_level="normal",
    )
    print("\nFitting size factors and dispersions (PyDESeq2)...")
    dds.deseq2()

    print("Running Wald hypothesis test...")
    stat_res = DeseqStats(dds, contrast=["condition", "tumor", "normal"])
    stat_res.summary()

    results = stat_res.results_df
    results["significant"] = (results["padj"] < 0.01) & (results["log2FoldChange"] > 1.5)
    return results


def differential_expression_paired(filtered_raw_counts, tcga_tumor):
    """
    Sensitivity analysis: restricts to patients with BOTH a primary-tumor and a solid-tissue-normal sample and
    adds a patient term (design ~ patient + condition). The main analysis (differential_expression) treats all
    samples as independent even though ~50 patients contribute a tumor/normal pair.
    """
    t = tcga_tumor.copy()
    t["patient_id"] = t["tcga_id"].astype(str).str[:12]
    both = t.groupby("patient_id")["condition"].nunique()
    paired_patients = both[both == 2].index
    t = t[t["patient_id"].isin(paired_patients)]
    t = t.drop_duplicates(subset=["patient_id", "condition"])
    valid = [s for s in t["sample_id"] if s in filtered_raw_counts.columns]
    counts_df = filtered_raw_counts[valid].T.round().astype(int)
    meta = t.set_index("sample_id").loc[valid][["patient_id", "condition"]]
    dds = DeseqDataSet(counts=counts_df, metadata=meta, design_factors=["patient_id", "condition"], ref_level=["condition", "normal"])
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", "tumor", "normal"])
    stat_res.summary()
    return stat_res.results_df


def annotate_genes(de_results):
    print("\nMapping Ensembl IDs to Gene Symbols via MyGene...")
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(
        de_results.index.tolist(),
        scopes="ensembl.gene",
        fields=["symbol", "name", "type_of_gene"],
        species="human",
        as_dataframe=True,
    )
    gene_info = gene_info[~gene_info.index.duplicated(keep="first")]

    df = de_results.copy()
    df["symbol"] = gene_info["symbol"].reindex(df.index)
    df["gene_type"] = gene_info["type_of_gene"].reindex(df.index) if "type_of_gene" in gene_info.columns else np.nan
    # Unannotated genes keep their Ensembl ID as a placeholder label ONLY; downstream steps must not treat
    # these placeholders as gene symbols (see is_annotated_symbol in survival_and_hubs.py).
    df["symbol"] = df["symbol"].fillna(pd.Series(df.index, index=df.index))
    return df


def add_gene_type(annotated_de):
    """Adds a 'gene_type' (mygene type_of_gene, e.g. 'protein-coding') column to an annotated DE table that lacks it."""
    mg = mygene.MyGeneInfo()
    info = mg.querymany(
        annotated_de.index.astype(str).tolist(), scopes="ensembl.gene",
        fields=["type_of_gene"], species="human", as_dataframe=True,
    )
    info = info[~info.index.duplicated(keep="first")]
    out = annotated_de.copy()
    out["gene_type"] = info["type_of_gene"].reindex(out.index) if "type_of_gene" in info.columns else np.nan
    return out


def get_top_overexpressed_genes(annotated_de, n_top=25):
    upregulated = annotated_de[
        (annotated_de["padj"] < 0.01) & (annotated_de["log2FoldChange"] > 1.5)
    ].copy()
    upregulated["rank_score"] = upregulated["log2FoldChange"] * (-np.log10(upregulated["padj"].replace(0, 1e-300)))
    return upregulated.sort_values(by="rank_score", ascending=False).head(n_top)


def create_heatmap(top_candidates, filtered_expression, tcga_tumor, output_fig="results/top_genes_heatmap.png"):
    sample_annot = tcga_tumor.set_index("sample_id").loc[filtered_expression.columns]
    sorted_samples = sample_annot.sort_values(by="condition").index

    target_ids = [i for i in top_candidates.index if i in filtered_expression.index]
    expr_subset = filtered_expression.loc[target_ids, sorted_samples].copy()
    id_to_symbol = top_candidates["symbol"].to_dict()
    expr_subset.index = [id_to_symbol.get(i, i) for i in expr_subset.index]

    expr_zscore = expr_subset.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)
    condition_colors = sample_annot.loc[sorted_samples, "condition"].map(
        {"tumor": "#e74c3c", "normal": "#3498db"}
    )

    g = sns.clustermap(
        expr_zscore,
        col_cluster=False,
        row_cluster=True,
        col_colors=condition_colors,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        figsize=(12, 8),
        cbar_kws={"label": "Expression Z-Score"},
    )
    g.fig.suptitle("Expression Profile of Top Prioritized Genes in LUAD", fontsize=14, weight="bold", y=1.02)
    g.ax_heatmap.set_xlabel("Samples (Normal: Blue | Tumor: Red)", fontsize=11)
    g.ax_heatmap.set_ylabel("Candidate Genes", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    cache_file = "results/stage1_cache.pkl"

    if os.path.exists(cache_file):
        print("\n[CACHE HIT] Loading pre-computed expression & PyDESeq2 results...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        filtered_expression = cache["filtered_expression"]
        tcga_tumor = cache["tcga_tumor"]
        annotated_de = cache["annotated_de"]
        top_candidates = cache["top_candidates"]
    else:
        print("\n[CACHE MISS] Running full Stage 1 ingestion and PyDESeq2 screen...")
        raw_counts, _, _, tcga_tumor = run_pipeline(
            "luad",
            os.path.join("data", "raw"),
            os.path.join("data", "biospecimen.cart.2026-01-17", "sample.tsv"),
            "metadata.cart.2026-01-17.json",
        )
        filtered_raw, filtered_expression = filter_low_expression(raw_counts)
        de_results = differential_expression(filtered_raw, tcga_tumor)
        annotated_de = annotate_genes(de_results)
        annotated_de.to_csv("results/LUAD_DESeq2_annotated.csv")
        try:
            paired = differential_expression_paired(filtered_raw, tcga_tumor)
            paired.to_csv("results/LUAD_DESeq2_paired_sensitivity.csv")
        except Exception as e:  # sensitivity analysis only; never blocks the main pipeline
            print(f"[WARN] Paired sensitivity analysis skipped: {e}")
        top_candidates = get_top_overexpressed_genes(annotated_de, n_top=25)
        with open(cache_file, "wb") as f:
            pickle.dump({
                "filtered_expression": filtered_expression,
                "tcga_tumor": tcga_tumor,
                "annotated_de": annotated_de,
                "top_candidates": top_candidates,
            }, f)
        print(f"Saved Stage 1 cache to '{cache_file}'")

    if "gene_type" not in annotated_de.columns or annotated_de["gene_type"].isna().all():
        print("[INFO] Cached DE table has no gene_type; querying MyGene for biotype (needs network)...")
        try:
            annotated_de = add_gene_type(annotated_de)
            annotated_de.to_csv("results/LUAD_DESeq2_annotated.csv")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    _c = pickle.load(f)
                _c["annotated_de"] = annotated_de
                with open(cache_file, "wb") as f:
                    pickle.dump(_c, f)
        except Exception as e:
            print(f"[WARN] Biotype lookup failed ({e}); survival screen will fall back to annotated-symbol filtering "
                  "and the dashboard will NOT claim 'protein-coding'.")

    create_heatmap(top_candidates, filtered_expression, tcga_tumor)

    # Run Stage 2 Survival Screen & STRING v12 Network Prioritization
    res_df, cand_df = run_survival_screen(cache_pkl=cache_file)
    if not cand_df.empty:
        hub_df = query_string_hubs(cand_df["symbol"].tolist())
        final_targets = pd.merge(cand_df, hub_df, on="symbol").sort_values(
            by=["network_degree", "km_logrank_padj"], ascending=[False, True]
        )
        final_targets.to_csv("results/LUAD_final_docking_targets.csv", index=False)
        print("\n--- PRIORITIZED CANDIDATE TARGETS (DEG + EXPLORATORY KM FDR + STRING DEGREE) ---")
        print(final_targets.head(10).to_string(index=False))