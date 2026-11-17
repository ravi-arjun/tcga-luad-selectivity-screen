import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from survival_and_hubs import load_clean_clinical


def generate_km_panel(
    genes=("NEK2", "PLK1", "KIF14", "DLGAP5"),
    cache_path="results/stage1_cache.pkl",
    clinical_path=os.path.join("data", "clinical.cart.2026-01-17", "clinical.tsv"),
    output_png="plots/LUAD_top_targets_KM_panel.png",
    output_json="results/LUAD_km_curves_web.json",
):
    # 1. Load cached expression & annotated DE table in 0.1s
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    filtered_expression = cache["filtered_expression"]
    tcga_tumor = cache["tcga_tumor"]
    annotated_de = cache["annotated_de"]

    # 2. Load deduplicated clinical survival data and align with tumor samples
    patient_df = load_clean_clinical(clinical_path)
    tumor_samples = tcga_tumor[tcga_tumor["condition"] == "tumor"].copy()
    tumor_samples["patient_id"] = tumor_samples["tcga_id"].str.slice(0, 12)
    tumor_samples = tumor_samples.drop_duplicates(subset=["patient_id"])
    merged_meta = pd.merge(
        tumor_samples, patient_df, left_on="patient_id", right_on="cases.submitter_id"
    )

    # Map gene symbol -> Ensembl ID
    symbol_to_ensg = (
        annotated_de.reset_index()
        .dropna(subset=["symbol"])
        .drop_duplicates(subset=["symbol"])
        .set_index("symbol")["gene_id"]
        .to_dict()
    )

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    web_export = {}

    for idx, symbol in enumerate(genes):
        ax = axes[idx]
        ensg_id = symbol_to_ensg.get(symbol)
        if not ensg_id or ensg_id not in filtered_expression.index:
            print(f"Warning: {symbol} not found in expression matrix.")
            continue

        df_gene = merged_meta.copy()
        df_gene["expr"] = filtered_expression.loc[ensg_id, df_gene["sample_id"]].values
        # Convert days to months for cleaner clinical x-axis
        df_gene["months"] = df_gene["time"] / 30.44

        median_expr = df_gene["expr"].median()
        high = df_gene[df_gene["expr"] >= median_expr]
        low = df_gene[df_gene["expr"] < median_expr]

        lr = logrank_test(
            high["months"],
            low["months"],
            event_observed_A=high["event"],
            event_observed_B=low["event"],
        )

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(high["months"], event_observed=high["event"], label=f"High {symbol} (n={len(high)})")
        kmf_low.fit(low["months"], event_observed=low["event"], label=f"Low {symbol} (n={len(low)})")

        kmf_high.plot_survival_function(ax=ax, color="#e74c3c", linewidth=2.2, ci_alpha=0.15)
        kmf_low.plot_survival_function(ax=ax, color="#2980b9", linewidth=2.2, ci_alpha=0.15)

        ax.set_title(f"{symbol} Overall Survival (TCGA-LUAD)", fontsize=12, weight="bold")
        ax.set_xlabel("Overall Survival (Months)", fontsize=10)
        ax.set_ylabel("Survival Probability", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower left", frameon=True, fontsize=9)

        # Annotate Log-Rank p-value & median survival
        p_text = f"Log-Rank p = {lr.p_value:.4f}"
        ax.text(
            0.96,
            0.92,
            p_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
        )

        # Save JSON step coordinates for interactive Plotly.js dashboard
        web_export[symbol] = {
            "ensembl_id": ensg_id,
            "logrank_p": round(float(lr.p_value), 6),
            "median_cutoff_log2cpm": round(float(median_expr), 3),
            "high": {
                "n": int(len(high)),
                "timeline_months": [round(float(t), 2) for t in kmf_high.survival_function_.index],
                "survival_prob": [round(float(s), 4) for s in kmf_high.survival_function_.iloc[:, 0]],
            },
            "low": {
                "n": int(len(low)),
                "timeline_months": [round(float(t), 2) for t in kmf_low.survival_function_.index],
                "survival_prob": [round(float(s), 4) for s in kmf_low.survival_function_.iloc[:, 0]],
            },
        }

    fig.suptitle(
        "Kaplan–Meier Survival Stratification of Prioritized LUAD Hubs (n = 361)",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved high-res KM panel to '{output_png}'")

    with open(output_json, "w") as jf:
        json.dump(web_export, jf, indent=2)
    print(f"Saved interactive web KM coordinates to '{output_json}'")
    plt.show()


if __name__ == "__main__":
    generate_km_panel()