import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def generate_km_panel(
    genes=("NEK2", "PLK1", "KIF14", "DLGAP5"),
    cache_path="results/stage1_cache.pkl",
    cohort_csv="results/LUAD_survival_cohort.csv",
    screen_csv="results/LUAD_all_survival_screen.csv",
    output_png="plots/LUAD_top_targets_KM_panel.png",
    output_json="results/LUAD_km_curves_web.json",
):
    """
    KM curves for the SAME cohort (patients, samples, expression scale, median split) used by the survival screen.
    Requires survival_and_hubs.run_survival_screen() to have been run first (it writes LUAD_survival_cohort.csv).
    The log-rank p-value is asserted to match the screen CSV so the figure and the dashboard cannot drift apart.
    """
    if not os.path.exists(cohort_csv):
        raise FileNotFoundError(f"{cohort_csv} not found. Run the survival screen (deseq2_screen.py / run_survival_screen) first.")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    expr_mat = cache["filtered_expression"]
    annotated_de = cache["annotated_de"]

    # Same expression scale logic as the screen (cached matrix is already log2-CPM)
    if float(expr_mat.max().max()) > 50:
        expr_log = np.log2(expr_mat.div(expr_mat.sum(axis=0), axis=1) * 1e6 + 1.0)
    else:
        expr_log = expr_mat
    expr_log.index = expr_log.index.astype(str).str.split(".").str[0]

    cohort = pd.read_csv(cohort_csv)
    screen = pd.read_csv(screen_csv).set_index("symbol") if os.path.exists(screen_csv) else None

    de = annotated_de.copy()
    de["ensembl_id"] = de.index.astype(str).str.split(".").str[0]
    symbol_to_ensg = de[~de["symbol"].astype(str).str.startswith("ENSG")].drop_duplicates(subset=["symbol"]).set_index("symbol")["ensembl_id"].to_dict()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    web_export = {}
    n_used = None

    for idx, symbol in enumerate(genes):
        ax = axes[idx]
        ensg_id = symbol_to_ensg.get(symbol)
        if not ensg_id or ensg_id not in expr_log.index:
            print(f"Warning: {symbol} not found in expression matrix.")
            continue

        df_gene = cohort.copy()
        df_gene["expr"] = df_gene["sample_id"].map(expr_log.loc[ensg_id])
        df_gene = df_gene.dropna(subset=["time", "event", "expr"])
        df_gene = df_gene[df_gene["time"] > 0].copy()
        df_gene["months"] = df_gene["time"] / 30.44
        n_used = len(df_gene)

        median_expr = df_gene["expr"].median()
        high = df_gene[df_gene["expr"] >= median_expr]
        low = df_gene[df_gene["expr"] < median_expr]

        lr = logrank_test(high["months"], low["months"], event_observed_A=high["event"], event_observed_B=low["event"])

        if screen is not None and symbol in screen.index and np.isfinite(screen.loc[symbol, "km_logrank_p"]):
            ref_p = float(screen.loc[symbol, "km_logrank_p"])
            assert abs(lr.p_value - ref_p) <= 1e-6 + 1e-3 * ref_p, (
                f"{symbol}: figure log-rank p ({lr.p_value:.6g}) != survival screen p ({ref_p:.6g}); "
                "figure and screen are using different cohorts."
            )

        kmf_high, kmf_low = KaplanMeierFitter(), KaplanMeierFitter()
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
        ax.text(0.96, 0.92, f"Log-Rank p = {lr.p_value:.4g} (nominal)", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9))

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
        f"Exploratory Median-Split Kaplan–Meier Curves, TCGA-LUAD (n = {n_used})",
        fontsize=14, weight="bold", y=0.99,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KM panel to '{output_png}'")

    with open(output_json, "w") as jf:
        json.dump(web_export, jf, indent=2)
    print(f"Saved web KM coordinates to '{output_json}'")


if __name__ == "__main__":
    generate_km_panel()