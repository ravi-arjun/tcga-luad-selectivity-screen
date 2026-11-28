"""
Recomputes the comparator-filtered delta-S table from an EXISTING Vina screen CSV (old 'closest_offtarget' schema)
and LUAD_docking_pockets.json, WITHOUT re-running Vina. Applies the same rule as vina_selectivity_screen.py:
comparators whose ATP-pocket C-alpha RMSD to the primary target exceeds MAX_COMPARATOR_POCKET_RMSD are excluded
from delta-S (they stay in the per-comparator score columns).

Usage:  python src/recompute_selectivity_from_scores.py
"""
import json
import numpy as np
import pandas as pd

CSV = "results/LUAD_vina_selectivity_screen.csv"
POCKETS = "results/LUAD_docking_pockets.json"
MAX_RMSD = 3.0


def main():
    df = pd.read_csv(CSV)
    if "best_scoring_comparator" in df.columns:
        print("CSV already uses the new schema; nothing to do.")
        return
    with open(POCKETS) as f:
        pockets = json.load(f)
    aln = pockets["alignments"]

    out_rows = []
    for _, r in df.iterrows():
        t = r["primary_target"]
        scores = {k: r[f"score_{k}_vina"] for k in aln[t] if f"score_{k}_vina" in df.columns and pd.notna(r[f"score_{k}_vina"])}
        reliable = {k: v for k, v in scores.items() if aln[t][k]["pocket_rmsd_A"] <= MAX_RMSD}
        best = min(reliable, key=reliable.get)
        best_all = min(scores, key=scores.get)
        row = r.to_dict()
        row.pop("closest_offtarget"); row.pop("score_closest_offtarget_vina")
        row.update({
            "best_scoring_comparator": best,
            "score_best_comparator_vina": reliable[best],
            "delta_S_vina": round(float(r["score_primary_vina"]) - reliable[best], 3),
            "comparators_used": ";".join(sorted(reliable)),
            "best_scoring_comparator_all": best_all,
            "delta_S_vina_all_comparators": round(float(r["score_primary_vina"]) - scores[best_all], 3),
        })
        out_rows.append(row)

    res = pd.DataFrame(out_rows)
    res["atp_pocket_selectivity_rank"] = np.nan
    for t in res["primary_target"].unique():
        m = (res["primary_target"] == t) & (res["atp_competitive"] == True)
        idx = res.loc[m].sort_values(["delta_S_vina", "score_primary_vina"]).index
        res.loc[idx, "atp_pocket_selectivity_rank"] = np.arange(1, len(idx) + 1)
    res = res.sort_values(["primary_target", "atp_competitive", "delta_S_vina", "score_primary_vina"], ascending=[True, False, True, True])
    res.to_csv(CSV, index=False)

    pockets["max_comparator_pocket_rmsd_A"] = MAX_RMSD
    with open(POCKETS, "w") as f:
        json.dump(pockets, f, indent=2)
    print(res[["primary_target", "ligand", "atp_pocket_selectivity_rank", "score_primary_vina", "best_scoring_comparator",
               "score_best_comparator_vina", "delta_S_vina", "delta_S_vina_all_comparators"]].to_string(index=False))


if __name__ == "__main__":
    main()
