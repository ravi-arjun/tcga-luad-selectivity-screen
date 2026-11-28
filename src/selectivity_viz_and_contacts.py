import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_pdbqt_atoms(pdbqt_path):
    """Parses ATOM/HETATM coordinates, atom names, residue names, and AutoDock4 atom types from PDBQT."""
    atoms = []
    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("MODEL") and atoms:
                break  # Only read Mode 1 (top-ranked Vina pose)
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ad4_type = line[77:79].strip().upper() or atom_name[0].upper()
                if ad4_type not in {"H", "HD"}:
                    atoms.append({
                        "atom": atom_name,
                        "res": f"{res_name}{res_seq}",
                        "ad4_type": ad4_type,
                        "coord": np.array([x, y, z]),
                    })
    return atoms


def extract_pocket_contacts(receptor_pdbqt, ligand_pose_pdbqt, cutoff=3.8):
    """Identifies receptor residues within `cutoff` (Angstroms) of the docked ligand pose."""
    rec_atoms = parse_pdbqt_atoms(receptor_pdbqt)
    lig_atoms = parse_pdbqt_atoms(ligand_pose_pdbqt)
    if not rec_atoms or not lig_atoms:
        return []

    lig_coords = np.array([a["coord"] for a in lig_atoms])
    residue_min_dist = {}

    for ratom in rec_atoms:
        dists = np.linalg.norm(lig_coords - ratom["coord"], axis=1)
        min_d = float(np.min(dists))
        if min_d <= cutoff:
            res_label = ratom["res"]
            is_polar = ratom["ad4_type"] in {"N", "NA", "O", "OA", "S", "SA"}
            if res_label not in residue_min_dist or min_d < residue_min_dist[res_label]["min_dist_A"]:
                residue_min_dist[res_label] = {
                    "residue": res_label,
                    "min_dist_A": round(min_d, 2),
                    "contact_category": "Polar Atom Proximity (<3.4 A)" if (is_polar and min_d <= 3.4) else "Steric / Non-Polar Proximity (<3.8 A)",
                }

    return sorted(residue_min_dist.values(), key=lambda x: x["min_dist_A"])


def build_selectivity_plots_and_contacts(
    screen_csv="results/LUAD_vina_selectivity_screen.csv",
    out_png="plots/LUAD_selectivity_summary.png",
    out_json="results/LUAD_web_dashboard_bundle.json",
):
    df = pd.read_csv(screen_csv)
    os.makedirs("plots", exist_ok=True)

    # 1. Extract pocket contacts for all 20 primary target-ligand poses
    contact_summary = {}
    for _, row in df.iterrows():
        tgt = row["primary_target"]
        lig = row["ligand"]
        rec_p = f"docking/receptors/{tgt}_primary.pdbqt"
        pose_p = row["pose_file"]
        if os.path.exists(rec_p) and os.path.exists(pose_p):
            contacts = extract_pocket_contacts(rec_p, pose_p, cutoff=3.8)
            contact_summary[f"{tgt}_{lig}"] = contacts
            if (tgt == "NEK2" and lig in {"Alisertib", "BI-2536"}) or (tgt == "PLK1" and lig in {"Rigosertib", "BI-2536"}):
                top_res = ", ".join([f"{c['residue']} ({c['min_dist_A']}A)" for c in contacts[:6]])
                print(f"[{tgt} + {lig}] Key Pocket Proximity Contacts (<3.8 A): {top_res}")

    # 2. Generate 2-Panel Summary Figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    ax1 = axes[0]
    colors = {"NEK2": "#e67e22", "PLK1": "#8e44ad"}
    for tgt, grp in df.groupby("primary_target"):
        ax1.scatter(
            grp["score_primary_vina"],
            grp["delta_S_vina"],
            s=110,
            color=colors[tgt],
            edgecolors="black",
            alpha=0.88,
            label=f"{tgt} Primary Target",
            zorder=3,
        )
        for _, r in grp.iterrows():
            ax1.annotate(
                r["ligand"],
                (r["score_primary_vina"], r["delta_S_vina"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
            )

    ax1.axhline(-0.5, color="#27ae60", linestyle="--", linewidth=1.2, label="Score Margin Reference (dS_Vina <= -0.5)")
    ax1.axvline(-8.5, color="#2980b9", linestyle=":", linewidth=1.2, label="Vina Score Reference (S_Vina <= -8.5)")
    ax1.axvspan(-10.5, -8.5, ymin=0, ymax=0.35, color="#2ecc71", alpha=0.08)
    ax1.set_title("A. Vina Docking Score vs. Pocket-Aligned Off-Target Differential", fontsize=12, weight="bold")
    ax1.set_xlabel("Primary Target Vina Score S_primary  [More Negative = Stronger]", fontsize=10)
    ax1.set_ylabel("Score Differential dS_Vina = S_primary - S_best-scoring pocket-aligned comparator", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2 = axes[1]
    ligands = df[df["primary_target"] == "NEK2"].sort_values("delta_S_vina")["ligand"].tolist()
    x = np.arange(len(ligands))
    width = 0.38

    nek2_ds = [df[(df["primary_target"] == "NEK2") & (df["ligand"] == l)]["delta_S_vina"].values[0] for l in ligands]
    plk1_ds = [df[(df["primary_target"] == "PLK1") & (df["ligand"] == l)]["delta_S_vina"].values[0] for l in ligands]

    ax2.bar(x - width / 2, nek2_ds, width, label="NEK2 vs. pocket-aligned comparators", color="#e67e22", edgecolor="black")
    ax2.bar(x + width / 2, plk1_ds, width, label="PLK1 vs. pocket-aligned comparators", color="#8e44ad", edgecolor="black")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(-0.5, color="#27ae60", linestyle="--", linewidth=1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ligands, rotation=35, ha="right", fontsize=9)
    ax2.set_title("B. Pocket-Aligned Counter-Screen Score Differential (dS_Vina)", fontsize=12, weight="bold")
    ax2.set_ylabel("dS_Vina Differential  [Negative = Primary Preferred]", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved selectivity summary figure to '{out_png}'")

    bundle = {
        "generated_from": screen_csv,
        "generated_from_mtime": os.path.getmtime(screen_csv),
        "docking_results": df.to_dict(orient="records"),
        "pocket_contacts": contact_summary,
    }
    with open(out_json, "w") as jf:
        json.dump(bundle, jf, indent=2)
    print(f"Saved web dashboard data bundle to '{out_json}'")


if __name__ == "__main__":
    build_selectivity_plots_and_contacts()