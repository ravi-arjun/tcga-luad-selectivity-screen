import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_pdbqt_atoms(pdbqt_path):
    """Parses ATOM/HETATM coordinates, atom names, residue names, and residue numbers from PDBQT."""
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
                elem = line[77:79].strip() or atom_name[0]
                if elem.upper() != "H":
                    atoms.append({
                        "atom": atom_name,
                        "res": f"{res_name}{res_seq}",
                        "elem": elem.upper(),
                        "coord": np.array([x, y, z]),
                    })
    return atoms


def extract_pocket_contacts(receptor_pdbqt, ligand_pose_pdbqt, cutoff=3.8):
    """Identifies receptor residues within cutoff (Angstroms) of the docked ligand pose."""
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
            is_polar = ratom["elem"] in {"N", "O", "S"}
            if res_label not in residue_min_dist or min_d < residue_min_dist[res_label]["min_dist_A"]:
                residue_min_dist[res_label] = {
                    "residue": res_label,
                    "min_dist_A": round(min_d, 2),
                    "interaction_type": "Polar / H-Bond Candidate" if (is_polar and min_d <= 3.4) else "Hydrophobic / vdW",
                }

    return sorted(residue_min_dist.values(), key=lambda x: x["min_dist_A"])


def build_selectivity_plots_and_contacts(
    screen_csv="results/LUAD_vina_selectivity_screen.csv",
    out_png="plots/LUAD_selectivity_summary.png",
    out_json="results/LUAD_web_dashboard_bundle.json",
):
    df = pd.read_csv(screen_csv)
    os.makedirs("plots", exist_ok=True)

    # 1. Extract pocket contacts for top candidates
    contact_summary = {}
    pairs_to_inspect = [
        ("NEK2", "BI-2536", "docking/receptors/NEK2_primary.pdbqt", "docking/poses/NEK2_BI-2536_pose.pdbqt"),
        ("NEK2", "Dasatinib", "docking/receptors/NEK2_primary.pdbqt", "docking/poses/NEK2_Dasatinib_pose.pdbqt"),
        ("PLK1", "Alisertib", "docking/receptors/PLK1_primary.pdbqt", "docking/poses/PLK1_Alisertib_pose.pdbqt"),
        ("PLK1", "BI-2536", "docking/receptors/PLK1_primary.pdbqt", "docking/poses/PLK1_BI-2536_pose.pdbqt"),
    ]
    for tgt, lig, rec_p, pose_p in pairs_to_inspect:
        if os.path.exists(rec_p) and os.path.exists(pose_p):
            contacts = extract_pocket_contacts(rec_p, pose_p, cutoff=3.8)
            contact_summary[f"{tgt}_{lig}"] = contacts
            top_res = ", ".join([f"{c['residue']} ({c['min_dist_A']}A)" for c in contacts[:6]])
            print(f"[{tgt} + {lig}] Key Pocket Contacts (<3.8 A): {top_res}")

    # 2. Generate 2-Panel Summary Figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Panel A: Selectivity Margin (ddG) vs Primary Affinity (dG)
    ax1 = axes[0]
    colors = {"NEK2": "#e67e22", "PLK1": "#8e44ad"}
    for tgt, grp in df.groupby("primary_target"):
        ax1.scatter(
            grp["dG_primary_kcal"],
            grp["ddG_selectivity"],
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
                (r["dG_primary_kcal"], r["ddG_selectivity"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
            )

    ax1.axhline(-0.5, color="#27ae60", linestyle="--", linewidth=1.2, label="Selectivity Threshold (ddG <= -0.5)")
    ax1.axvline(-8.5, color="#2980b9", linestyle=":", linewidth=1.2, label="Potency Threshold (dG <= -8.5)")
    ax1.axvspan(-10.5, -8.5, ymin=0, ymax=0.35, color="#2ecc71", alpha=0.08)
    ax1.set_title("A. Potency vs. Paralog Selectivity Quadrant", fontsize=12, weight="bold")
    ax1.set_xlabel("Primary Target Binding Affinity dG (kcal/mol)  [More Negative = Stronger]", fontsize=10)
    ax1.set_ylabel("Selectivity Margin ddG = dG_primary - dG_offtarget (kcal/mol)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel B: Grouped Bar Chart of ddG Selectivity Across All 10 Ligands
    ax2 = axes[1]
    ligands = df[df["primary_target"] == "PLK1"].sort_values("ddG_selectivity")["ligand"].tolist()
    x = np.arange(len(ligands))
    width = 0.38

    nek2_ddg = [df[(df["primary_target"] == "NEK2") & (df["ligand"] == l)]["ddG_selectivity"].values[0] for l in ligands]
    plk1_ddg = [df[(df["primary_target"] == "PLK1") & (df["ligand"] == l)]["ddG_selectivity"].values[0] for l in ligands]

    ax2.bar(x - width / 2, nek2_ddg, width, label="NEK2 vs. NEK4/7/8", color="#e67e22", edgecolor="black")
    ax2.bar(x + width / 2, plk1_ddg, width, label="PLK1 vs. RPS6KB1/MARK2/PHKG2", color="#8e44ad", edgecolor="black")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(-0.5, color="#27ae60", linestyle="--", linewidth=1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ligands, rotation=35, ha="right", fontsize=9)
    ax2.set_title("B. Off-Target Counter-Screen Selectivity Margin (ddG)", fontsize=12, weight="bold")
    ax2.set_ylabel("ddG Selectivity Margin (kcal/mol)  [Negative = Selective]", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved publication selectivity summary figure to '{out_png}'")

    # 3. Export complete JSON bundle for web dashboard
    bundle = {
        "docking_results": df.to_dict(orient="records"),
        "pocket_contacts": contact_summary,
    }
    with open(out_json, "w") as jf:
        json.dump(bundle, jf, indent=2)
    print(f"Saved web dashboard data bundle to '{out_json}'")
    plt.show()


if __name__ == "__main__":
    build_selectivity_plots_and_contacts()