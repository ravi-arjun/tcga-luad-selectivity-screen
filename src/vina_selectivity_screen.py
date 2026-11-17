import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy

VINA_EXE = os.path.join("bin", "vina.exe")

# Curated 10-ligand clinical & FDA-approved kinase inhibitor library (including positive controls)
LIGAND_LIBRARY = {
    "BI-2536": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCN(C)CC4)cc3OC)ncc2N1C1CCCC1",
    "Volasertib": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCC(N5CCN(CC(C)C)CC5)CC4)cc3OC)ncc2N1C(C)C",
    "Hesperadin": "CCN1CCN(CC(=O)Nc2ccc(C(=C3C(=O)Nc4ccc(S(=O)(=O)CC)cc43)c3ccccc3)cc2)CC1",
    "Rigosertib": "COc1cc(OC)c(CS(=O)(=O)C=Cc2ccc(OC)c(NCC(=O)O)c2)c(OC)c1",
    "Alisertib": "COc1cccc(C2=NCc3cnc(Nc4ccc(C(=O)O)cc4OC)nc3-c3ccc(Cl)cc32)c1F",
    "Barasertib": "CCN(CC)CC(=O)Nc1ccc(Nc2ncnc3cc(OCCCN4CCN(CCO)CC4)c(OC)cc23)cc1",
    "Dasatinib": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "Erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "Sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "Tofacitinib": "CC1CCN(C(=O)CC#N)CC1N(C)c1ncnc2[nH]ccc12",
}

TARGET_CONFIG = {
    "NEK2": {
        "holo_pdb": "structures/targets/NEK2_2W5B.pdb",
        "chain": "A",
        "offtargets": [
            ("NEK4", "structures/offtargets/NEK2_offtarget_NEK4_P51957.pdb"),
            ("NEK8", "structures/offtargets/NEK2_offtarget_NEK8_Q86SG6.pdb"),
            ("NEK7", "structures/offtargets/NEK2_offtarget_NEK7_Q8TDX7.pdb"),
        ],
    },
    "PLK1": {
        "holo_pdb": "structures/targets/PLK1_2OWB.pdb",
        "chain": "A",
        "offtargets": [
            ("RPS6KB1", "structures/offtargets/PLK1_offtarget_RPS6KB1_P23443.pdb"),
            ("MARK2", "structures/offtargets/PLK1_offtarget_MARK2_Q7KZI7.pdb"),
            ("PHKG2", "structures/offtargets/PLK1_offtarget_PHKG2_P15735.pdb"),
        ],
    },
}


class CleanProteinSelect(Select):
    """Keeps only standard amino acid protein atoms (strips waters, ions, co-crystallized ligands)."""
    def __init__(self, chain_id="A"):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "


def extract_pocket_center(pdb_path, chain_id="A"):
    """Computes the 3D centroid of the co-crystallized organic ligand in the active site."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("holo", pdb_path)
    chain = structure[0][chain_id]

    ignore_het = {"HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "CL", "NA", "MG", "ZN", "DMS", "PEG"}
    best_coords = []

    for res in chain:
        hetflag = res.id[0]
        resname = res.resname.strip()
        if hetflag.startswith("H_") and resname not in ignore_het:
            coords = [atom.get_coord() for atom in res if atom.element != "H"]
            if len(coords) > len(best_coords):
                best_coords = coords

    if not best_coords:
        # Fallback to C-alpha center if no organic HETATM found
        best_coords = [res["CA"].get_coord() for res in chain if "CA" in res]

    center = np.mean(best_coords, axis=0)
    return tuple(round(float(c), 3) for c in center)


def pdb_to_receptor_pdbqt(pdb_path, pdbqt_path, chain_id="A"):
    """Writes a clean receptor PDBQT compatible with AutoDock Vina 1.2.7."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    tmp_pdb = pdbqt_path + ".clean.pdb"
    io.save(tmp_pdb, CleanProteinSelect(chain_id))

    # Map standard PDB elements to AutoDock4 atom types
    ad4_map = {"C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "HD", "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I"}
    with open(tmp_pdb, "r") as fin, open(pdbqt_path, "w") as fout:
        for line in fin:
            if line.startswith("ATOM"):
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = line[12:16].strip()[0].upper()
                if elem == "H":
                    continue
                ad_type = ad4_map.get(elem, "C")
                # Format valid PDBQT line: columns 1-66 from PDB + partial charge + AD4 type
                pdbqt_line = f"{line[:54]:<54}  1.00  0.00    +0.000 {ad_type:<2}\n"
                fout.write(pdbqt_line)
    os.remove(tmp_pdb)


def superpose_offtarget_to_primary(primary_pdb, offtarget_pdb, aligned_pdb_out, primary_chain="A"):
    """
    Superposes an AlphaFold off-target structure onto the primary holo crystal structure
    via C-alpha backbone alignment so the active-site docking grid maps onto the paralog pocket.
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", primary_pdb)
    mob_struct = parser.get_structure("mob", offtarget_pdb)

    ref_cas = [r["CA"] for r in ref_struct[0][primary_chain] if r.id[0] == " " and "CA" in r]
    mob_chain = list(mob_struct[0].get_chains())[0]
    mob_cas = [r["CA"] for r in mob_chain if r.id[0] == " " and "CA" in r]

    # Align across matching kinase-domain length window
    n_align = min(len(ref_cas), len(mob_cas))
    sup = Superimposer()
    sup.set_atoms(ref_cas[:n_align], mob_cas[:n_align])
    sup.apply(mob_struct.get_atoms())

    io = PDBIO()
    io.set_structure(mob_struct)
    io.save(aligned_pdb_out, CleanProteinSelect(mob_chain.id))
    return round(float(sup.rms), 3)


def prepare_ligands(lig_dir="docking/ligands"):
    """Generates 3D MMFF94-minimized PDBQT ligands via RDKit + Meeko and computes ADME metrics."""
    os.makedirs(lig_dir, exist_ok=True)
    preparator = MoleculePreparation()
    adme_records = []

    for name, smiles in LIGAND_LIBRARY.items():
        mol = Chem.MolFromSmiles(smiles)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol_h)

        mol_setups = preparator.prepare(mol_h)
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            print(f"[WARN] Meeko failed for {name}: {err}")
            continue

        lig_pdbqt = os.path.join(lig_dir, f"{name}.pdbqt")
        with open(lig_pdbqt, "w") as f:
            f.write(pdbqt_str)

        adme_records.append({
            "ligand": name,
            "smiles": smiles,
            "MW": round(Descriptors.MolWt(mol), 2),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "pdbqt_path": lig_pdbqt,
        })

    return pd.DataFrame(adme_records)


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, out_pose_pdbqt, box_size=(20, 20, 20), exhaustiveness=8):
    """Executes AutoDock Vina 1.2.7 and parses the top binding affinity (kcal/mol)."""
    cmd = [
        VINA_EXE,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--out", out_pose_pdbqt,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse mode 1 affinity from Vina stdout table
    for line in res.stdout.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1))
    return np.nan


if __name__ == "__main__":
    os.makedirs("docking/receptors", exist_ok=True)
    os.makedirs("docking/poses", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("1. Preparing 3D ligand library (RDKit ETKDGv3 + MMFF94 + Meeko)...")
    lig_df = prepare_ligands()
    lig_df.to_csv("results/LUAD_ligand_adme_profile.csv", index=False)
    print(f"Prepared {len(lig_df)} drug-like ligands.\n")

    all_docking_rows = []
    pocket_metadata = {}

    for target_sym, cfg in TARGET_CONFIG.items():
        center = extract_pocket_center(cfg["holo_pdb"], cfg["chain"])
        pocket_metadata[target_sym] = {"center": center, "box_size": [20, 20, 20]}
        print(f"=== TARGET: {target_sym} | ATP-Pocket Center: {center} ===")

        primary_rec_pdbqt = f"docking/receptors/{target_sym}_primary.pdbqt"
        pdb_to_receptor_pdbqt(cfg["holo_pdb"], primary_rec_pdbqt, cfg["chain"])

        # Prepare aligned off-target receptors
        offtarget_receptors = {}
        for off_sym, off_pdb in cfg["offtargets"]:
            aligned_pdb = f"docking/receptors/{target_sym}_aligned_{off_sym}.pdb"
            rmsd = superpose_offtarget_to_primary(cfg["holo_pdb"], off_pdb, aligned_pdb, cfg["chain"])
            off_pdbqt = f"docking/receptors/{target_sym}_offtarget_{off_sym}.pdbqt"
            pdb_to_receptor_pdbqt(aligned_pdb, off_pdbqt, chain_id="A")
            offtarget_receptors[off_sym] = off_pdbqt
            print(f"  Aligned off-target {off_sym} onto {target_sym} (C-alpha RMSD = {rmsd} A)")

        # Dock every ligand against primary target + 3 structural off-targets
        for _, lrow in lig_df.iterrows():
            lig_name = lrow["ligand"]
            lig_pdbqt = lrow["pdbqt_path"]
            pose_out = f"docking/poses/{target_sym}_{lig_name}_pose.pdbqt"

            dg_primary = run_vina_docking(primary_rec_pdbqt, lig_pdbqt, center, pose_out)

            off_scores = {}
            for off_sym, off_pdbqt in offtarget_receptors.items():
                off_pose = f"docking/poses/{target_sym}_vs_{off_sym}_{lig_name}.pdbqt"
                off_scores[off_sym] = run_vina_docking(off_pdbqt, lig_pdbqt, center, off_pose)

            best_off_sym = min(off_scores, key=off_scores.get)
            best_off_dg = off_scores[best_off_sym]
            # Negative ddG = stronger binding to primary target than closest structural off-target
            ddg = round(dg_primary - best_off_dg, 3)

            row_data = {
                "primary_target": target_sym,
                "ligand": lig_name,
                "dG_primary_kcal": dg_primary,
                "closest_offtarget": best_off_sym,
                "dG_closest_offtarget": best_off_dg,
                "ddG_selectivity": ddg,
                "MW": lrow["MW"],
                "cLogP": lrow["cLogP"],
                "TPSA": lrow["TPSA"],
                "pose_file": pose_out,
            }
            for osym, osc in off_scores.items():
                row_data[f"dG_{osym}"] = osc

            all_docking_rows.append(row_data)
            print(f"  {lig_name:<12} | {target_sym}: {dg_primary:>6.2f} kcal/mol | Best Off-Target ({best_off_sym}): {best_off_dg:>6.2f} | ddG: {ddg:>+6.2f}")

    results_df = pd.DataFrame(all_docking_rows).sort_values(
        by=["primary_target", "ddG_selectivity", "dG_primary_kcal"], ascending=[True, True, True]
    )
    results_df.to_csv("results/LUAD_vina_selectivity_screen.csv", index=False)

    with open("results/LUAD_docking_pockets.json", "w") as jf:
        json.dump(pocket_metadata, jf, indent=2)

    print("\n--- TOP SELECTIVE CANDIDATES (SORTED BY ddG SELECTIVITY MARGIN) ---")
    print(
        results_df[
            ["primary_target", "ligand", "dG_primary_kcal", "closest_offtarget", "dG_closest_offtarget", "ddG_selectivity", "cLogP", "TPSA"]
        ].to_string(index=False)
    )