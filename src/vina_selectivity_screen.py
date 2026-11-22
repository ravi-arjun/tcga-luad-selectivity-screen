import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy

VINA_EXE = os.path.join("bin", "vina.exe")

# Curated 10-ligand clinical & FDA-approved kinase inhibitor cross-screening panel
LIGAND_LIBRARY = {
    "BI-2536": {
        "smiles": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCN(C)CC4)cc3OC)ncc2N1C1CCCC1",
        "established_target": "PLK1 (Polo-like kinase 1; secondary PLK2/PLK3)",
        "clinical_status": "Phase II (mitotic kinase inhibitor)",
    },
    "Volasertib": {
        "smiles": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCC(N5CCN(CC(C)C)CC5)CC4)cc3OC)ncc2N1C(C)C",
        "established_target": "PLK1 (Polo-like kinase 1)",
        "clinical_status": "Phase III (mitotic kinase inhibitor)",
    },
    "Hesperadin": {
        "smiles": "CCN1CCN(CC(=O)Nc2ccc(C(=C3C(=O)Nc4ccc(S(=O)(=O)CC)cc43)c3ccccc3)cc2)CC1",
        "established_target": "AURKB (Aurora B kinase)",
        "clinical_status": "Preclinical mitotic tool compound",
    },
    "Rigosertib": {
        "smiles": "COc1cc(OC)c(CS(=O)(=O)C=Cc2ccc(OC)c(NCC(=O)O)c2)c(OC)c1",
        "established_target": "PLK1 / RAS-PI3K pathway modulator",
        "clinical_status": "Phase III",
    },
    "Alisertib": {
        "smiles": "COc1cccc(C2=NCc3cnc(Nc4ccc(C(=O)O)cc4OC)nc3-c3ccc(Cl)cc32)c1F",
        "established_target": "AURKA (Aurora A kinase)",
        "clinical_status": "Phase III (mitotic kinase inhibitor)",
    },
    "Barasertib": {
        "smiles": "CCN(CC)CC(=O)Nc1ccc(Nc2ncnc3cc(OCCCN4CCN(CCO)CC4)c(OC)cc23)cc1",
        "established_target": "AURKB (Aurora B kinase)",
        "clinical_status": "Phase II/III",
    },
    "Dasatinib": {
        "smiles": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "established_target": "ABL1 / SRC family kinases",
        "clinical_status": "FDA-approved",
    },
    "Erlotinib": {
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "established_target": "EGFR (ErbB1 receptor tyrosine kinase)",
        "clinical_status": "FDA-approved (NSCLC)",
    },
    "Sunitinib": {
        "smiles": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
        "established_target": "VEGFR2 / PDGFRB / KIT multi-RTK",
        "clinical_status": "FDA-approved",
    },
    "Tofacitinib": {
        "smiles": "CC1CCN(C(=O)CC#N)CC1N(C)c1ncnc2[nH]ccc12",
        "established_target": "JAK1 / JAK3 (Janus kinases)",
        "clinical_status": "FDA-approved",
    },
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
                pdbqt_line = f"{line[:54]:<54}  1.00  0.00    +0.000 {ad_type:<2}\n"
                fout.write(pdbqt_line)
    os.remove(tmp_pdb)


def superpose_offtarget_to_primary(primary_pdb, offtarget_pdb, aligned_pdb_out, primary_chain="A"):
    """
    Sequence-guided structural superposition of the off-target kinase domain onto the
    primary holo crystal structure so the ATP-binding site maps accurately onto the paralog pocket.
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", primary_pdb)
    mob_struct = parser.get_structure("mob", offtarget_pdb)

    ref_residues = [r for r in ref_struct[0][primary_chain] if r.id[0] == " " and "CA" in r]
    mob_chain = list(mob_struct[0].get_chains())[0]
    mob_residues = [r for r in mob_chain if r.id[0] == " " and "CA" in r]

    ref_seq = "".join(seq1(r.resname) for r in ref_residues)
    mob_seq = "".join(seq1(r.resname) for r in mob_residues)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    aln = aligner.align(ref_seq, mob_seq)[0]

    ref_cas, mob_cas = [], []
    for (r_start, r_end), (m_start, m_end) in zip(*aln.aligned):
        for ri, mi in zip(range(r_start, r_end), range(m_start, m_end)):
            ref_cas.append(ref_residues[ri]["CA"])
            mob_cas.append(mob_residues[mi]["CA"])

    sup = Superimposer()
    sup.set_atoms(ref_cas, mob_cas)
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

    for name, info in LIGAND_LIBRARY.items():
        smiles = info["smiles"]
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
            "established_target": info["established_target"],
            "clinical_status": info["clinical_status"],
            "smiles": smiles,
            "MW": round(Descriptors.MolWt(mol), 2),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "pdbqt_path": lig_pdbqt,
        })

    return pd.DataFrame(adme_records)


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, out_pose_pdbqt, box_size=(20, 20, 20), exhaustiveness=8, seed=42):
    """Executes AutoDock Vina 1.2.7 with a fixed seed and parses the top docking score."""
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
        "--seed", str(seed),
        "--out", out_pose_pdbqt,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1))
    return np.nan


if __name__ == "__main__":
    os.makedirs("docking/receptors", exist_ok=True)
    os.makedirs("docking/poses", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("1. Preparing 3D ligand cross-screening panel (RDKit ETKDGv3 + MMFF94 + Meeko)...")
    lig_df = prepare_ligands()
    lig_df.to_csv("results/LUAD_ligand_adme_profile.csv", index=False)
    print(f"Prepared {len(lig_df)} clinical/preclinical kinase inhibitors.\n")

    all_docking_rows = []
    pocket_metadata = {}

    for target_sym, cfg in TARGET_CONFIG.items():
        center = extract_pocket_center(cfg["holo_pdb"], cfg["chain"])
        pocket_metadata[target_sym] = {"center": center, "box_size": [20, 20, 20]}
        print(f"=== TARGET: {target_sym} | ATP-Pocket Center: {center} ===")

        primary_rec_pdbqt = f"docking/receptors/{target_sym}_primary.pdbqt"
        pdb_to_receptor_pdbqt(cfg["holo_pdb"], primary_rec_pdbqt, cfg["chain"])

        offtarget_receptors = {}
        for off_sym, off_pdb in cfg["offtargets"]:
            aligned_pdb = f"docking/receptors/{target_sym}_aligned_{off_sym}.pdb"
            rmsd = superpose_offtarget_to_primary(cfg["holo_pdb"], off_pdb, aligned_pdb, cfg["chain"])
            off_pdbqt = f"docking/receptors/{target_sym}_offtarget_{off_sym}.pdbqt"
            pdb_to_receptor_pdbqt(aligned_pdb, off_pdbqt, chain_id="A")
            offtarget_receptors[off_sym] = off_pdbqt
            print(f"  Aligned kinase domain of {off_sym} onto {target_sym} (Domain C-alpha RMSD = {rmsd} A)")

        for _, lrow in lig_df.iterrows():
            lig_name = lrow["ligand"]
            lig_pdbqt = lrow["pdbqt_path"]
            pose_out = f"docking/poses/{target_sym}_{lig_name}_pose.pdbqt"

            s_primary = run_vina_docking(primary_rec_pdbqt, lig_pdbqt, center, pose_out)

            off_scores = {}
            for off_sym, off_pdbqt in offtarget_receptors.items():
                off_pose = f"docking/poses/{target_sym}_vs_{off_sym}_{lig_name}.pdbqt"
                off_scores[off_sym] = run_vina_docking(off_pdbqt, lig_pdbqt, center, off_pose)

            best_off_sym = min(off_scores, key=off_scores.get)
            best_off_score = off_scores[best_off_sym]
            # Negative delta_S_vina = more favorable Vina docking score on primary target than closest off-target
            delta_s_vina = round(s_primary - best_off_score, 3)

            row_data = {
                "primary_target": target_sym,
                "ligand": lig_name,
                "established_target": lrow["established_target"],
                "clinical_status": lrow["clinical_status"],
                "score_primary_vina": s_primary,
                "closest_offtarget": best_off_sym,
                "score_closest_offtarget_vina": best_off_score,
                "delta_S_vina": delta_s_vina,
                "MW": lrow["MW"],
                "cLogP": lrow["cLogP"],
                "TPSA": lrow["TPSA"],
                "pose_file": pose_out,
            }
            for osym, osc in off_scores.items():
                row_data[f"score_{osym}_vina"] = osc

            all_docking_rows.append(row_data)
            print(f"  {lig_name:<12} | {target_sym}: {s_primary:>6.2f} | Best Off-Target ({best_off_sym}): {best_off_score:>6.2f} | dS_Vina: {delta_s_vina:>+6.2f}")

    results_df = pd.DataFrame(all_docking_rows).sort_values(
        by=["primary_target", "delta_S_vina", "score_primary_vina"], ascending=[True, True, True]
    )
    results_df.to_csv("results/LUAD_vina_selectivity_screen.csv", index=False)

    with open("results/LUAD_docking_pockets.json", "w") as jf:
        json.dump(pocket_metadata, jf, indent=2)

    print("\n--- TOP COMPUTATIONAL SELECTIVITY CANDIDATES (SORTED BY dS_Vina MARGIN) ---")
    print(
        results_df[
            ["primary_target", "ligand", "established_target", "score_primary_vina", "closest_offtarget", "score_closest_offtarget_vina", "delta_S_vina"]
        ].to_string(index=False)
    )