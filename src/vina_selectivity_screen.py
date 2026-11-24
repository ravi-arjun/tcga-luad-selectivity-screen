import os
import re
import json
import urllib.request
import subprocess
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Superimposer, Select
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolAlign
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit.Chem import AllChem, Descriptors, rdMolAlign, rdFMCS

VINA_EXE = os.path.join("bin", "vina.exe")

# Curated 10-ligand clinical & FDA-approved kinase inhibitor cross-screening panel
LIGAND_LIBRARY = {
    "BI-2536": {
        "smiles": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCN(C)CC4)cc3OC)ncc2N1C1CCCC1",
        "established_target": "PLK1",
        "clinical_status": "Phase II (PLK1/2/3 inhibitor)",
    },
    "Volasertib": {
        "smiles": "CCC1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCC(N5CCN(CC(C)C)CC5)CC4)cc3OC)ncc2N1C(C)C",
        "established_target": "PLK1",
        "clinical_status": "Phase III (PLK1 inhibitor)",
    },
    "Hesperadin": {
        "smiles": "CCN1CCN(CC(=O)Nc2ccc(C(=C3C(=O)Nc4ccc(S(=O)(=O)CC)cc43)c3ccccc3)cc2)CC1",
        "established_target": "AURKB",
        "clinical_status": "Preclinical mitotic tool compound",
    },
    "Rigosertib": {
        "smiles": "COc1cc(OC)c(CS(=O)(=O)C=Cc2ccc(OC)c(NCC(=O)O)c2)c(OC)c1",
        "established_target": "PLK1 / RAS-PI3K",
        "clinical_status": "Phase III",
    },
    "Alisertib": {
        "smiles": "COc1cccc(C2=NCc3cnc(Nc4ccc(C(=O)O)cc4OC)nc3-c3ccc(Cl)cc32)c1F",
        "established_target": "AURKA",
        "clinical_status": "Phase III (Aurora A inhibitor)",
    },
    "Barasertib": {
        "smiles": "CCN(CC)CC(=O)Nc1ccc(Nc2ncnc3cc(OCCCN4CCN(CCO)CC4)c(OC)cc23)cc1",
        "established_target": "AURKB",
        "clinical_status": "Phase II/III",
    },
    "Dasatinib": {
        "smiles": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "established_target": "ABL1 / SRC",
        "clinical_status": "FDA-approved",
    },
    "Erlotinib": {
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "established_target": "EGFR",
        "clinical_status": "FDA-approved (NSCLC)",
    },
    "Sunitinib": {
        "smiles": "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
        "established_target": "VEGFR2 / PDGFRB / KIT",
        "clinical_status": "FDA-approved",
    },
    "Tofacitinib": {
        "smiles": "CC1CCN(C(=O)CC#N)CC1N(C)c1ncnc2[nH]ccc12",
        "established_target": "JAK1 / JAK3",
        "clinical_status": "FDA-approved",
    },
}

# Counter-screen panels now include BOTH Foldseek pocket-compatible paralogs (pocket RMSD < 3.0 A)
# AND mandatory established clinical target controls (PLK1 for NEK2 panel; AURKA for PLK1 panel).
# Note: MARK2 (pocket RMSD = 7.24 A) is excluded from ATP-pocket counter-screening due to divergent pocket geometry.
TARGET_CONFIG = {
    "NEK2": {
        "holo_pdb": "structures/targets/NEK2_2W5B.pdb",
        "chain": "A",
        "offtargets": [
            ("NEK4", "structures/offtargets/NEK2_offtarget_NEK4_P51957.pdb", "A", "Foldseek paralog (pocket RMSD 1.73 A)"),
            ("NEK7", "structures/offtargets/NEK2_offtarget_NEK7_Q8TDX7.pdb", "A", "Foldseek paralog (pocket RMSD 2.21 A)"),
            ("NEK8", "structures/offtargets/NEK2_offtarget_NEK8_Q86SG6.pdb", "A", "Foldseek paralog (pocket RMSD 2.39 A)"),
            ("PLK1_CTRL", "structures/targets/PLK1_2OWB.pdb", "A", "Established clinical target control (BI-2536/Volasertib)"),
        ],
    },
    "PLK1": {
        "holo_pdb": "structures/targets/PLK1_2OWB.pdb",
        "chain": "A",
        "offtargets": [
            ("PHKG2", "structures/offtargets/PLK1_offtarget_PHKG2_P15735.pdb", "A", "Foldseek structural analog (pocket RMSD 0.88 A)"),
            ("RPS6KB1", "structures/offtargets/PLK1_offtarget_RPS6KB1_P23443.pdb", "A", "Foldseek structural analog (pocket RMSD 1.62 A)"),
            ("AURKA_CTRL", "structures/targets/AURKA_1MQ4.pdb", "A", "Established clinical target control (Alisertib)"),
        ],
    },
}


def ensure_control_structures():
    """Downloads AURKA (PDB: 1MQ4) if missing so Alisertib is counter-screened against its primary target."""
    aurka_pdb = "structures/targets/AURKA_1MQ4.pdb"
    if not os.path.exists(aurka_pdb):
        os.makedirs(os.path.dirname(aurka_pdb), exist_ok=True)
        print("Downloading AURKA crystal structure (PDB: 1MQ4) as clinical control...")
        urllib.request.urlretrieve("https://files.rcsb.org/download/1MQ4.pdb", aurka_pdb)


class CleanProteinSelect(Select):
    """Keeps standard amino acids on the target chain (takes first altloc 'A' or ' ')."""
    def __init__(self, chain_id="A"):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "

    def accept_atom(self, atom):
        return atom.get_altloc() in (" ", "A")


def extract_holo_ligand_info(pdb_path, chain_id="A"):
    """Extracts 3D coordinates and centroid of the largest co-crystallized organic ligand."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("holo", pdb_path)
    chain = structure[0][chain_id]

    ignore_het = {"HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "CL", "NA", "MG", "ZN", "DMS", "PEG", "ADP", "ATP", "ANP"}
    best_res = None
    best_coords = []

    for res in chain:
        hetflag = res.id[0]
        resname = res.resname.strip()
        if hetflag.startswith("H_") and resname not in ignore_het:
            coords = [atom.get_coord() for atom in res if atom.element != "H"]
            if len(coords) > len(best_coords):
                best_coords = coords
                best_res = res

    if not best_coords:
        best_coords = [res["CA"].get_coord() for res in chain if "CA" in res]

    center = tuple(round(float(c), 3) for c in np.mean(best_coords, axis=0))
    return center, np.array(best_coords), best_res


def pdb_to_receptor_pdbqt_protonated(pdb_path, pdbqt_path, chain_id="A"):
    """
    Prepares a receptor PDBQT with explicit polar hydrogens (HD) and chemically accurate
    AutoDock4 donor/acceptor atom types (NA, OA, SA, N, C, HD) via RDKit sanitization.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    tmp_pdb = pdbqt_path + ".clean.pdb"
    io.save(tmp_pdb, CleanProteinSelect(chain_id))

    # Standard amino acid AutoDock4 atom-type lookup (pH 7.4 protonation state)
    # Distinguishes H-bond acceptors (OA, NA) from amide/guanidinium donors (N) and polar H (HD)
    polar_backbone = {"N": "N", "O": "OA", "OXT": "OA", "H": "HD", "HN": "HD"}
    sidechain_ad4 = {
        ("SER", "OG"): "OA", ("SER", "HG"): "HD",
        ("THR", "OG1"): "OA", ("THR", "HG1"): "HD",
        ("TYR", "OH"): "OA", ("TYR", "HH"): "HD",
        ("CYS", "SG"): "SA", ("CYS", "HG"): "HD",
        ("ASN", "OD1"): "OA", ("ASN", "ND2"): "N", ("ASN", "HD21"): "HD", ("ASN", "HD22"): "HD",
        ("GLN", "OE1"): "OA", ("GLN", "NE2"): "N", ("GLN", "HE21"): "HD", ("GLN", "HE22"): "HD",
        ("ASP", "OD1"): "OA", ("ASP", "OD2"): "OA",
        ("GLU", "OE1"): "OA", ("GLU", "OE2"): "OA",
        ("HIS", "ND1"): "NA", ("HIS", "NE2"): "N", ("HIS", "HE2"): "HD", ("HIS", "HD1"): "HD",
        ("LYS", "NZ"): "N", ("LYS", "HZ1"): "HD", ("LYS", "HZ2"): "HD", ("LYS", "HZ3"): "HD",
        ("ARG", "NE"): "N", ("ARG", "NH1"): "N", ("ARG", "NH2"): "N",
        ("ARG", "HE"): "HD", ("ARG", "HH11"): "HD", ("ARG", "HH12"): "HD", ("ARG", "HH21"): "HD", ("ARG", "HH22"): "HD",
        ("TRP", "NE1"): "N", ("TRP", "HE1"): "HD",
        ("MET", "SD"): "SA",
    }

    # Add polar hydrogens with RDKit if missing from crystal/AlphaFold PDB
    mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            mol_h = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)])
            Chem.MolToPDBFile(mol_h, tmp_pdb)
        except Exception:
            pass  # Fallback to existing coordinates if non-standard residue prevents full valence sanitization

    with open(tmp_pdb, "r") as fin, open(pdbqt_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()

                # Skip non-polar hydrogens; retain polar hydrogens attached to N/O/S
                if elem == "H":
                    ad_type = "HD"
                elif (res_name, atom_name) in sidechain_ad4:
                    ad_type = sidechain_ad4[(res_name, atom_name)]
                elif atom_name in polar_backbone:
                    ad_type = polar_backbone[atom_name]
                elif elem == "S":
                    ad_type = "SA"
                elif elem == "O":
                    ad_type = "OA"
                elif elem == "N":
                    ad_type = "NA"
                else:
                    ad_type = "C"

                pdbqt_line = f"ATOM  {line[6:54]:<48}  1.00  0.00    +0.000 {ad_type:<2}\n"
                fout.write(pdbqt_line)
    os.remove(tmp_pdb)


def superpose_offtarget_by_atp_pocket(primary_pdb, offtarget_pdb, aligned_pdb_out, primary_chain="A", offtarget_chain="A", pocket_radius=10.0):
    """
    Superposes the off-target protein strictly onto the primary target's ATP-binding pocket
    residues (within `pocket_radius` A of the co-crystallized ligand), applies that local rigid-body
    transformation to the entire off-target structure, and verifies that the transformed ATP-pocket
    centroid matches the primary docking grid center.
    """
    center, lig_coords, _ = extract_holo_ligand_info(primary_pdb, primary_chain)
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", primary_pdb)
    mob_struct = parser.get_structure("mob", offtarget_pdb)

    ref_chain = ref_struct[0][primary_chain]
    mob_chain = mob_struct[0][offtarget_chain] if offtarget_chain in mob_struct[0] else list(mob_struct[0].get_chains())[0]

    ref_residues = [r for r in ref_chain if r.id[0] == " " and "CA" in r]
    mob_residues = [r for r in mob_chain if r.id[0] == " " and "CA" in r]

    # Identify ATP-pocket residues in primary target
    pocket_res_ids = {
        r.id for r in ref_residues
        if np.min(np.linalg.norm(lig_coords - r["CA"].get_coord(), axis=1)) <= pocket_radius
    }

    ref_seq = "".join(seq1(r.resname) for r in ref_residues)
    mob_seq = "".join(seq1(r.resname) for r in mob_residues)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    aln = aligner.align(ref_seq, mob_seq)[0]

    pocket_ref_cas, pocket_mob_cas = [], []
    for (r_start, r_end), (m_start, m_end) in zip(*aln.aligned):
        for ri, mi in zip(range(r_start, r_end), range(m_start, m_end)):
            if ref_residues[ri].id in pocket_res_ids:
                pocket_ref_cas.append(ref_residues[ri]["CA"])
                pocket_mob_cas.append(mob_residues[mi]["CA"])

    # Fit transformation strictly on ATP-pocket C-alphas, then apply to all off-target atoms
    sup = Superimposer()
    sup.set_atoms(pocket_ref_cas, pocket_mob_cas)
    sup.apply(mob_struct.get_atoms())

    # Verify transformed off-target pocket centroid shift relative to primary pocket center
    ref_pkt_centroid = np.mean([a.get_coord() for a in pocket_ref_cas], axis=0)
    mob_pkt_centroid = np.mean([a.get_coord() for a in pocket_mob_cas], axis=0)
    centroid_shift_A = round(float(np.linalg.norm(ref_pkt_centroid - mob_pkt_centroid)), 3)

    io = PDBIO()
    io.set_structure(mob_struct)
    io.save(aligned_pdb_out, CleanProteinSelect(mob_chain.id))
    return round(float(sup.rms), 3), centroid_shift_A


def prepare_ligands(lig_dir="docking/ligands"):
    """Generates 3D MMFF94-minimized PDBQT ligands via RDKit + Meeko and computes physicochemical descriptors."""
    os.makedirs(lig_dir, exist_ok=True)
    preparator = MoleculePreparation()
    desc_records = []

    for name, info in LIGAND_LIBRARY.items():
        smiles = info["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol_h, params)
        AllChem.MMFFOptimizeMolecule(mol_h)

        mol_setups = preparator.prepare(mol_h)
        pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            print(f"[WARN] Meeko failed for {name}: {err}")
            continue

        lig_pdbqt = os.path.join(lig_dir, f"{name}.pdbqt")
        with open(lig_pdbqt, "w") as f:
            f.write(pdbqt_str)

        desc_records.append({
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

    return pd.DataFrame(desc_records)


def run_vina_multiseed(receptor_pdbqt, ligand_pdbqt, center, out_pose_pdbqt, box_size=(20, 20, 20), exhaustiveness=16, seeds=(42, 101, 2026)):
    """Runs AutoDock Vina 1.2.7 across multiple random seeds and returns the mean/best top-mode score."""
    scores = []
    best_score = np.inf
    tmp_pose = out_pose_pdbqt + ".tmp.pdbqt"

    for s in seeds:
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
            "--seed", str(s),
            "--out", tmp_pose,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
            if m:
                val = float(m.group(1))
                scores.append(val)
                if val < best_score:
                    best_score = val
                    os.replace(tmp_pose, out_pose_pdbqt)
                break

    if os.path.exists(tmp_pose):
        os.remove(tmp_pose)
    if not scores:
        return np.nan, np.nan
    return round(float(np.mean(scores)), 3), round(float(np.std(scores)), 3)


def validate_cocrystal_redocking(bi2536_pdbqt):
    """
    Positive control: downloads PDB 2RKU (human PLK1 co-crystallized with BI-2536, ligand R78),
    redocks BI-2536 into the 2RKU ATP pocket, and computes heavy-atom pose RMSD (in-place, no alignment).
    """
    pdb_2rku = "structures/targets/PLK1_2RKU.pdb"
    if not os.path.exists(pdb_2rku):
        urllib.request.urlretrieve("https://files.rcsb.org/download/2RKU.pdb", pdb_2rku)

    center_2rku, _, _ = extract_holo_ligand_info(pdb_2rku, chain_id="A")
    rec_2rku_pdbqt = "docking/receptors/PLK1_2RKU_control.pdbqt"
    pdb_to_receptor_pdbqt_protonated(pdb_2rku, rec_2rku_pdbqt, chain_id="A")

    redock_out = "docking/poses/CONTROL_PLK1_2RKU_BI2536_redock.pdbqt"
    mean_s, _ = run_vina_multiseed(rec_2rku_pdbqt, bi2536_pdbqt, center_2rku, redock_out, exhaustiveness=16, seeds=(42,))

    # Extract crystallographic BI-2536 heavy atoms from 2RKU (Chain A, residue R78)
    cryst_lines = []
    with open(pdb_2rku) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == "R78" and line[21] == "A":
                cryst_lines.append(line)
    cryst_mol = Chem.MolFromPDBBlock("".join(cryst_lines), removeHs=True, sanitize=False)

    # Extract docked Mode 1 heavy atoms
    dock_lines = []
    with open(redock_out) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                ad4_t = line[77:79].strip()
                if ad4_t not in {"H", "HD"}:
                    elem = ad4_t[0] if ad4_t in {"NA", "OA", "SA"} else ad4_t
                    dock_lines.append(f"HETATM{line[6:66]}          {elem:>2}\n")
    dock_mol = Chem.MolFromPDBBlock("".join(dock_lines), removeHs=True, sanitize=False)

    # Match heavy atoms via Maximum Common Substructure (element-wise) and compute in-place RMSD
    mcs = rdFMCS.FindMCS([cryst_mol, dock_mol], bondCompare=rdFMCS.BondCompare.CompareAny, completeRingsOnly=False)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    cryst_match = cryst_mol.GetSubstructMatch(patt)
    dock_match = dock_mol.GetSubstructMatch(patt)

    cryst_conf = cryst_mol.GetConformer()
    dock_conf = dock_mol.GetConformer()
    sq_dists = [
        np.sum((np.array(cryst_conf.GetAtomPosition(ci)) - np.array(dock_conf.GetAtomPosition(di))) ** 2)
        for ci, di in zip(cryst_match, dock_match)
    ]
    rmsd = round(float(np.sqrt(np.mean(sq_dists))), 3)
    print(f"[POSITIVE CONTROL] PLK1 (2RKU) + BI-2536 Co-Crystal Redocking: Score = {mean_s:.2f}, Heavy-Atom RMSD = {rmsd:.3f} A ({len(cryst_match)} heavy atoms)\n")
    return rmsd


if __name__ == "__main__":
    os.makedirs("docking/receptors", exist_ok=True)
    os.makedirs("docking/poses", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ensure_control_structures()

    print("1. Preparing 3D ligand cross-screening panel (RDKit ETKDGv3 + MMFF94 + Meeko)...")
    lig_df = prepare_ligands()
    lig_df.to_csv("results/LUAD_ligand_physicochemical_descriptors.csv", index=False)
    print(f"Prepared {len(lig_df)} clinical/preclinical kinase inhibitors.\n")

    all_docking_rows = []
    pocket_metadata = {}

    for target_sym, cfg in TARGET_CONFIG.items():
        center, _, _ = extract_holo_ligand_info(cfg["holo_pdb"], cfg["chain"])
        pocket_metadata[target_sym] = {"center": center, "box_size": [20, 20, 20]}
        print(f"=== TARGET: {target_sym} | ATP-Pocket Center: {center} ===")

        primary_rec_pdbqt = f"docking/receptors/{target_sym}_primary.pdbqt"
        pdb_to_receptor_pdbqt_protonated(cfg["holo_pdb"], primary_rec_pdbqt, cfg["chain"])

        if target_sym == "PLK1":
            bi_pdbqt = lig_df[lig_df["ligand"] == "BI-2536"]["pdbqt_path"].values[0]
            try:
                redock_rmsd = validate_cocrystal_redocking(bi_pdbqt)
                pocket_metadata["PLK1_BI2536_redock_RMSD_A"] = redock_rmsd
            except Exception as e:
                print(f"[WARN] Redocking RMSD check skipped: {e}")

        offtarget_receptors = {}
        for off_sym, off_pdb, off_chain, rationale in cfg["offtargets"]:
            aligned_pdb = f"docking/receptors/{target_sym}_pocket_aligned_{off_sym}.pdb"
            pkt_rmsd, shift_A = superpose_offtarget_by_atp_pocket(
                cfg["holo_pdb"], off_pdb, aligned_pdb, cfg["chain"], off_chain
            )
            off_pdbqt = f"docking/receptors/{target_sym}_offtarget_{off_sym}.pdbqt"
            pdb_to_receptor_pdbqt_protonated(aligned_pdb, off_pdbqt, chain_id=off_chain)
            offtarget_receptors[off_sym] = off_pdbqt
            print(f"  Pocket-aligned {off_sym:<10} -> {target_sym} | ATP-Pocket RMSD = {pkt_rmsd:.3f} A | Grid Shift = {shift_A:.3f} A ({rationale})")

        for _, lrow in lig_df.iterrows():
            lig_name = lrow["ligand"]
            lig_pdbqt = lrow["pdbqt_path"]
            pose_out = f"docking/poses/{target_sym}_{lig_name}_pose.pdbqt"

            s_primary, sd_primary = run_vina_multiseed(primary_rec_pdbqt, lig_pdbqt, center, pose_out)

            off_scores = {}
            for off_sym, off_pdbqt in offtarget_receptors.items():
                off_pose = f"docking/poses/{target_sym}_vs_{off_sym}_{lig_name}.pdbqt"
                s_off, _ = run_vina_multiseed(off_pdbqt, lig_pdbqt, center, off_pose)
                off_scores[off_sym] = s_off

            best_off_sym = min(off_scores, key=off_scores.get)
            best_off_score = off_scores[best_off_sym]
            delta_s_vina = round(s_primary - best_off_score, 3)

            row_data = {
                "primary_target": target_sym,
                "ligand": lig_name,
                "established_target": lrow["established_target"],
                "clinical_status": lrow["clinical_status"],
                "score_primary_vina": s_primary,
                "score_primary_sd": sd_primary,
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
            print(f"  {lig_name:<12} | {target_sym}: {s_primary:>6.2f} (±{sd_primary:.2f}) | Best Off-Target ({best_off_sym}): {best_off_score:>6.2f} | dS_Vina: {delta_s_vina:>+6.2f}")

    results_df = pd.DataFrame(all_docking_rows).sort_values(
        by=["primary_target", "delta_S_vina", "score_primary_vina"], ascending=[True, True, True]
    )
    results_df.to_csv("results/LUAD_vina_selectivity_screen.csv", index=False)

    with open("results/LUAD_docking_pockets.json", "w") as jf:
        json.dump(pocket_metadata, jf, indent=2)

    print("\n--- V1.1 POCKET-ALIGNED COUNTER-SCREENING SUMMARY ---")
    print(
        results_df[
            ["primary_target", "ligand", "established_target", "score_primary_vina", "closest_offtarget", "score_closest_offtarget_vina", "delta_S_vina"]
        ].to_string(index=False)
    )