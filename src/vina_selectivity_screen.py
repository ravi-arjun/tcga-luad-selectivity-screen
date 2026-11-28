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
from rdkit.Chem import AllChem, Descriptors, rdFMCS
from meeko import MoleculePreparation, PDBQTWriterLegacy

VINA_EXE = os.path.join("bin", "vina.exe")

# Comparators whose ATP-pocket superposition onto the primary target is worse than this (C-alpha RMSD, Angstrom)
# are still docked and reported, but are EXCLUDED from the delta-S margin because the same rigid box does not
# sample an equivalent pocket geometry.
MAX_COMPARATOR_POCKET_RMSD = 3.0

# Isomeric SMILES (PubChem / RCSB PDB ligand definitions). Clinical-status strings are deliberately coarse
# (approved vs. investigational) because dated trial outcomes must be verified against primary sources before being added.
LIGAND_LIBRARY = {
    "BI-2536": {
        "smiles": "CC[C@@H]1C(=O)N(C)c2nc(Nc3ccc(C(=O)NC4CCN(C)CC4)cc3OC)ncc2N1C1CCCC1",
        "stereochemistry_note": "(7R)-enantiomer (matches RCSB ligand R78 in PDB 2RKU)",
        "established_target": "PLK1",
        "atp_competitive": True,
        "clinical_status": "Investigational (clinical trials); not approved; development discontinued",
    },
    "Volasertib": {
        "smiles": "CC[C@@H]1C(=O)N(C)c2nc(Nc3ccc(C(=O)N[C@H]4CC[C@H](N5CCN(CC(C)C)CC5)CC4)cc3OC)ncc2N1C(C)C",
        "stereochemistry_note": "(7R)-enantiomer with trans-1,4-cyclohexyl linker",
        "established_target": "PLK1",
        "atp_competitive": True,
        "clinical_status": "Investigational (reached Phase III in AML); not approved",
    },
    "Hesperadin": {
        "smiles": "CCN1CCN(CC(=O)Nc2ccc(C(=C3C(=O)Nc4ccc(S(=O)(=O)CC)cc43)c3ccccc3)cc2)CC1",
        "stereochemistry_note": "Z-oxindole isomer (achiral)",
        "established_target": "AURKB",
        "atp_competitive": True,
        "clinical_status": "Preclinical mitotic tool compound",
    },
    "Rigosertib": {
        "smiles": "COc1cc(OC)c(CS(=O)(=O)/C=C/c2ccc(OC)c(NCC(=O)O)c2)c(OC)c1",
        "stereochemistry_note": "(E)-styryl sulfone isomer",
        "established_target": "RAS-PI3K / PLK1 pathway",
        "atp_competitive": False,
        "clinical_status": "Investigational; Phase III INSPIRE (HR-MDS) did not meet its OS endpoint (2020); non-ATP-competitive modulator",
    },
    "Alisertib": {
        "smiles": "COc1cccc(C2=NCc3cnc(Nc4ccc(C(=O)O)cc4OC)nc3-c3ccc(Cl)cc32)c1F",
        "stereochemistry_note": "Achiral benzazepine scaffold",
        "established_target": "AURKA",
        "atp_competitive": True,
        "clinical_status": "Investigational; not approved",
    },
    "Barasertib": {
        "smiles": "CCN(CC)CC(=O)Nc1ccc(Nc2ncnc3cc(OCCCN4CCN(CCO)CC4)c(OC)cc23)cc1",
        "stereochemistry_note": "Achiral quinazoline scaffold",
        "established_target": "AURKB",
        "atp_competitive": True,
        "clinical_status": "Investigational (AML trials); not approved",
    },
    "Dasatinib": {
        "smiles": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "stereochemistry_note": "Achiral thiazole-carboxamide scaffold",
        "established_target": "ABL1 / SRC",
        "atp_competitive": True,
        "clinical_status": "FDA-approved (2006; CML / Ph+ ALL)",
    },
    "Erlotinib": {
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "stereochemistry_note": "Achiral quinazoline scaffold",
        "established_target": "EGFR",
        "atp_competitive": True,
        "clinical_status": "FDA-approved (2004; EGFR-mutant NSCLC)",
    },
    "Sunitinib": {
        "smiles": "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",
        "stereochemistry_note": "(Z)-indolin-2-one isomer",
        "established_target": "VEGFR2 / PDGFRB / KIT",
        "atp_competitive": True,
        "clinical_status": "FDA-approved (2006; RCC / GIST / pNET)",
    },
    "Tofacitinib": {
        "smiles": "C[C@@H]1CCN(C(=O)CC#N)C[C@@H]1N(C)c1ncnc2[nH]ccc12",
        "stereochemistry_note": "(3R,4R)-enantiomer (matches clinical citrate salt)",
        "established_target": "JAK1 / JAK3",
        "atp_competitive": True,
        "clinical_status": "FDA-approved (2012; RA / PsA / UC)",
    },
}

# Verified RCSB PDB co-crystal entries:
# - 2W5B: Human NEK2 kinase bound to ATPgammaS
# - 2OWB: Human PLK1 kinase bound to PHA-680626
# - 2RKU: Human PLK1 kinase bound to BI-2536 (R78; separate redocking control)
TARGET_CONFIG = {
    "NEK2": {
        "holo_pdb": "structures/targets/NEK2_2W5B.pdb",
        "pdb_description": "PDB 2W5B: Human NEK2 kinase bound to ATPgammaS",
        "chain": "A",
        "offtargets": [
            ("NEK4", "structures/offtargets/NEK2_offtarget_NEK4_P51957.pdb", "A", "Foldseek structural analog"),
            ("NEK7", "structures/offtargets/NEK2_offtarget_NEK7_Q8TDX7.pdb", "A", "Foldseek structural analog"),
            ("NEK8", "structures/offtargets/NEK2_offtarget_NEK8_Q86SG6.pdb", "A", "Foldseek structural analog"),
            ("PLK1_CTRL", "structures/targets/PLK1_2OWB.pdb", "A", "Established clinical target comparator (PDB 2OWB, PHA-680626-bound PLK1)"),
            ("AURKA_CTRL", "structures/targets/AURKA_1MQ4.pdb", "A", "Established clinical target comparator (PDB 1MQ4, ADP-bound AURKA)"),
        ],
    },
    "PLK1": {
        "holo_pdb": "structures/targets/PLK1_2OWB.pdb",
        "pdb_description": "PDB 2OWB: Human PLK1 kinase bound to PHA-680626",
        "chain": "A",
        "offtargets": [
            ("PHKG2", "structures/offtargets/PLK1_offtarget_PHKG2_P15735.pdb", "A", "Foldseek structural analog"),
            ("RPS6KB1", "structures/offtargets/PLK1_offtarget_RPS6KB1_P23443.pdb", "A", "Foldseek structural analog"),
            ("AURKA_CTRL", "structures/targets/AURKA_1MQ4.pdb", "A", "Established clinical target comparator (PDB 1MQ4, ADP-bound AURKA)"),
            ("NEK2_CTRL", "structures/targets/NEK2_2W5B.pdb", "A", "Primary mitotic kinase comparator (PDB 2W5B, ATPgammaS-bound NEK2)"),
        ],
    },
}


def get_vina_version():
    """Return the version string reported by the Vina executable (so reports never hardcode it)."""
    try:
        out = subprocess.run([VINA_EXE, "--version"], capture_output=True, text=True, check=True)
        txt = (out.stdout + out.stderr).strip()
        m = re.search(r"(\d+\.\d+\.\d+)", txt)
        return m.group(1) if m else txt.splitlines()[0]
    except Exception as e:
        return f"unknown ({e})"


def ensure_control_structures():
    aurka_pdb = "structures/targets/AURKA_1MQ4.pdb"
    if not os.path.exists(aurka_pdb):
        os.makedirs(os.path.dirname(aurka_pdb), exist_ok=True)
        urllib.request.urlretrieve("https://files.rcsb.org/download/1MQ4.pdb", aurka_pdb)


class CleanProteinSelect(Select):
    """Retains standard amino acid residues on the specified chain (explicitly strips waters/ions/cofactors)."""
    def __init__(self, chain_id="A"):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "

    def accept_atom(self, atom):
        return atom.get_altloc() in (" ", "A")


def extract_holo_ligand_info(pdb_path, chain_id="A"):
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
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    tmp_pdb = pdbqt_path + ".clean.pdb"
    io.save(tmp_pdb, CleanProteinSelect(chain_id))

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

    rdkit_h_added = False
    mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            polar_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)]
            mol_h = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=polar_idxs)
            Chem.MolToPDBFile(mol_h, tmp_pdb)
            rdkit_h_added = True
        except Exception as e:
            print(f"    [WARN] RDKit sanitization skipped for {os.path.basename(pdb_path)}: {e}")

    hd_count = 0
    with open(tmp_pdb, "r") as fin, open(pdbqt_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()

                if elem == "H":
                    ad_type = "HD"
                    hd_count += 1
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
    return {"rdkit_sanitized": rdkit_h_added, "polar_HD_atoms": hd_count}


def superpose_offtarget_by_atp_pocket(primary_pdb, offtarget_pdb, aligned_pdb_out, primary_chain="A", offtarget_chain="A", pocket_radius=10.0):
    """
    Superposes off-target protein onto primary ATP-pocket residues (within 10 A of ligand)
    and records local pocket C-alpha RMSD and pocket-to-grid-center offset.
    (Note: Geometric alignment alone does not guarantee rigid-receptor comparative docking equivalence.)
    """
    grid_center, lig_coords, _ = extract_holo_ligand_info(primary_pdb, primary_chain)
    grid_center_arr = np.array(grid_center)

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", primary_pdb)
    mob_struct = parser.get_structure("mob", offtarget_pdb)

    ref_chain = ref_struct[0][primary_chain]
    mob_chain = mob_struct[0][offtarget_chain] if offtarget_chain in mob_struct[0] else list(mob_struct[0].get_chains())[0]

    ref_residues = [r for r in ref_chain if r.id[0] == " " and "CA" in r]
    mob_residues = [r for r in mob_chain if r.id[0] == " " and "CA" in r]

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
    identical_res = 0
    for (r_start, r_end), (m_start, m_end) in zip(*aln.aligned):
        for ri, mi in zip(range(r_start, r_end), range(m_start, m_end)):
            if ref_residues[ri].id in pocket_res_ids:
                pocket_ref_cas.append(ref_residues[ri]["CA"])
                pocket_mob_cas.append(mob_residues[mi]["CA"])
                if ref_residues[ri].resname == mob_residues[mi].resname:
                    identical_res += 1

    sup = Superimposer()
    sup.set_atoms(pocket_ref_cas, pocket_mob_cas)
    sup.apply(mob_struct.get_atoms())

    mob_pkt_centroid = np.mean([a.get_coord() for a in pocket_mob_cas], axis=0)
    pocket_to_grid_dist = round(float(np.linalg.norm(mob_pkt_centroid - grid_center_arr)), 3)
    pocket_seq_id = round(100.0 * identical_res / max(len(pocket_ref_cas), 1), 1)

    io = PDBIO()
    io.set_structure(mob_struct)
    io.save(aligned_pdb_out, CleanProteinSelect(mob_chain.id))

    return {
        "pocket_rmsd_A": round(float(sup.rms), 3),
        "pocket_to_grid_center_A": pocket_to_grid_dist,
        "matched_pocket_residues": len(pocket_ref_cas),
        "pocket_seq_id_pct": pocket_seq_id,
    }


def prepare_ligands(lig_dir="docking/ligands"):
    os.makedirs(lig_dir, exist_ok=True)
    preparator = MoleculePreparation()
    desc_records = []

    for name, info in LIGAND_LIBRARY.items():
        smiles = info["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {name}: {smiles}")

        expected_chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if any(cip == "?" for _, cip in expected_chiral):
            raise ValueError(f"Unspecified stereocenter detected in {name}: {expected_chiral}")

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            raise RuntimeError(f"ETKDGv3 3D conformer embedding failed for {name}")
        AllChem.MMFFOptimizeMolecule(mol_h)

        # Verify 3D coordinates preserved exact CIP stereochemistry
        Chem.AssignStereochemistryFrom3D(mol_h)
        embedded_chiral = Chem.FindMolChiralCenters(mol_h, includeUnassigned=True)
        if expected_chiral != embedded_chiral:
            raise ValueError(f"3D stereochemistry mismatch for {name}: expected {expected_chiral}, got {embedded_chiral}")

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
            "atp_competitive": info["atp_competitive"],
            "stereochemistry": info["stereochemistry_note"],
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
    Positive control: redocks stereochemically verified (7R)-BI-2536 into PLK1 co-crystal structure
    PDB 2RKU (ligand R78) across 3 seeds and computes symmetry-aware heavy-atom RMSD.
    """
    pdb_2rku = "structures/targets/PLK1_2RKU.pdb"
    if not os.path.exists(pdb_2rku):
        urllib.request.urlretrieve("https://files.rcsb.org/download/2RKU.pdb", pdb_2rku)

    center_2rku, _, _ = extract_holo_ligand_info(pdb_2rku, chain_id="A")
    rec_2rku_pdbqt = "docking/receptors/PLK1_2RKU_control.pdbqt"
    pdb_to_receptor_pdbqt_protonated(pdb_2rku, rec_2rku_pdbqt, chain_id="A")

    redock_out = "docking/poses/CONTROL_PLK1_2RKU_BI2536_redock.pdbqt"
    mean_s, sd_s = run_vina_multiseed(rec_2rku_pdbqt, bi2536_pdbqt, center_2rku, redock_out, exhaustiveness=16, seeds=(42, 101, 2026))

    cryst_lines = []
    with open(pdb_2rku) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20].strip() == "R78" and line[21] == "A":
                cryst_lines.append(line)
    cryst_mol = Chem.MolFromPDBBlock("".join(cryst_lines), removeHs=True, sanitize=False)

    dock_lines = []
    with open(redock_out) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                ad4_t = line[77:79].strip()
                if ad4_t not in {"H", "HD"}:
                    elem = "C" if ad4_t == "A" else (ad4_t[0] if ad4_t in {"NA", "OA", "SA"} else ad4_t)
                    dock_lines.append(f"HETATM{line[6:66]}          {elem:>2}\n")
    dock_mol = Chem.MolFromPDBBlock("".join(dock_lines), removeHs=True, sanitize=False)

    mcs = rdFMCS.FindMCS([cryst_mol, dock_mol], bondCompare=rdFMCS.BondCompare.CompareAny, completeRingsOnly=False)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    cryst_matches = cryst_mol.GetSubstructMatches(patt, uniquify=False)
    dock_match = dock_mol.GetSubstructMatch(patt)

    cryst_conf = cryst_mol.GetConformer()
    dock_conf = dock_mol.GetConformer()

    # Evaluate all symmetry-equivalent atom mappings and take minimum heavy-atom RMSD
    best_rmsd = np.inf
    for cm in cryst_matches:
        sq_dists = [
            np.sum((np.array(cryst_conf.GetAtomPosition(ci)) - np.array(dock_conf.GetAtomPosition(di))) ** 2)
            for ci, di in zip(cm, dock_match)
        ]
        best_rmsd = min(best_rmsd, float(np.sqrt(np.mean(sq_dists))))

    rmsd = round(best_rmsd, 3)
    passed = bool(rmsd <= 2.0)
    status_lbl = "PASSED (<= 2.0 A)" if passed else "FAILED (> 2.0 A)"

    # Diagnostic: split the RMSD (best symmetry mapping) into ring atoms vs. non-ring (substituent/tail) atoms
    Chem.FastFindRings(cryst_mol)
    best_map, best_val = None, np.inf
    for cm in cryst_matches:
        val = float(np.sqrt(np.mean([
            np.sum((np.array(cryst_conf.GetAtomPosition(ci)) - np.array(dock_conf.GetAtomPosition(di))) ** 2)
            for ci, di in zip(cm, dock_match)
        ])))
        if val < best_val:
            best_val, best_map = val, cm
    ring_sq, chain_sq = [], []
    for ci, di in zip(best_map, dock_match):
        d2 = float(np.sum((np.array(cryst_conf.GetAtomPosition(ci)) - np.array(dock_conf.GetAtomPosition(di))) ** 2))
        (ring_sq if cryst_mol.GetAtomWithIdx(ci).IsInRing() else chain_sq).append(d2)
    ring_rmsd = round(float(np.sqrt(np.mean(ring_sq))), 3) if ring_sq else None
    nonring_rmsd = round(float(np.sqrt(np.mean(chain_sq))), 3) if chain_sq else None
    if passed:
        diagnosis = "Pose recovered within 2.0 A."
    elif ring_rmsd is not None and ring_rmsd <= 2.0:
        diagnosis = f"Ring atoms recovered ({ring_rmsd} A); the deviation arises from non-ring substituent atoms ({nonring_rmsd} A)."
    else:
        diagnosis = f"Ring atoms were also not recovered ({ring_rmsd} A); failure is not confined to flexible substituents (cause not diagnosed)."
    n_ligand_heavy = int(cryst_mol.GetNumAtoms())
    print(f"[POSITIVE CONTROL] PLK1 (2RKU) + (7R)-BI-2536 Redocking: Score = {mean_s:.2f} +/- {sd_s:.2f}, Heavy-Atom RMSD = {rmsd:.3f} A "
          f"({len(dock_match)}/{n_ligand_heavy} heavy atoms matched; ring {ring_rmsd} A, non-ring {nonring_rmsd} A) -> {status_lbl}\n{diagnosis}\n")
    return {
        "pdb_id": "2RKU",
        "ligand": "(7R)-BI-2536 (R78)",
        "vina_score": mean_s,
        "seed_sd": sd_s,
        "heavy_atom_rmsd_A": rmsd,
        "matched_heavy_atoms": len(dock_match),
        "ligand_heavy_atoms": n_ligand_heavy,
        "ring_atom_rmsd_A": ring_rmsd,
        "nonring_atom_rmsd_A": nonring_rmsd,
        "diagnosis": diagnosis,
        "acceptance_threshold_A": 2.0,
        "passed": passed,
        "status_label": status_lbl,
        "pose_file": redock_out,
    }


if __name__ == "__main__":
    os.makedirs("docking/receptors", exist_ok=True)
    os.makedirs("docking/poses", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ensure_control_structures()
    vina_version = get_vina_version()
    print(f"AutoDock Vina version: {vina_version}")

    print("1. Preparing stereochemically verified 3D ligand panel (RDKit ETKDGv3 + MMFF94 + Meeko)...")
    lig_df = prepare_ligands()
    lig_df.to_csv("results/LUAD_ligand_physicochemical_descriptors.csv", index=False)

    all_docking_rows = []
    pocket_metadata = {"vina_version": vina_version, "max_comparator_pocket_rmsd_A": MAX_COMPARATOR_POCKET_RMSD,
                       "targets": {}, "alignments": {}}

    for target_sym, cfg in TARGET_CONFIG.items():
        center, _, holo_res = extract_holo_ligand_info(cfg["holo_pdb"], cfg["chain"])
        holo_resname = holo_res.resname.strip() if holo_res is not None else None
        if holo_resname is None:
            print(f"[WARN] No co-crystal ligand found in {cfg['holo_pdb']}; grid centered on C-alpha centroid.")
        primary_rec_pdbqt = f"docking/receptors/{target_sym}_primary.pdbqt"
        prep_stats = pdb_to_receptor_pdbqt_protonated(cfg["holo_pdb"], primary_rec_pdbqt, cfg["chain"])

        pocket_metadata["targets"][target_sym] = {
            "holo_pdb": cfg["holo_pdb"],
            "pdb_description": cfg["pdb_description"],
            "holo_ligand_resname": holo_resname,
            "grid_center": center,
            "box_size": [20, 20, 20],
            "receptor_prep": prep_stats,
        }
        print(f"=== TARGET: {target_sym} ({cfg['pdb_description']}) | Grid: {center} | Polar HD: {prep_stats['polar_HD_atoms']} ===")

        if target_sym == "PLK1":
            bi_pdbqt = lig_df[lig_df["ligand"] == "BI-2536"]["pdbqt_path"].values[0]
            pocket_metadata["positive_control_redocking"] = validate_cocrystal_redocking(bi_pdbqt)

        offtarget_receptors = {}
        pocket_metadata["alignments"][target_sym] = {}

        for off_sym, off_pdb, off_chain, rationale in cfg["offtargets"]:
            aligned_pdb = f"docking/receptors/{target_sym}_pocket_aligned_{off_sym}.pdb"
            aln_metrics = superpose_offtarget_by_atp_pocket(
                cfg["holo_pdb"], off_pdb, aligned_pdb, cfg["chain"], off_chain
            )
            off_pdbqt = f"docking/receptors/{target_sym}_offtarget_{off_sym}.pdbqt"
            off_prep = pdb_to_receptor_pdbqt_protonated(aligned_pdb, off_pdbqt, chain_id=off_chain)
            aln_metrics["rationale"] = rationale
            aln_metrics["receptor_prep"] = off_prep
            pocket_metadata["alignments"][target_sym][off_sym] = aln_metrics
            offtarget_receptors[off_sym] = off_pdbqt

            print(
                f"  {off_sym:<10} -> {target_sym} | Pocket RMSD = {aln_metrics['pocket_rmsd_A']:.3f} A | "
                f"Pocket-to-Grid Dist = {aln_metrics['pocket_to_grid_center_A']:.3f} A"
            )

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

            # Reliable comparators = pocket-aligned within MAX_COMPARATOR_POCKET_RMSD; all are still reported
            reliable = {k: v for k, v in off_scores.items()
                        if np.isfinite(v) and pocket_metadata["alignments"][target_sym][k]["pocket_rmsd_A"] <= MAX_COMPARATOR_POCKET_RMSD}
            if not reliable:
                raise RuntimeError(f"No comparator for {target_sym} has pocket RMSD <= {MAX_COMPARATOR_POCKET_RMSD} A")
            best_off_sym = min(reliable, key=reliable.get)
            best_off_score = reliable[best_off_sym]
            delta_s_vina = round(s_primary - best_off_score, 3)
            best_all_sym = min(off_scores, key=off_scores.get)
            delta_s_all = round(s_primary - off_scores[best_all_sym], 3)

            row_data = {
                "primary_target": target_sym,
                "ligand": lig_name,
                "established_target": lrow["established_target"],
                "atp_competitive": lrow["atp_competitive"],
                "clinical_status": lrow["clinical_status"],
                "score_primary_vina": s_primary,
                "seed_sd_vina": sd_primary,
                "best_scoring_comparator": best_off_sym,
                "score_best_comparator_vina": best_off_score,
                "delta_S_vina": delta_s_vina,
                "comparators_used": ";".join(sorted(reliable)),
                "best_scoring_comparator_all": best_all_sym,
                "delta_S_vina_all_comparators": delta_s_all,
                "MW": lrow["MW"],
                "cLogP": lrow["cLogP"],
                "TPSA": lrow["TPSA"],
                "pose_file": pose_out,
            }
            for osym, osc in off_scores.items():
                row_data[f"score_{osym}_vina"] = osc

            all_docking_rows.append(row_data)

    results_df = pd.DataFrame(all_docking_rows)

    # Rank ATP-pocket selectivity ONLY across ATP-competitive ligands (exclude non-ATP-competitive Rigosertib)
    results_df["atp_pocket_selectivity_rank"] = np.nan
    for t_sym in results_df["primary_target"].unique():
        mask = (results_df["primary_target"] == t_sym) & (results_df["atp_competitive"] == True)
        ranked_idx = results_df.loc[mask].sort_values(
            by=["delta_S_vina", "score_primary_vina"], ascending=[True, True]
        ).index
        results_df.loc[ranked_idx, "atp_pocket_selectivity_rank"] = np.arange(1, len(ranked_idx) + 1)

    results_df = results_df.sort_values(
        by=["primary_target", "atp_competitive", "delta_S_vina", "score_primary_vina"],
        ascending=[True, False, True, True],
    )
    results_df.to_csv("results/LUAD_vina_selectivity_screen.csv", index=False)

    with open("results/LUAD_docking_pockets.json", "w") as jf:
        json.dump(pocket_metadata, jf, indent=2)

    print("\n--- COUNTER-SCREENING SUMMARY (ATP-COMPETITIVE PANEL + RIGOSERTIB COMPARATOR; dS = S_primary - best-scoring pocket-aligned comparator) ---")
    print(
        results_df[
            ["primary_target", "ligand", "atp_competitive", "atp_pocket_selectivity_rank", "score_primary_vina", "best_scoring_comparator", "score_best_comparator_vina", "delta_S_vina"]
        ].to_string(index=False)
    )