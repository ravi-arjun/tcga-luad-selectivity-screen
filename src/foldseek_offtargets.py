import os
import time
import requests
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1


TARGETS = {
    "NEK2": {
        "uniprot": "P51955",
        "pdb_id": "2W5B",   # Co-crystallized human NEK2 kinase domain
        "chain": "A",
    },
    "PLK1": {
        "uniprot": "P53350",
        "pdb_id": "2OWB",   # Co-crystallized human PLK1 kinase domain (with BI-2536)
        "chain": "A",
    },
}


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded: {dest_path}")


def download_alphafold_pdb(uniprot_id, dest_path):
    """Queries AlphaFold DB API to get the latest model version URL and downloads it."""
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    pdb_url = resp.json()[0]["pdbUrl"]
    download_file(pdb_url, dest_path)


def fetch_target_structures():
    """Downloads both co-crystallized RCSB PDB and AlphaFold PDB structures."""
    for symbol, info in TARGETS.items():
        pdb_url = f"https://files.rcsb.org/download/{info['pdb_id']}.pdb"
        download_file(pdb_url, f"structures/targets/{symbol}_{info['pdb_id']}.pdb")
        download_alphafold_pdb(
            info["uniprot"], f"structures/targets/{symbol}_AF_{info['uniprot']}.pdb"
        )


def validate_atp_pocket_alignment(holo_pdb, offtarget_pdb, chain_id="A", pocket_radius=10.0):
    """
    Validates Foldseek candidates at the catalytic ATP-binding pocket level:
    1. Identifies holo ATP-pocket residues within `pocket_radius` A of the co-crystallized ligand.
    2. Aligns kinase domain sequences to map equivalent pocket C-alpha atoms.
    3. Computes both Kinase-Domain RMSD and ATP-Pocket Local RMSD (A).
    """
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", holo_pdb)
    mob_struct = parser.get_structure("mob", offtarget_pdb)

    ref_chain = ref_struct[0][chain_id]
    mob_chain = list(mob_struct[0].get_chains())[0]

    # Locate co-crystallized organic ligand atoms
    ignore_het = {"HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "CL", "NA", "MG", "ZN", "DMS", "PEG"}
    lig_coords = []
    for res in ref_chain:
        if res.id[0].startswith("H_") and res.resname.strip() not in ignore_het:
            coords = [a.get_coord() for a in res if a.element != "H"]
            if len(coords) > len(lig_coords):
                lig_coords = coords
    lig_coords = np.array(lig_coords)

    ref_residues = [r for r in ref_chain if r.id[0] == " " and "CA" in r]
    mob_residues = [r for r in mob_chain if r.id[0] == " " and "CA" in r]

    # Identify ATP-pocket residues in primary target
    pocket_res_ids = set()
    for r in ref_residues:
        ca = r["CA"].get_coord()
        if len(lig_coords) > 0 and np.min(np.linalg.norm(lig_coords - ca, axis=1)) <= pocket_radius:
            pocket_res_ids.add(r.id)

    ref_seq = "".join(seq1(r.resname) for r in ref_residues)
    mob_seq = "".join(seq1(r.resname) for r in mob_residues)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5
    aln = aligner.align(ref_seq, mob_seq)[0]

    domain_ref_cas, domain_mob_cas = [], []
    pocket_ref_cas, pocket_mob_cas = [], []
    pocket_conserved = 0

    for (r_start, r_end), (m_start, m_end) in zip(*aln.aligned):
        for ri, mi in zip(range(r_start, r_end), range(m_start, m_end)):
            r_res = ref_residues[ri]
            m_res = mob_residues[mi]
            domain_ref_cas.append(r_res["CA"])
            domain_mob_cas.append(m_res["CA"])
            if r_res.id in pocket_res_ids:
                pocket_ref_cas.append(r_res["CA"])
                pocket_mob_cas.append(m_res["CA"])
                if r_res.resname == m_res.resname:
                    pocket_conserved += 1

    sup_dom = Superimposer()
    sup_dom.set_atoms(domain_ref_cas, domain_mob_cas)
    domain_rmsd = round(float(sup_dom.rms), 3)

    if len(pocket_ref_cas) >= 4:
        sup_pkt = Superimposer()
        sup_pkt.set_atoms(pocket_ref_cas, pocket_mob_cas)
        pocket_rmsd = round(float(sup_pkt.rms), 3)
    else:
        pocket_rmsd = domain_rmsd

    pkt_identity = round(100.0 * pocket_conserved / max(len(pocket_ref_cas), 1), 1)
    return domain_rmsd, pocket_rmsd, pkt_identity, len(pocket_ref_cas)


def query_foldseek_human_proteome(symbol, pdb_path, top_k=5):
    """
    Submits a 3D protein structure to the Foldseek API against the human AlphaFold proteome
    to identify candidate structural off-targets for ATP-pocket superposition.
    """
    print(f"\nSubmitting {symbol} ({pdb_path}) to Foldseek API...")
    submit_url = "https://search.foldseek.com/api/ticket"
    with open(pdb_path, "rb") as f:
        resp = requests.post(
            submit_url,
            files={"q": f},
            data={"mode": "3diaa", "database[]": ["afdb-proteome"]},
            timeout=60,
        )
    resp.raise_for_status()
    ticket_id = resp.json()["id"]
    print(f"Foldseek ticket created: {ticket_id}")

    status_url = f"https://search.foldseek.com/api/ticket/{ticket_id}"
    for _ in range(30):
        st = requests.get(status_url, timeout=30).json()
        if st.get("status") == "COMPLETE":
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"Foldseek search timed out for {symbol}")

    res_url = f"https://search.foldseek.com/api/result/{ticket_id}/0"
    res_data = requests.get(res_url, timeout=60).json()

    hits = []
    self_uniprot = TARGETS[symbol]["uniprot"]

    for result_block in res_data.get("results", []):
        for aln in result_block.get("alignments", [[]])[0]:
            target_name = aln.get("target", "")
            if "9606" not in str(aln.get("taxId", "")) and "Homo sapiens" not in str(aln.get("taxName", "")):
                continue
            parts = target_name.split("-")
            uniprot_id = parts[1] if len(parts) > 1 else target_name
            if uniprot_id == self_uniprot:
                continue

            hits.append({
                "primary_target": symbol,
                "offtarget_uniprot": uniprot_id,
                "offtarget_desc": aln.get("description", target_name),
                "prob": aln.get("prob", 0),
                "evalue": aln.get("eval", 1.0),
                "seq_id": aln.get("seqId", 0),
                "aln_len": aln.get("alnLength", 0),
            })

    df_hits = pd.DataFrame(hits).drop_duplicates(subset=["offtarget_uniprot"]).head(top_k)

    symbols = []
    for uid in df_hits["offtarget_uniprot"]:
        u_resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json", timeout=15)
        if u_resp.status_code == 200:
            u_json = u_resp.json()
            gname = u_json.get("genes", [{}])[0].get("geneName", {}).get("value", uid)
        else:
            gname = uid
        symbols.append(gname)

        if len(symbols) <= 3:
            dest = f"structures/offtargets/{symbol}_offtarget_{gname}_{uid}.pdb"
            if not os.path.exists(dest):
                try:
                    download_alphafold_pdb(uid, dest)
                except Exception as e:
                    print(f"Could not download {uid}: {e}")

    df_hits["offtarget_symbol"] = symbols
    return df_hits


if __name__ == "__main__":
    fetch_target_structures()
    offtarget_csv = "results/LUAD_foldseek_offtargets.csv"

    if os.path.exists(offtarget_csv):
        print(f"\n[CACHE HIT] Loading existing Foldseek hits from '{offtarget_csv}'...")
        combined = pd.read_csv(offtarget_csv)
    else:
        all_offtargets = []
        for sym, info in TARGETS.items():
            af_path = f"structures/targets/{sym}_AF_{info['uniprot']}.pdb"
            off_df = query_foldseek_human_proteome(sym, af_path, top_k=5)
            all_offtargets.append(off_df)
        combined = pd.concat(all_offtargets, ignore_index=True)

    # Validate kinase-domain & ATP-pocket structural similarity for top 3 off-targets per target
    pocket_records = []
    for target, group in combined.groupby("primary_target"):
        holo_pdb = f"structures/targets/{target}_{TARGETS[target]['pdb_id']}.pdb"
        chain_id = TARGETS[target]["chain"]
        for _, row in group.head(3).iterrows():
            uid = row["offtarget_uniprot"]
            sym = row["offtarget_symbol"]
            dest = f"structures/offtargets/{target}_offtarget_{sym}_{uid}.pdb"
            if not os.path.exists(dest):
                download_alphafold_pdb(uid, dest)
            dom_rmsd, pkt_rmsd, pkt_id, n_pkt = validate_atp_pocket_alignment(holo_pdb, dest, chain_id)
            pocket_records.append({
                "primary_target": target,
                "offtarget_symbol": sym,
                "kinase_domain_rmsd_A": dom_rmsd,
                "atp_pocket_rmsd_A": pkt_rmsd,
                "atp_pocket_seq_id_pct": pkt_id,
                "aligned_pocket_residues": n_pkt,
            })

    pocket_df = pd.DataFrame(pocket_records)
    combined = pd.merge(
        combined.drop(columns=[c for c in pocket_df.columns if c not in ("primary_target", "offtarget_symbol")], errors="ignore"),
        pocket_df,
        on=["primary_target", "offtarget_symbol"],
        how="left",
    )
    os.makedirs("results", exist_ok=True)
    combined.to_csv(offtarget_csv, index=False)

    print("\n--- FOLDSEEK + ATP-POCKET SUPERPOSITION VALIDATION ---")
    print(combined[["primary_target", "offtarget_symbol", "seq_id", "kinase_domain_rmsd_A", "atp_pocket_rmsd_A", "atp_pocket_seq_id_pct"]].to_string(index=False))