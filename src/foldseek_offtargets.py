import os
import time
import json
import requests
import pandas as pd


TARGETS = {
    "NEK2": {
        "uniprot": "P51955",
        "pdb_id": "2W5B",   # Co-crystallized human NEK2 kinase domain with inhibitor
        "chain": "A",
    },
    "PLK1": {
        "uniprot": "P53350",
        "pdb_id": "2OWB",   # Co-crystallized human PLK1 kinase domain with BI2536
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

def query_foldseek_human_proteome(symbol, pdb_path, top_k=5):
    """
    Submits a 3D protein structure to the Foldseek API against the human AlphaFold proteome
    to identify structurally homologous off-target paralogs.
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

    # Poll until job completes
    status_url = f"https://search.foldseek.com/api/ticket/{ticket_id}"
    for _ in range(30):
        st = requests.get(status_url, timeout=30).json()
        if st.get("status") == "COMPLETE":
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"Foldseek search timed out for {symbol}")

    # Fetch alignment results
    res_url = f"https://search.foldseek.com/api/result/{ticket_id}/0"
    res_data = requests.get(res_url, timeout=60).json()

    hits = []
    self_uniprot = TARGETS[symbol]["uniprot"]

    for result_block in res_data.get("results", []):
        for aln in result_block.get("alignments", [[]])[0]:
            target_name = aln.get("target", "")
            # Filter to Homo sapiens (TAXID 9606 / UP000005640) and exclude self-hit
            if "9606" not in str(aln.get("taxId", "")) and "Homo sapiens" not in str(aln.get("taxName", "")):
                continue
            # Extract UniProt ID from AF-<UNIPROT>-F1-model_v4
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

    # Map UniProt IDs -> Gene Symbols via MyGene/UniProt REST API
    symbols = []
    for uid in df_hits["offtarget_uniprot"]:
        u_resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json", timeout=15)
        if u_resp.status_code == 200:
            u_json = u_resp.json()
            gname = u_json.get("genes", [{}])[0].get("geneName", {}).get("value", uid)
        else:
            gname = uid
        symbols.append(gname)

        # Download AlphaFold structure for top 3 off-targets for counter-docking
        if len(symbols) <= 3:
            af_off_url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
            try:
                download_file(af_off_url, f"structures/offtargets/{symbol}_offtarget_{gname}_{uid}.pdb")
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
        os.makedirs("results", exist_ok=True)
        combined.to_csv(offtarget_csv, index=False)

    # Download top 3 off-target AlphaFold structures per target (skips if already downloaded)
    for target, group in combined.groupby("primary_target"):
        for _, row in group.head(3).iterrows():
            uid = row["offtarget_uniprot"]
            sym = row["offtarget_symbol"]
            dest = f"structures/offtargets/{target}_offtarget_{sym}_{uid}.pdb"
            if not os.path.exists(dest):
                download_alphafold_pdb(uid, dest)

    print("\n--- FOLDSEEK STRUCTURAL OFF-TARGET PARALOGS ---")
    print(combined[["primary_target", "offtarget_symbol", "offtarget_uniprot", "prob", "evalue", "seq_id"]].to_string(index=False))