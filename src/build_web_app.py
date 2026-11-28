import os
import json
import math
import pandas as pd

PLACEHOLDER_NOTE = "not recorded; rerun the pipeline script that produces it"


def pdbqt_to_pdb_block(pdbqt_path, first_model_only=True):
    """Converts a receptor or docked ligand PDBQT file into a clean PDB text block for 3Dmol.js."""
    if not os.path.exists(pdbqt_path):
        return ""
    lines = []
    with open(pdbqt_path, "r") as f:
        for line in f:
            if first_model_only and line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")):
                elem = line[77:79].strip()
                if elem == "A":
                    elem = "C"
                elif elem in {"NA", "OA", "SA", "HD"}:
                    elem = elem[0]
                pdb_line = f"{line[:66]:<66}          {elem:>2}\n"
                lines.append(pdb_line)
    lines.append("END\n")
    return "".join(lines)


def build_static_site(out_dir="docs"):
    os.makedirs(out_dir, exist_ok=True)

    with open("results/LUAD_km_curves_web.json", "r") as f:
        km_data = json.load(f)

    with open("results/LUAD_web_dashboard_bundle.json", "r") as f:
        docking_bundle = json.load(f)

    # Read directly from the regenerated v1.2 Vina selectivity CSV so HTML never goes out of sync
    vina_csv = "results/LUAD_vina_selectivity_screen.csv"
    if os.path.exists(vina_csv):
        docking_df = pd.read_csv(vina_csv)
    else:
        docking_df = pd.DataFrame(docking_bundle["docking_results"])
    docking_df = docking_df.where(pd.notnull(docking_df), None)

    foldseek_df = pd.read_csv("results/LUAD_foldseek_offtargets.csv")
    foldseek_df = foldseek_df.where(pd.notnull(foldseek_df), None)

    # ---- survival summary: every number below is read/derived from result files, never hardcoded ----
    surv_csv = "results/LUAD_all_survival_screen.csv"
    sdf = pd.read_csv(surv_csv)
    summ_path = "results/LUAD_survival_summary.json"
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            surv = json.load(f)
    else:
        surv = {}
    n_tested = int(sdf["km_logrank_p"].notna().sum())
    n_nom_all = int((sdf["km_logrank_p"] < 0.05).sum())
    n_nom_hr = int(((sdf["km_logrank_p"] < 0.05) & (sdf["cox_HR"] > 1.0)).sum())
    n_fdr_all = int((sdf["km_logrank_padj"] < 0.10).sum())
    n_fdr_hr = int(((sdf["km_logrank_padj"] < 0.10) & (sdf["cox_HR"] > 1.0)).sum())
    n_mv_fdr = int((sdf["cox_multivar_padj"] < 0.10).sum())
    n_cohort = int(surv.get("n_km_cohort", sdf["n_km"].max()))
    n_cox = int(sdf["n_cox_complete_case"].median())
    filt_mode = surv.get("gene_filter_mode", "unknown")
    if filt_mode == "protein-coding":
        gene_phrase = "protein-coding genes"
    elif filt_mode == "annotated-symbol-only":
        gene_phrase = "annotated-symbol genes (gene biotype was not available, so protein-coding status was not verified)"
    else:
        gene_phrase = "upregulated genes (gene-biotype filtering status not recorded)"
    if "cox_penalizer" not in surv or surv.get("cox_penalizer") != 0.0:
        raise RuntimeError("Survival results were produced before the Cox penalizer fix; rerun survival_and_hubs.py.")

    def _mv_q(sym):
        r = sdf[sdf["symbol"] == sym]
        return None if r.empty or pd.isna(r["cox_multivar_padj"].iloc[0]) else float(r["cox_multivar_padj"].iloc[0])

    # Network degree / rank of the two selected kinases within the candidate subnetwork
    net_csv = "results/LUAD_final_docking_targets.csv"
    net_lbl = "Selected mitotic kinases (chosen a priori)"
    if os.path.exists(net_csv):
        ndf = pd.read_csv(net_csv).reset_index(drop=True)
        parts = []
        for g in ("NEK2", "PLK1"):
            hit = ndf.index[ndf["symbol"] == g]
            if len(hit):
                parts.append(f"{g}: degree {int(ndf.loc[hit[0], 'network_degree'])}, rank {int(hit[0]) + 1}/{len(ndf)}")
        if parts:
            net_lbl = "Selected mitotic kinases (chosen a priori); STRING degree within candidate subnetwork &mdash; " + "; ".join(parts)

    # Compute exact top ATP-competitive dS_Vina candidate (excluding non-ATP-competitive Rigosertib)
    if "best_scoring_comparator" not in docking_df.columns:
        raise RuntimeError("Vina CSV uses the old 'closest_offtarget' schema; rerun vina_selectivity_screen.py "
                           "(or recompute_selectivity_from_scores.py).")
    atp_df = docking_df[docking_df["atp_competitive"] == True].copy()
    best_row = atp_df.sort_values("delta_S_vina").iloc[0]
    best_ds_str = f"{best_row['delta_S_vina']:+.2f} (&Delta;S<sub>Vina</sub>)"
    best_ds_lbl = (f"Largest Vina score difference vs. pocket-aligned comparators "
                   f"({best_row['ligand']} on {best_row['primary_target']}; kcal/mol; all-comparator &Delta;S = {best_row['delta_S_vina_all_comparators']:+.2f}; unvalidated &mdash; within typical Vina error)")

    with open("results/LUAD_docking_pockets.json", "r") as f:
        pockets = json.load(f)

    # Fail loudly if inputs are out of sync (stale bundle / KM cohort)
    if os.path.getmtime("results/LUAD_web_dashboard_bundle.json") < os.path.getmtime(vina_csv):
        raise RuntimeError("LUAD_web_dashboard_bundle.json is older than the Vina CSV; rerun selectivity_viz_and_contacts.py.")
    _g = next(iter(km_data))
    km_n = km_data[_g]["high"]["n"] + km_data[_g]["low"]["n"]
    if km_n != n_cohort:
        raise RuntimeError(f"KM curves use n={km_n} but the survival screen cohort is n={n_cohort}; rerun plot_km_curves.py.")

    vina_version = pockets.get("vina_version", "version not recorded")
    thr = pockets.get("max_comparator_pocket_rmsd_A", 3.0)

    def _tlig(t):
        code = pockets["targets"][t].get("holo_ligand_resname")
        return f"PDB {pockets['targets'][t]['holo_pdb'].split('_')[-1].split('.')[0]}" + (f", co-crystal ligand {code}" if code else "")

    n_analogs = {t: sum(1 for k in pockets["alignments"][t] if not k.endswith("_CTRL")) for t in pockets["alignments"]}
    n_ctrls = {t: sum(1 for k in pockets["alignments"][t] if k.endswith("_CTRL")) for t in pockets["alignments"]}
    analog_val = f"{n_analogs.get('NEK2', 0)} / {n_analogs.get('PLK1', 0)} Analogs"
    analog_lbl = (f"Foldseek analogs docked (NEK2 / PLK1), each plus {n_ctrls.get('NEK2', 0)} clinical-target controls; "
                  f"comparators with pocket RMSD &gt; {thr} &Aring; are reported but excluded from &Delta;S")

    redock = pockets.get("positive_control_redocking", {})
    redock_rmsd = redock.get("heavy_atom_rmsd_A", "N/A")
    redock_score = redock.get("vina_score", "N/A")
    redock_passed = redock.get("passed", False)
    _n = f"{redock.get('matched_heavy_atoms', '?')}/{redock.get('ligand_heavy_atoms', '?')} heavy atoms matched"
    redock_summary = (
        f"PASSED ({redock_rmsd} &Aring; &le; 2.00 &Aring;; Vina score {redock_score}; {_n})"
        if redock_passed
        else f"Did not pass ({redock_rmsd} &Aring; &gt; 2.00 &Aring;; Vina score {redock_score}; {_n})"
    )
    redock_diag = redock.get("diagnosis", "Failure cause not diagnosed (rerun vina_selectivity_screen.py for the ring / non-ring RMSD breakdown).")
    redock_concl = (
        "Pose recovery met the 2.00 &Aring; benchmark, but Vina scores remain hypothesis-generating rankings rather than validated binding affinities (&Delta;G)."
        if redock_passed else
        "Because pose recovery did not meet the 2.00 &Aring; benchmark, all Vina scores are interpreted strictly as hypothesis-generating rankings rather than validated binding affinities (&Delta;G)."
    )
    _q_n, _q_p = _mv_q("NEK2"), _mv_q("PLK1")
    _fmt = lambda q: "n/a" if q is None else f"{q:.2f}"

    structures_3d = {
        "NEK2_receptor": pdbqt_to_pdb_block("docking/receptors/NEK2_primary.pdbqt"),
        "PLK1_receptor": pdbqt_to_pdb_block("docking/receptors/PLK1_primary.pdbqt"),
        "poses": {},
    }

    for row in docking_df.to_dict(orient="records"):
        key = f"{row['primary_target']}_{row['ligand']}"
        structures_3d["poses"][key] = pdbqt_to_pdb_block(row["pose_file"], first_model_only=True)

    site_payload = {
        "km_curves": km_data,
        "docking_results": docking_df.to_dict(orient="records"),
        "pocket_contacts": docking_bundle.get("pocket_contacts", {}),
        "foldseek_offtargets": foldseek_df.to_dict(orient="records"),
        "pockets": pockets,
        "structures_3d": structures_3d,
    }

    def clean_nans(obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nans(v) for v in obj]
        return obj

    embedded_json = json.dumps(clean_nans(site_payload), allow_nan=False)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TCGA-LUAD Target Prioritization &amp; Computational Counter-Screen (v1.2)</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --nek2: #f97316;
      --plk1: #a855f7;
      --green: #22c55e;
      --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border-bottom: 1px solid var(--border);
      padding: 2rem 1.5rem;
    }}
    .container {{ max-width: 1240px; margin: 0 auto; padding: 0 1.25rem; }}
    .badge {{
      display: inline-block;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.6rem;
    }}
    h1 {{ margin: 0 0 0.5rem; font-size: 1.85rem; }}
    .subtitle {{ color: var(--muted); max-width: 920px; margin: 0 0 1.2rem; font-size: 0.98rem; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.9rem;
      margin-top: 1rem;
    }}
    .metric-card {{
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.9rem 1rem;
    }}
    .metric-val {{ font-size: 1.35rem; font-weight: 700; color: var(--accent); }}
    .metric-lbl {{ font-size: 0.8rem; color: var(--muted); }}
    .section {{ margin: 2rem auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.5rem;
    }}
    .card h2 {{ margin-top: 0; font-size: 1.2rem; display: flex; align-items: center; gap: 0.5rem; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; align-items: center; }}
    select, button {{
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 0.85rem;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
    }}
    button.active, button:hover {{ border-color: var(--accent); color: var(--accent); }}
    #viewer3d {{
      width: 100%;
      height: 440px;
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #090d16;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ padding: 0.55rem 0.65rem; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pill-sel {{
      background: rgba(34, 197, 94, 0.16);
      color: var(--green);
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      font-weight: 600;
    }}
    .pill-non {{
      background: rgba(239, 68, 68, 0.16);
      color: #f87171;
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
    }}
    .pill-warn {{
      background: rgba(245, 158, 11, 0.16);
      color: var(--amber);
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      font-weight: 600;
    }}
    .contact-chip {{
      display: inline-block;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.25rem 0.55rem;
      margin: 0.2rem;
      font-size: 0.82rem;
    }}
    .disclaimer {{
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.5rem;
    }}
    footer {{
      border-top: 1px solid var(--border);
      padding: 1.5rem;
      text-align: center;
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 3rem;
    }}
  </style>
</head>
<body>
  <header>
    <div class="container">
      <div class="badge">Computational Oncology &amp; Structure-Guided Target Prioritization (v1.2)</div>
      <h1>Transcriptomic Target Prioritization &amp; Structure-Guided Computational Counter-Screen in TCGA-LUAD</h1>
      <p class="subtitle">
        Exploratory computational pipeline integrating <strong>PyDESeq2</strong> differential expression (log<sub>2</sub>FC &gt; 1.5, FDR &lt; 0.01),
        exploratory <strong>Kaplan–Meier / Cox</strong> survival stratification, <strong>STRING v12</strong> functional association network degree,
        <strong>Foldseek</strong> 3D structural analog search with local ATP-pocket superposition, and multi-seed <strong>AutoDock Vina (&Delta;S<sub>Vina</sub>)</strong> counter-screening.
      </p>
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-val">{n_fdr_hr} FDR / {n_nom_all} Nom.</div><div class="metric-lbl">Exploratory survival associations among {n_tested:,} upregulated genes (FDR: KM BH q &lt; 0.10 and HR &gt; 1; Nom.: KM p &lt; 0.05, either direction)</div></div>
        <div class="metric-card"><div class="metric-val">NEK2 &amp; PLK1</div><div class="metric-lbl">{net_lbl}</div></div>
        <div class="metric-card"><div class="metric-val">{analog_val}</div><div class="metric-lbl">{analog_lbl}</div></div>
        <div class="metric-card"><div class="metric-val">{best_ds_str}</div><div class="metric-lbl">{best_ds_lbl}</div></div>
      </div>
    </div>
  </header>

  <main class="container section">
    <div class="card">
      <h2>1. Interactive 3D ATP-Pocket Docking &amp; Off-Target Counter-Screen Viewer (AutoDock Vina {vina_version})</h2>
      <div class="controls">
        <label>Primary Target:
          <select id="targetSelect">
            <option value="NEK2">NEK2 ({_tlig("NEK2")})</option>
            <option value="PLK1">PLK1 ({_tlig("PLK1")})</option>
          </select>
        </label>
        <label>Cross-Screened Ligand:
          <select id="ligandSelect"></select>
        </label>
        <button id="spinBtn">Toggle Spin</button>
        <button id="resetCamBtn">Focus ATP Pocket</button>
      </div>
      <div class="grid-2">
        <div>
          <div id="viewer3d"></div>
          <p class="disclaimer">*Predicted rigid-receptor Vina poses represent exploratory computational hypotheses rather than experimentally resolved co-crystal structures.</p>
        </div>
        <div>
          <div id="ligandStatsCard" style="margin-bottom: 1rem;"></div>
          <h3 style="font-size: 1rem; margin-bottom: 0.4rem;">Nearby ATP-Pocket Residues (&lt; 3.8 &Aring; Proximity)</h3>
          <div id="contactsContainer"></div>
          <h3 style="font-size: 1rem; margin: 1rem 0 0.4rem;">Foldseek Structural Analogs &amp; Local ATP-Pocket RMSD</h3>
          <table id="offtargetTable">
            <thead><tr><th>Structural Analog</th><th>UniProt</th><th>Foldseek E-val</th><th>Seq ID</th><th>Pocket RMSD</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>2. Vina Docking Score (S<sub>primary</sub>) vs. Off-Target Margin (&Delta;S<sub>Vina</sub>)</h2>
        <div id="selectivityPlot" style="height: 390px;"></div>
      </div>
      <div class="card">
        <h2>3. TCGA-LUAD Overall Survival (Exploratory Median-Split Kaplan–Meier, n = {n_cohort})</h2>
        <div class="controls" id="kmGeneButtons"></div>
        <div id="kmPlot" style="height: 345px;"></div>
      </div>
    </div>

    <div class="card">
      <h2>4. Pocket-Aligned Cross-Screening Matrix (ATP-Competitive Panel &amp; Non-ATP Comparator)</h2>
      <div style="overflow-x: auto;">
        <table id="fullScreenTable">
          <thead>
            <tr>
              <th>Screen Target</th>
              <th>ATP Rank</th>
              <th>Ligand</th>
              <th>Mechanism / Clinical Status</th>
              <th>S<sub>primary</sub> (Vina)</th>
              <th>Best-Scoring Comparator</th>
              <th>S<sub>comp</sub> (Vina)</th>
              <th>&Delta;S<sub>Vina</sub> (aligned comparators)</th>
              <th>&Delta;S<sub>Vina</sub> (all comparators)</th>
              <th>MW</th>
              <th>cLogP</th>
              <th>TPSA (&Aring;&sup2;)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>5. Methodological Scope, Controls &amp; Scientific Limitations (v1.2)</h2>
      <ul style="color: #cbd5e1; font-size: 0.9rem; margin: 0; padding-left: 1.25rem;">
        <li><strong>Exploratory Transcriptomic &amp; Survival Screen:</strong> Across {n_tested:,} strongly upregulated {gene_phrase} (log<sub>2</sub>FC &gt; 1.5, DESeq2 FDR &lt; 0.01; cohort n = {n_cohort}), median-split Kaplan–Meier log-rank testing gave {n_nom_all} nominal associations (p &lt; 0.05, either direction; {n_nom_hr} with HR &gt; 1) and {n_fdr_all} at Benjamini–Hochberg FDR q &lt; 0.10 ({n_fdr_hr} with HR &gt; 1, the candidate set). Complete-case multivariable Cox regression (age, sex, pathological stage; median n = {n_cox}; unpenalized) was also run: only {n_mv_fdr} of {n_tested:,} genes have multivariable FDR q &lt; 0.10 (NEK2 q = {_fmt(_q_n)}, PLK1 q = {_fmt(_q_p)}). These are exploratory survival associations, not validated clinical biomarkers. The differential-expression model treats all tumor and normal samples as independent, including patients contributing both.</li>
        <li><strong>STRING v12 Functional Association Network:</strong> Network degree (number of STRING links to other genes in the candidate set; combined confidence score &ge; 0.70, encompassing physical and functional associations) quantifies connectivity within the candidate subnetwork only, not the whole STRING network, and is not direct physical binding strength or druggability. Betweenness centrality was not computed.</li>
        <li><strong>Positive Control Redocking Audit (PLK1 2RKU + (7R)-BI-2536):</strong> Co-crystal redocking with stereochemically verified (7R)-BI-2536 yielded a symmetry-aware heavy-atom RMSD of <strong>{redock_summary}</strong>. {redock_diag} {redock_concl}</li>
        <li><strong>&Delta;S<sub>Vina</sub> definition:</strong> S<sub>primary</sub> minus the best (most negative) Vina score among pocket-aligned comparators (ATP-pocket C&alpha; RMSD &le; {thr} &Aring; to the primary target); other comparators (including the clinical-target controls when poorly aligned) are docked and reported but excluded, so the table also shows &Delta;S against all comparators; the two can differ in sign. Margins of a few tenths of a kcal/mol are within typical Vina scoring error (seed SD reflects search noise only), so rankings are not evidence of selectivity.</li>
        <li><strong>Mechanism-Aware Comparator Separation:</strong> <strong>Rigosertib</strong> is flagged as a non-ATP-competitive modulator and excluded from the ATP-pocket selectivity ranking (displayed separately as an external comparator).</li>
      </ul>
    </div>
  </main>

  <footer>
    Developed by Arjun Ravi &bull; B.Tech. Industrial Biotechnology, Anna University &bull; Data: TCGA-LUAD, STRING v12, AlphaFold DB, Foldseek, RCSB PDB (2W5B, 2OWB, 2RKU, 1MQ4)
  </footer>

  <script>
    const DATA = {embedded_json};
    let glviewer = null;
    let spinning = false;

    function init() {{
      const ligSel = document.getElementById("ligandSelect");
      const uniqueLigs = [...new Set(DATA.docking_results.map(r => r.ligand))];
      uniqueLigs.forEach(l => {{
        const opt = document.createElement("option");
        opt.value = l;
        opt.textContent = l;
        ligSel.appendChild(opt);
      }});
      ligSel.value = "Alisertib";

      glviewer = $3Dmol.createViewer("viewer3d", {{ backgroundColor: "#090d16" }});

      document.getElementById("targetSelect").addEventListener("change", () => {{
        const tgt = document.getElementById("targetSelect").value;
        document.getElementById("ligandSelect").value = tgt === "NEK2" ? "Alisertib" : "BI-2536";
        update3DView();
      }});
      ligSel.addEventListener("change", update3DView);
      document.getElementById("spinBtn").addEventListener("click", () => {{
        spinning = !spinning;
        glviewer.spin(spinning ? "y" : false, 0.6);
      }});
      document.getElementById("resetCamBtn").addEventListener("click", () => {{
        glviewer.zoomTo({{ model: 1 }}, 600);
      }});

      update3DView();
      renderSelectivityQuadrant();
      initKMExplorer();
      renderFullTable();
    }}

    function update3DView() {{
      const tgt = document.getElementById("targetSelect").value;
      const lig = document.getElementById("ligandSelect").value;
      const recPDB = DATA.structures_3d[`${{tgt}}_receptor`];
      const posePDB = DATA.structures_3d.poses[`${{tgt}}_${{lig}}`];

      glviewer.clear();
      glviewer.addModel(recPDB, "pdb");
      glviewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: tgt === "PLK1" ? "#a855f7" : "#f97316", opacity: 0.85 }} }});

      if (posePDB) {{
        glviewer.addModel(posePDB, "pdb");
        glviewer.setStyle({{ model: 1 }}, {{ stick: {{ colorscheme: "greenCarbon", radius: 0.24 }} }});
        glviewer.setStyle(
          {{ model: 0, within: {{ distance: 4.5, sel: {{ model: 1 }} }} }},
          {{ cartoon: {{ color: "#38bdf8" }}, stick: {{ colorscheme: "whiteCarbon", radius: 0.12 }} }}
        );
      }}

      glviewer.zoomTo({{ model: 1 }});
      glviewer.zoom(0.75);
      glviewer.render();

      const row = DATA.docking_results.find(r => r.primary_target === tgt && r.ligand === lig);
      const isAtp = row.atp_competitive !== false;
      const pillClass = !isAtp ? "pill-warn" : (row.delta_S_vina <= -0.5 ? "pill-sel" : "pill-non");
      const pillLabel = !isAtp
        ? `Non-ATP Comparator (&Delta;S<sub>Vina</sub> = ${{row.delta_S_vina > 0 ? '+' : ''}}${{row.delta_S_vina.toFixed(2)}})`
        : `&Delta;S<sub>Vina</sub> = ${{row.delta_S_vina > 0 ? '+' : ''}}${{row.delta_S_vina.toFixed(2)}}`;

      document.getElementById("ligandStatsCard").innerHTML = `
        <div style="background:#0f172a; padding:0.85rem; border-radius:8px; border:1px solid #334155;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
            <strong style="font-size:1.05rem;">${{lig}} &rarr; ${{tgt}}</strong>
            <span class="${{pillClass}}">${{pillLabel}}</span>
          </div>
          <p style="margin:0.4rem 0 0; font-size:0.84rem; color:#94a3b8;">
            Target / Status: <strong style="color:#e2e8f0;">${{row.established_target || 'Kinase Inhibitor'}}</strong> &mdash; ${{row.clinical_status || ''}}
          </p>
          <p style="margin:0.35rem 0 0; font-size:0.86rem; color:#cbd5e1;">
            Primary Vina Score: <strong>${{row.score_primary_vina.toFixed(2)}}</strong> &bull;
            Best-Scoring Comparator (${{row.best_scoring_comparator}}): <strong>${{row.score_best_comparator_vina.toFixed(2)}}</strong><br/>
            MW: ${{row.MW}} Da &bull; cLogP: ${{row.cLogP}} &bull; TPSA: ${{row.TPSA}} &Aring;&sup2;
          </p>
        </div>
      `;

      const contacts = DATA.pocket_contacts[`${{tgt}}_${{lig}}`] || [];
      const cDiv = document.getElementById("contactsContainer");
      cDiv.innerHTML = contacts.length
        ? contacts.slice(0, 10).map(c => `<span class="contact-chip" title="${{c.contact_category}}">${{c.residue}} (${{c.min_dist_A}}&Aring;)</span>`).join("")
        : `<span style="color:#94a3b8; font-size:0.85rem;">Rotate &amp; zoom the 3D pocket viewer to inspect steric fit.</span>`;

      const offRows = DATA.foldseek_offtargets.filter(o => o.primary_target === tgt).slice(0, 3);
      document.querySelector("#offtargetTable tbody").innerHTML = offRows.map(o => `
        <tr>
          <td><strong>${{o.offtarget_symbol}}</strong></td>
          <td>${{o.offtarget_uniprot}}</td>
          <td>${{Number(o.evalue).toExponential(2)}}</td>
          <td>${{o.seq_id}}%</td>
          <td>${{o.atp_pocket_rmsd_A ? o.atp_pocket_rmsd_A + ' &Aring;' : 'N/A'}}</td>
        </tr>
      `).join("");
    }}

    function renderSelectivityQuadrant() {{
      const traces = ["NEK2", "PLK1"].map(tgt => {{
        const sub = DATA.docking_results.filter(r => r.primary_target === tgt && r.atp_competitive !== false);
        return {{
          x: sub.map(r => r.score_primary_vina),
          y: sub.map(r => r.delta_S_vina),
          text: sub.map(r => r.ligand),
          mode: "markers+text",
          textposition: "top center",
          name: `${{tgt}} ATP-Competitive Panel`,
          marker: {{ size: 11, color: tgt === "PLK1" ? "#a855f7" : "#f97316", line: {{ width: 1, color: "#fff" }} }},
          hovertemplate: "<b>%{{text}}</b> (" + tgt + ")<br>S_primary: %{{x:.2f}}<br>&Delta;S_Vina: %{{y:.2f}}<extra></extra>"
        }};
      }});

      Plotly.newPlot("selectivityPlot", traces, {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#0f172a",
        font: {{ color: "#e2e8f0" }},
        margin: {{ t: 20, r: 20, b: 48, l: 55 }},
        xaxis: {{ title: "Primary Vina Score S_primary (ATP-Competitive Panel)", gridcolor: "#1e293b" }},
        yaxis: {{ title: "Score Differential &Delta;S_Vina", gridcolor: "#1e293b" }},
        shapes: [
          {{ type: "line", x0: -10.5, x1: -7, y0: -0.5, y1: -0.5, line: {{ color: "#22c55e", dash: "dash", width: 1.5 }} }},
          {{ type: "line", x0: -8.5, x1: -8.5, y0: -1.5, y1: 1.7, line: {{ color: "#38bdf8", dash: "dot", width: 1.5 }} }}
        ]
      }}, {{ responsive: true }});
    }}

    function initKMExplorer() {{
      const genes = Object.keys(DATA.km_curves);
      const btnBox = document.getElementById("kmGeneButtons");
      genes.forEach((g, i) => {{
        const b = document.createElement("button");
        b.textContent = g;
        if (i === 0) b.classList.add("active");
        b.onclick = () => {{
          document.querySelectorAll("#kmGeneButtons button").forEach(x => x.classList.remove("active"));
          b.classList.add("active");
          plotKM(g);
        }};
        btnBox.appendChild(b);
      }});
      plotKM(genes[0]);
    }}

    function plotKM(symbol) {{
      const g = DATA.km_curves[symbol];
      Plotly.newPlot("kmPlot", [
        {{
          x: g.high.timeline_months,
          y: g.high.survival_prob,
          mode: "lines",
          line: {{ shape: "hv", color: "#ef4444", width: 2.5 }},
          name: `High ${{symbol}} (n=${{g.high.n}})`
        }},
        {{
          x: g.low.timeline_months,
          y: g.low.survival_prob,
          mode: "lines",
          line: {{ shape: "hv", color: "#38bdf8", width: 2.5 }},
          name: `Low ${{symbol}} (n=${{g.low.n}})`
        }}
      ], {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#0f172a",
        font: {{ color: "#e2e8f0" }},
        title: {{ text: `${{symbol}} Overall Survival (Nominal Log-Rank p = ${{g.logrank_p}})`, font: {{ size: 13 }} }},
        margin: {{ t: 40, r: 20, b: 45, l: 50 }},
        xaxis: {{ title: "Months", gridcolor: "#1e293b" }},
        yaxis: {{ title: "Survival Probability", range: [0, 1.05], gridcolor: "#1e293b" }},
        legend: {{ x: 0.03, y: 0.1 }}
      }}, {{ responsive: true }});
    }}

    function renderFullTable() {{
      const tbody = document.querySelector("#fullScreenTable tbody");
      tbody.innerHTML = DATA.docking_results.map(r => {{
        const isAtp = r.atp_competitive !== false;
        const rankStr = isAtp && r.atp_pocket_selectivity_rank ? `#${{Math.round(r.atp_pocket_selectivity_rank)}}` : "Excluded";
        const pillClass = !isAtp ? "pill-warn" : (r.delta_S_vina <= -0.5 ? "pill-sel" : "pill-non");
        return `
          <tr>
            <td><strong>${{r.primary_target}}</strong></td>
            <td>${{rankStr}}</td>
            <td><strong>${{r.ligand}}</strong></td>
            <td style="color:#94a3b8;">${{r.established_target}} &bull; ${{r.clinical_status}}</td>
            <td>${{r.score_primary_vina.toFixed(2)}}</td>
            <td>${{r.best_scoring_comparator}}</td>
            <td>${{r.score_best_comparator_vina.toFixed(2)}}</td>
            <td><span class="${{pillClass}}">${{r.delta_S_vina > 0 ? '+' : ''}}${{r.delta_S_vina.toFixed(2)}}</span></td>
            <td>${{r.delta_S_vina_all_comparators > 0 ? '+' : ''}}${{r.delta_S_vina_all_comparators.toFixed(2)}}</td>
            <td>${{r.MW}}</td>
            <td>${{r.cLogP}}</td>
            <td>${{r.TPSA}}</td>
          </tr>
        `;
      }}).join("");
    }}

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as hf:
        hf.write(html_code)

    print(f"[SUCCESS] v1.2 dashboard written to '{out_dir}/index.html' ({os.path.getsize(os.path.join(out_dir, 'index.html')) // 1024} KB)")


if __name__ == "__main__":
    build_static_site()