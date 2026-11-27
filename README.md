# Transcriptomic Target Prioritization & Structure-Guided Computational Counter-Screen in TCGA-LUAD (v1.2)

An exploratory computational oncology workflow integrating bulk RNA-seq differential expression (`PyDESeq2`), survival stratification (`lifelines`), functional association network analysis (`STRING v12`), 3D structural analog identification (`Foldseek`), and multi-seed rigid-receptor counter-screening (`AutoDock Vina 1.2.7`).

---

## Pipeline Overview & Key Parameters

1. **Stage 1 — Differential Expression (`src/deseq2_screen.py`):**
   * **Cohort:** TCGA-LUAD GDC STAR unstranded counts (`cart 2026-01-17`), restricted explicitly to `Primary Tumor` vs. `Solid Tissue Normal` samples.
   * **Filtering & Model:** Low-expression CPM filter (`CPM > 1` in $\ge 10$ samples), `PyDESeq2` negative binomial GLM (`~ condition`, reference = `normal`), Benjamini–Hochberg FDR correction.
   * **Selection Cutoff:** Strongly upregulated protein-coding candidates ($\log_2\text{FC} > 1.5$, $\text{padj} < 0.01$; $n = 3,492$ valid genes).
2. **Stage 2 — Exploratory Survival & Network Prioritization (`src/survival_and_hubs.py`):**
   * **Survival Screen:** Pre-specified median-split Kaplan–Meier log-rank test ($619$ nominal $p < 0.05$; $183$ survive Benjamini–Hochberg FDR $q < 0.10$ with $\text{HR} > 1.0$) paired with complete-case multivariable Cox Proportional Hazards regression (`expr_z` adjusted for age, sex, and AJCC pathological stage) and Schoenfeld proportional-hazards diagnostics.
   * **Functional Network:** `STRING v12` human functional association network (`species = 9606`, combined confidence score $\ge 0.70$), prioritizing connected mitotic kinase candidates **`NEK2`** and **`PLK1`**.
3. **Stage 3 — 3D Structural Analog Search (`src/foldseek_offtargets.py`):**
   * `Foldseek` (`3Di` structural alphabet) search against the human AlphaFold proteome, followed by local ATP-pocket $\text{C}\alpha$ superposition (residues within $10.0\text{ \AA}$ of the co-crystallized ligand; pocket RMSD $< 3.0\text{ \AA}$).
4. **Stage 4 — Structure-Guided Counter-Screening (`src/vina_selectivity_screen.py`):**
   * **Receptors:** `NEK2` (`PDB 2W5B`, ATP$\gamma$S-bound) and `PLK1` (`PDB 2OWB`, PHA-680626-bound), plus Foldseek structural analogs (`NEK4`, `NEK7`, `NEK8`, `PHKG2`, `RPS6KB1`) and clinical target comparators (`AURKA 1MQ4`, `PLK1 2OWB`, `NEK2 2W5B`). Waters/ions stripped; polar hydrogens (`HD`) and AutoDock4 donor/acceptor types assigned.
   * **Ligands & Seeds:** Stereochemically verified isomeric SMILES (including `(7R)`-BI-2536, `(7R)`-Volasertib, and `(3R,4R)`-Tofacitinib) embedded via `RDKit ETKDGv3`, minimized with `MMFF94`, prepared with `Meeko`, and docked across 3 seeds (`42, 101, 2026`; box $20 \times 20 \times 20\text{ \AA}$, `exhaustiveness = 16`).
   * **Mechanism Separation:** Non-ATP-competitive **Rigosertib** is excluded from the ATP-pocket selectivity ranking and reported as an external comparator.
5. **Stage 5 — Interactive Web Dashboard (`src/build_web_app.py` -> `docs/index.html`):**
   * Self-contained Plotly + `3Dmol.js` static dashboard synchronized directly with exported CSV/JSON outputs.

---

## Scientific Qualifications & Reproducibility Requirements

* **Exploratory Nature:** Survival associations, STRING functional network degrees, and $\Delta S_{\text{Vina}} = S_{\text{primary}} - S_{\text{best comparator}}$ score differentials represent hypothesis-generating computational rankings rather than validated clinical biomarkers or experimental binding free energies ($\Delta G$).
* **Positive Control Redocking Audit:** Redocking `(7R)`-BI-2536 into co-crystallized `PLK1` (`PDB 2RKU`, ligand `R78`) yielded a symmetry-aware heavy-atom RMSD of `5.556 Å` (above the `2.00 Å` pose-recovery benchmark due to flexible tail movement in a rigid receptor), which is documented transparently in `results/LUAD_docking_pockets.json` and on the dashboard.
* **Environment & Large Raw Inputs:**
  * Python 3.11+ with `numpy`, `pandas`, `scipy`, `statsmodels`, `pydeseq2`, `lifelines`, `mygene`, `biopython`, `rdkit`, `meeko`, `matplotlib`, `seaborn`, `requests`, and `AutoDock Vina 1.2.7` (`bin/vina.exe`).
  * Raw GDC TCGA-LUAD STAR count TSVs (`data/raw/`), GDC clinical/biospecimen TSVs (`data/clinical.cart.*/`, `data/biospecimen.cart.*/`), and the Vina executable are excluded from version control due to size/licensing; obtain TCGA-LUAD controlled/open-access manifest files from the NCI GDC Data Portal to reproduce Stage 1 from scratch.