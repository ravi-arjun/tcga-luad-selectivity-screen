# TCGA-LUAD: Transcriptomic Target Prioritization and Structure-Guided Computational Screening (v1.1)

**Live Interactive Dashboard:** [https://ravi-arjun.github.io/tcga-luad-selectivity-screen/](https://ravi-arjun.github.io/tcga-luad-selectivity-screen/)

An exploratory computational oncology and structural bioinformatics workflow for Lung Adenocarcinoma (**TCGA-LUAD**), integrating bulk RNA-seq differential expression, clinical survival stratification, functional network prioritization, 3D structural analog search, and ATP-pocket-aligned molecular docking counter-screening.

---

## Pipeline Architecture & Fact-Checked Results (v1.1)

1. **Differential Gene Expression (`PyDESeq2`):**
   - Analyzed **539 TCGA-LUAD primary tumors** vs. **59 solid tissue normal** samples using unstranded STAR integer gene counts.
   - Applied negative-binomial modeling (`~ condition`) with candidate selection thresholds of $\log_2\text{FC} > 1.5$ and Benjamini–Hochberg $\text{padj} < 0.01$.
2. **Survival Stratification (`src/survival_and_hubs.py`):**
   - Screened **149** overexpressed protein-coding genes across deduplicated TCGA-LUAD patients (`cases.submitter_id`) using pre-specified median-split Kaplan–Meier log-rank tests and clinical-covariate Cox proportional hazards regression (age, sex, AJCC stage) with Schoenfeld residual diagnostics.
   - **42 candidates** met nominal survival thresholds ($\text{KM } p < 0.05$ or $\text{Cox } p < 0.05$, $\text{HR} > 1.0$); **24 candidates** remained significant after Benjamini–Hochberg FDR correction on log-rank p-values ($q < 0.10$, $\text{HR} > 1.0$).
3. **Functional Network Prioritization (`STRING v12.0`):**
   - Queried the human STRING functional association network (confidence score $\ge 400$).
   - Prioritized mitotic serine/threonine kinases **NEK2** (`degree = 23`, `betweenness = 0.0155`) and **PLK1** (`degree = 22`, `betweenness = 0.0140`) based on combined network connectivity and catalytic kinase druggability.
4. **3D Structural Analog Search & ATP-Pocket Validation (`src/foldseek_offtargets.py`):**
   - Queried AlphaFold structures of **NEK2 (`P51955`)** and **PLK1 (`P53350`)** against the human AlphaFold proteome via the Foldseek API (`3diaa`).
   - Evaluated local ATP-binding pocket C$\alpha$ RMSD within $10\text{ \AA}$ of co-crystallized ligands:
     - **NEK2 analogs:** `NEK4` (`P51957`, pocket RMSD `1.732 Å`), `NEK7` (`Q8TDX7`, `2.212 Å`), `NEK8` (`Q86SG6`, `2.391 Å`).
     - **PLK1 analogs:** `PHKG2` (`P15735`, pocket RMSD `0.884 Å`), `RPS6KB1` (`P23443`, `1.615 Å`). `MARK2` (`Q7KZI7`, pocket RMSD `7.237 Å`) was excluded from ATP-pocket counter-screening due to divergent pocket geometry.
5. **ATP-Pocket-Aligned Molecular Docking Counter-Screen (`src/vina_selectivity_screen.py`):**
   - **Positive Control:** Co-crystal redocking of **BI-2536** into human PLK1 (**PDB `2RKU`**, ligand `R78`) to verify pose recovery.
   - **Receptor & Grid Setup:** Receptors prepared with polar hydrogens (`HD`) and chemically differentiated AutoDock4 donor/acceptor types (`N`, `NA`, `OA`, `SA`). Off-target structures were superposed strictly onto the primary ATP-binding pocket (`0.000 Å` grid centroid shift) alongside mandatory established clinical target controls (**PLK1 `2OWB`** in the NEK2 screen; **AURKA `1MQ4`** in the PLK1 screen).
   - **Cross-Screening Summary (AutoDock Vina 1.2.7, 3 seeds, exhaustiveness = 16):**
     - **NEK2 (`2W5B`, ATP$\gamma$S-bound):** Largest negative score differentials ($\Delta S_{\text{Vina}} = S_{\text{primary}} - S_{\text{best off-target}}$) observed for **Alisertib** ($S_{\text{primary}} = -10.093$, $\Delta S_{\text{Vina}} = -0.758$ vs. `NEK4`), **Volasertib** ($-8.790$, $\Delta S_{\text{Vina}} = -0.596$ vs. `NEK7`), and **BI-2536** ($-9.606$, $\Delta S_{\text{Vina}} = -0.504$ vs. `PLK1_CTRL`).
     - **PLK1 (`2OWB`, PHA-680626-bound):** Largest negative score differential observed for **Rigosertib** ($S_{\text{primary}} = -8.451$, $\Delta S_{\text{Vina}} = -0.812$ vs. `AURKA_CTRL`), followed by **Alisertib** ($-9.055$, $\Delta S_{\text{Vina}} = -0.288$ vs. `PHKG2`).

---

## Methodological Scope & Scientific Qualifications

- **Hypothesis-Generating Computational Screen:** All differential expression, survival associations, structural alignments, and Vina docking scores represent *in silico* prioritization hypotheses rather than experimentally validated biochemical selectivity or clinical efficacy.
- **Vina Scoring Function:** $S_{\text{primary}}$ and $\Delta S_{\text{Vina}}$ are empirical AutoDock Vina scoring differentials (averaged across 3 random seeds), not experimental binding free energies ($\Delta G$).
- **Cohort & Structural Caveats:** Survival candidates were derived and tested within the same TCGA-LUAD cohort and require independent external cohort validation (planned for v2.0 alongside DepMap CRISPR essentiality and CPTAC proteomics integration). Rigid-receptor docking does not model induced-fit kinase loop rearrangements (e.g., DFG-in vs. DFG-out states).

---

## Reproducing the Structural & Web Pipeline

```bash
pip install numpy pandas scipy statsmodels lifelines networkx requests biopython rdkit meeko matplotlib
python src/foldseek_offtargets.py
python src/vina_selectivity_screen.py
python src/selectivity_viz_and_contacts.py
python src/build_web_app.py
