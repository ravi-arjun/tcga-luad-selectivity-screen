# Structure-Guided Target & Selective Ligand Discovery in Lung Adenocarcinoma (TCGA-LUAD)

**Live Interactive 3D Dashboard:** https://<new-username>.github.io/tcga-luad-selectivity-screen/

An end-to-end multi-omics and structure-based counter-screening pipeline identifying prognostic mitotic kinase drivers in Lung Adenocarcinoma (TCGA-LUAD) and ranking clinical kinase inhibitors by thermodynamic selectivity ($\Delta\Delta G$) against 3D structural off-target kinases.

## Pipeline Overview
1. **Differential Expression (`PyDESeq2`):** Transcriptomic profiling across 598 TCGA-LUAD samples (539 Primary Tumor vs. 59 Solid Tissue Normal; $\text{FDR} < 0.01$, $\log_2\text{FC} > 1.5$).
2. **Survival & Network Prioritization (`lifelines` + `STRING`):** Univariate Cox proportional hazards ($\text{Cox } p < 0.05$) and Kaplan–Meier stratification narrowing hits to 42 prognostic drivers, isolating central mitotic kinase hubs **NEK2** and **PLK1**.
3. **3D Proteome Off-Target Mapping (`Foldseek` API):** Structural alignment against the human AlphaFold proteome to identify structurally homologous ATP-pocket off-targets (`NEK4`, `NEK7`, `NEK8`, `RPS6KB1`, `MARK2`, `PHKG2`).
4. **Selectivity Counter-Screening (`RDKit`, `Meeko`, `AutoDock Vina 1.2.7`):** 80-run docking screen evaluating primary affinity ($\Delta G$) vs. off-target kinase binding ($\Delta\Delta G = \Delta G_{\text{primary}} - \Delta G_{\text{off-target}}$), identifying **Alisertib** ($\Delta\Delta G = -1.90\text{ kcal/mol}$ on `PLK1`) and **BI-2536** ($\Delta\Delta G = -0.85\text{ kcal/mol}$ on `NEK2`) as selective leads.
5. **Interactive Web Application (`3Dmol.js`, `Plotly.js`):** Static single-page dashboard for 3D ATP-pocket pose inspection, residue contact profiling ($<3.8\text{ \AA}$), and survival exploration.
