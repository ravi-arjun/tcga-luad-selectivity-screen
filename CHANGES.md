# Patch notes

Rerun order (the dashboard builder now refuses to build from stale or mismatched inputs):
1. `python src/deseq2_screen.py`  (cache hit: fetches gene biotype via MyGene if missing -> survival screen -> STRING)
2. `python src/vina_selectivity_screen.py`  (records Vina version, co-crystal ligand code, redocking ring/non-ring RMSD)
   - Or, to only recompute the comparator-filtered ΔS from the existing CSV: `python src/recompute_selectivity_from_scores.py`
3. `python src/selectivity_viz_and_contacts.py`
4. `python src/plot_km_curves.py`  (asserts its log-rank p-values equal the screen's)
5. `python src/build_web_app.py`

| File | Change |
|---|---|
| deseq2_screen.py | "nan" string bug in sample typing fixed; MyGene `type_of_gene` stored as `gene_type`; `add_gene_type()` backfills cached tables; optional paired (patient + condition) sensitivity DE written to `LUAD_DESeq2_paired_sensitivity.csv` (untested) |
| survival_and_hubs.py | strict stage regex; primary-tumor test uses barcode sample code (not substring "-01"); protein-coding filter (or annotated-symbol fallback, recorded); Ensembl-ID placeholders excluded; Cox penalizer 0.1 -> 0.0; writes `LUAD_survival_cohort.csv` and `LUAD_survival_summary.json` (all counts, any-direction and HR>1) |
| plot_km_curves.py | rewritten to use the screen's exact cohort; title n computed; p-value asserted against screen CSV; no `plt.show()` |
| vina_selectivity_screen.py | comparator ΔS uses only pocket-aligned comparators (<= 3.0 Å), all-comparator ΔS also kept; columns renamed `best_scoring_comparator` / `score_best_comparator_vina`; redocking ring vs non-ring RMSD + auto diagnosis + atoms matched/total; Vina version and holo ligand code recorded; dated/unverified trial statuses replaced with coarse ones |
| selectivity_viz_and_contacts.py | legend/axis text corrected; bundle stamped with source CSV mtime |
| build_web_app.py | every figure/claim computed from files (counts, "protein-coding" wording, NEK2/PLK1 degree+rank, analog counts, redocking text, Vina version, ΔS definition, Cox FDR caveat); no betweenness claim; "Closest" -> "Best-Scoring Comparator"; raises if bundle/KM cohort/Cox results are stale |
| recompute_selectivity_from_scores.py | new helper (see above) |
