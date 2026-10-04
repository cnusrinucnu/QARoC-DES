# QARoC-DES: code and results for peer review
For peer review only. Not licensed for reuse until publication.

## Naming (code -> paper)
DES_MHA = published DES-bADE (first execution); bADE-RoC-DES and ASyDES = QARoC-DES.

## Setup
Python 3.11. `PYTHONNOUSERSITE=1 pip install -r requirements.txt`

## Reproduce the reported statistics and figures (no heavy data needed)
| Paper item | Command | Output |
|---|---|---|
| Tables 5, 6 | python generate_primary_ieee_statistics.py | publication_results/IEEE_Transactions_Analysis/ |
| Fig. 3 | python generate_cd_diagram.py | .../IEEE_Transactions_Analysis/figures/ |
| Tables 7, 8 | python analyze_topk_ablation.py | publication_results/topk_ablation/statistical_analysis/ |
| Tables 9-11 | python ctrl_analysis.py | publication_results/controls/ |
| Section 5.8 | analyze_query_difficulty_roc.py, analyze_fitness_quantiles.py | .../topk_ablation/query_difficulty_analysis/ |
| Figs. 2, 4, 5, 6 | python make_paper_figures.py | publication_results/paper_figures/ |
| Baseline means (Tables 13-14) | python rebuild_baselines.py | baseline_rebuild_check.csv |

Inputs: publication_results/topk_ablation/topk_ablation_run_level.csv (QARoC-DES,
Top-3 rows = main configuration), controls/control_runs.csv,
All_Methods_30_Datasets_Mean.csv (per-dataset means of the other methods, rebuilt and
checked against Experiment/Results/ by rebuild_baselines.py).

## Full rerun (needs Experiment/Datasets and Experiment/Pools)
- Splits and pools: helpers.py
- QARoC-DES with Top-K: topk_ablation_bade_roc.py
- Published DES-bADE (public implementation, installed separately): run_published_desbade.py
- Controls: ctrl_features.py
- DESlib baselines (k=7): helpers.py (cached fitted models are not distributed)
- Development only: step13b_frozen_k3_20iterations_fixed.py (Laryngeal3 check of K_E, Section 6)

## Notes
The ADE search is unseeded, so a rerun gives slightly different QARoC-DES values.
The per-query optimization takes about 1.2 s per query.
