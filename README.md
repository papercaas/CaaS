# Market structure and law-enforcement disruption in the RaaS economy

Replication code and aggregated results for the article "From Oligopoly to Fragmentation: Market Structure and Law-Enforcement Disruption in the Ransomware-as-a-Service Economy".

## Data

- Leak-site claims from the public ransomware.live API v2 (`https://api.ransomware.live/v2/victims/YYYY/MM`), January 2022 – September 2026, downloaded on 24 October 2026 (30,085 raw records; 29,162 claims by 287 groups after cleaning).
- Second tracker: `posts.json` from the ransomwatch repository (github.com/joshhighet/ransomwatch), used for 2022–2024.

Victim-level data are not redistributed here, to avoid republishing victim names. They can be re-downloaded with the notebook, or obtained from the authors on reasonable request (contact details withheld for double-anonymous review).

## Reproduce

```bash
pip install -r code/requirements.txt
cd code
jupyter nbconvert --to notebook --execute caas_leaksite_eda.ipynb   # download (cached), cleaning, main figures and tables
python paper_stats.py                                                 # statistics cited in the text
python robustness_step2.py                                            # CR_k, rebrand lineages, survival sensitivity, seven-operation event study
curl -L -o data/rw_posts.json https://raw.githubusercontent.com/joshhighet/ransomwatch/main/posts.json
python compare_ransomwatch.py data/rw_posts.json data/victims_clean.csv.gz tables/coverage_ransomwatch_quarterly.csv 2025-01-01
```

## Contents

- `code/`: analysis notebook and scripts
- `tables/`: aggregated results (group-level and quarterly), including `paper_stats.json` and `robustness_step2.json`
- `figures/`: figures of the article (PDF)

## Licence

Code: MIT (see `LICENSE`). Aggregated tables and figures: CC BY 4.0.
