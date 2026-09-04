# GR00T-H Surgical Language Analysis

Training-mixture-weighted analysis of the language used by the Surgical portion
of the GR00T-H N1.7 training recipe.

**[Open the interactive report](https://janusmaple.github.io/gr00t-h-surgical-language-analysis/extracted_language/visualizations/)**

The pipeline reconstructs VLA input from public Open-H data, removes audited
template artifacts, normalizes the task text, and runs weighted NLTK analysis.

## Setup

Python 3.10+ is recommended. Allow at least **32 GB** of storage; **35–40 GB of
free space** provides safer download headroom.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the required Surgical Parquet and metadata files, then clone GR00T-H:

```bash
hf download nvidia/PhysicalAI-Robotics-Open-H-Embodiment \
  --repo-type dataset \
  --include "Surgical/**/data/**/*.parquet" \
  --include "Surgical/**/meta/**" \
  --include "Surgical/**/*.md" \
  --local-dir open_h_subset

git clone https://github.com/NVIDIA-Medtech/GR00T-H.git gr00t_h
```

The download excludes videos. In the tested checkout, the selected Open-H files
occupy about 30 GB and GR00T-H occupies about 80 MB. Both input directories are
excluded from Git by `.gitignore`.

## Run

```bash
python extract_gr00t_h_n17_training_language.py
python analyse_language.py
python visualize_language_analysis.py
```

Use `--rescan` with the extraction script only after changing the local source
data. NLTK downloads missing language resources automatically on first use.

## Interpretation

The report distinguishes **VLA input**, **clean task text**, and **normalized
task text**. Training-group weights come from the N1.7 recipe, while within-group
language proportions are reconstructed from mapped public rows. Missing
historical snapshots and split exclusions mean this is not an exact replay of
optimizer sampling.
