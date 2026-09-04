# GR00T-H Surgical Language Analysis

This repository reconstructs and analyzes the language mixture used by the
Surgical portion of the GR00T-H N1.7 training recipe.

[Open the generated analysis report](extracted_language/visualizations/index.html)

## Pipeline

The code is split into four stages:

1. `extract_gr00t_h_n17_training_language.py` maps the 57 training entries to
   the current public Surgical datasets and produces normalized training-mixture
   weights.
2. `language_normalization.py` contains the auditable, mechanism-specific rules
   that convert VLA input into clean task text and normalized task text.
3. `analyse_language.py` performs mixture-weighted NLTK analysis without loading
   or expanding the full frame-level corpus.
4. `visualize_language_analysis.py` creates the standalone SVG figures and the
   interactive HTML report.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The extractor expects these local directories, both excluded by `.gitignore`:

```text
open_h_subset/Surgical/   # local Open-H Surgical datasets
gr00t_h/                  # NVIDIA-Medtech/GR00T-H checkout
```

Clone GR00T-H at the expected path:

```bash
git clone https://github.com/NVIDIA-Medtech/GR00T-H.git gr00t_h
```

## Run

```bash
python extract_gr00t_h_n17_training_language.py
python analyse_language.py
python visualize_language_analysis.py
```

The extractor reuses `extracted_language/gr00t_h_n17_public_language_counts.csv`
as a compact cache. Pass `--rescan` only when the local source data changed:

```bash
python extract_gr00t_h_n17_training_language.py --rescan
```

NLTK resources are downloaded automatically on the first analysis run when
they are not already installed.

## Interpretation

The report distinguishes VLA input (the model-facing language) from clean task
text and normalized task text used for analysis. Training-group ratios come
from the N1.7 recipe; within-group language proportions are reconstructed from
the mapped public rows. Missing historical snapshots and configured split
exclusions prevent the result from being an exact replay of optimizer sampling.

## GitHub Pages

After pushing the repository, open **Settings → Pages**, choose **Deploy from a
branch**, and publish the repository root from the default branch. The root
`index.html` redirects to the generated interactive report.
