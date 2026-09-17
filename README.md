# Public analysis code

The repository contains seven analysis files:

```text
.
├── isrsa/
│   ├── isrsa_analysis_cells.ipynb
│   └── isrsa_functions.py
├── slidingWindow_isc/
│   ├── slidingWindow_isc_analysis_cells.ipynb
│   └── slidingWindow_isc_functions.py
├── AllFive_group_median_meta_analysis_ALL_AFFECTS.py
├── ISFC_DMN_ALLTR_ALLFIVE_CHORD_CSHIFTPERM100000.py
└── Study2_behavioral_heterogeneity_paired_bootstrap.ipynb
```

## What is kept local

Do not commit files that contain participant-level identifiers or study-specific membership information. In particular, keep the following outside the repository (or under a gitignored `local_config/` directory):

- participant matching IDs used for bootstrap/resampling dependencies;
- dataset-specific participant inclusion lists;
- optional participant exclusion lists;
- the mixed-model dependence table described below;
- raw or derived participant-level data.

The public code accepts arbitrary pseudonymous IDs. The actual labels do not matter; only consistency of the matching relationships matters.

## Participant-matching configuration

Analyses that need participant-resampling dependencies read a local JSON file pointed to by:

```bash
export VISC_PARTICIPANT_CONFIG_PATH=/secure/path/participant_matching_config.json
```

A generic example is:

```json
{
  "study1": {
    "mode": "independent_modalities",
    "neural_ids": {
      "dataset_A": ["neural_001", "neural_002"],
      "dataset_B": ["neural_001", "neural_002"]
    },
    "behavior_ids": {
      "dataset_A": {
        "affect_1": ["rater_001", "rater_002"],
        "affect_2": ["rater_001", "rater_002"]
      }
    }
  },
  "study2": {
    "mode": "matched_modalities",
    "common_ids": ["participant_001", "participant_002"]
  }
}
```

The keys used in a local configuration must match the dataset keys expected by the corresponding script. The values shown above are **illustrative placeholders only**.

For IS-RSA, optional exclusions can also be supplied locally:

```bash
export VISC_EXCLUDED_PARTICIPANTS_PATH=/secure/path/exclusions.json
```

with, for example:

```json
{
  "excluded_participant_ids": ["participant_x", "participant_y"]
}
```

No exclusion IDs should be written into the public source code.

## Stage-2 mixed-model design

The primary sliding-window analysis uses a two-stage mixed-effects model. The public repository does not contain the study-specific information needed to define the following three model components:

1. **crossed-participant random intercepts**;
2. **repeated-dyad random intercepts**;
3. **stimulus/dataset fixed effects**.

Before Stage 2 is run, provide those relationships in a local CSV file and point the code to it:

```bash
export VISC_MIXED_MODEL_DESIGN_PATH=/secure/path/mixed_model_design.csv
```

The file must contain one row for each Stage-1 dyad coefficient and these columns:

```text
dataset_key
local_subject_i
local_subject_j
crossed_participant_i
crossed_participant_j
repeated_dyad_id
dataset_fixed_level
```

Example with fake labels:

```csv
dataset_key,local_subject_i,local_subject_j,crossed_participant_i,crossed_participant_j,repeated_dyad_id,dataset_fixed_level
dataset_A,local_A01,local_A02,person_001,person_002,dyad_001,level_A
dataset_B,local_B11,local_B12,person_001,person_002,dyad_001,level_B
dataset_A,local_A01,local_A03,person_001,person_003,dyad_002,level_A
```

Interpretation:

- `crossed_participant_i` and `crossed_participant_j` define the two participant random-effect identities for each dyad;
- `repeated_dyad_id` must be reused whenever the same dyad is observed again in another dataset/condition;
- `dataset_fixed_level` identifies the stimulus/dataset fixed-effect level;
- `local_subject_i` and `local_subject_j` are only join keys that connect the local design table to the local Stage-1 dyad output.

The same anonymous ID should be reused whenever the same random-effect unit recurs. The public code validates that every Stage-1 dyad is covered by the local design table but does not distribute the values in that table.

## Expected data structures

### Sliding-window ISC / group-median analysis

A typical local input organization contains:

- **neural time series:** one file per participant and dataset; rows are time points and columns are ROI time series;
- **behavioral ratings:** rows are participants/raters and columns are time points, with participant labels either in an ID column or supplied by the matching configuration;
- **window-level covariates:** one row per analysis window, with the audiovisual nuisance variables and time terms required by the analysis;
- **atlas metadata:** ROI metadata used to define the network/region groupings.

The scripts use `VISC_DATA_DIR` and `VISC_RESULTS_DIR` as the main path overrides. Additional dataset-specific path environment variables can be used where defined in the code.

### IS-RSA

Before running `isrsa_analysis_cells.ipynb`, load the required local objects into the notebook namespace. Structurally:

- neural arrays: `participant × time × ROI`;
- participant-label vectors: one label per neural-array participant;
- behavioral similarity matrices: square `participant × participant` matrices with labels that can be aligned to the neural participants;
- atlas table: ROI metadata containing the columns used by the regional analysis.

The public notebook does not contain participant labels or a fixed sample size.

### Whole-time-series ISFC

Each dataset is loaded as participant-level ROI time series. Dataset-specific inclusion lists are supplied through the local JSON configuration (`isfc.dataset_ids`). The script then computes the all-time-series ISFC estimates and circular-shift permutation inference.

### Behavioral heterogeneity notebook

The behavioral heterogeneity notebook expects one matrix for each session × affect combination:

```text
rows    = participants
columns = time points
```

CSV, TSV, XLSX, and NPY inputs are supported. If tabular files include an ID column, set `ID_COLUMN` in the notebook. For paired resampling, the local data must be aligned so that the same resampling unit is used across the paired sessions.

## Suggested local layout

The local configuration files and data can live outside the repository entirely. One possible local arrangement is:

```text
project/
├── code/                         # this public repository
├── local_config/                 # never commit
│   ├── participant_matching_config.json
│   ├── mixed_model_design.csv
│   └── exclusions.json
├── data/                         # never commit restricted data
└── results/                      # generated outputs
```

Example shell setup:

```bash
export VISC_DATA_DIR=/path/to/project/data
export VISC_RESULTS_DIR=/path/to/project/results
export VISC_PARTICIPANT_CONFIG_PATH=/path/to/project/local_config/participant_matching_config.json
export VISC_MIXED_MODEL_DESIGN_PATH=/path/to/project/local_config/mixed_model_design.csv
export VISC_EXCLUDED_PARTICIPANTS_PATH=/path/to/project/local_config/exclusions.json  # optional
```

## Running the analyses

- `slidingWindow_isc/slidingWindow_isc_analysis_cells.ipynb`: primary sliding-window temporal-coupling workflow.
- `isrsa/isrsa_analysis_cells.ipynb`: IS-RSA and planned control-seed analyses.
- `AllFive_group_median_meta_analysis_ALL_AFFECTS.py`: complementary group-median temporal-coupling analysis; self-contained and requires no sibling analysis scripts.
- `ISFC_DMN_ALLTR_ALLFIVE_CHORD_CSHIFTPERM100000.py`: whole-time-series within-network ISFC with circular-shift permutation inference.
- `Study2_behavioral_heterogeneity_paired_bootstrap.ipynb`: paired bootstrap comparison of cross-participant behavioral heterogeneity.

## Environment

The analyses were written for Python 3.11. Main non-standard packages used by the public code include NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib, nltools, HoloViews, Bokeh, and openpyxl/Jupyter for notebook and spreadsheet support.

## Acknowledgements and methodological references

### Naturalistic Data Analysis tutorials

Parts of the **sliding-window ISC** and **IS-RSA** code were developed with substantial reference to the [Naturalistic Data Analysis tutorials](https://naturalistic-data.org/content/intro.html). We gratefully acknowledge the authors and contributors of that resource.

### ISC

Methodological reference for the inter-subject correlation mixed-effects analysis:

> Chen, G., Taylor, P. A., Shin, Y. W., Reynolds, R. C., & Cox, R. W. Untangling the relatedness among correlations, Part II: Inter-subject correlation group analysis through linear mixed-effects modeling. NeuroImage 147, 825-840 (2017).

### IS-RSA

Methodological reference for the inter-subject representational similarity analysis:

> Finn, E. S., Glerean, E., Khojandi, A. Y., Nielson, D., Molfese, P. J., Handwerker, D. A., & Bandettini, P. A. Idiosynchrony: From shared responses to individual differences during naturalistic neuroimaging. NeuroImage 215, 116828 (2020).

### ISFC

Methodological reference for the leave-one-out inter-subject functional connectivity analysis:

> Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. Dynamic reconfiguration of the default mode network during narrative comprehension. Nature Communications 7, 12141 (2016).

