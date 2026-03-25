# Antibody Data Sources: What Actually Works

## TL;DR

- **Positives**: TheraSAbDab (~1100 therapeutic VH sequences, free CSV)
- **Negatives**: OAS (~2.4B sequences, human VH bulk download via wget)
- **What to avoid**: Synthetic mutations (current code), IMGT (license), abYsis (limited)

## The Problem

The current `prepare.py` generates negative examples by mutating therapeutic sequences at 15%. This is scientifically wrong:

1. Mutated therapeutics ≠ natural antibody diversity
2. Model learns "broken therapeutic" vs "good therapeutic" — not the actual task
3. No published work uses this approach

Raybould et al. 2019 PNAS ("Predicting Antibody Developability from Sequence") used TheraSAbDab + OAS. We should too.

## Data Sources Evaluated

### TheraSAbDab — Winner (Positives)

**URL**: `https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab`

| What | Details |
|------|---------|
| Content | ~1100 therapeutic antibodies (approved + clinical) |
| Format | CSV with VH/VL sequences, target, clinical status |
| Access | Free download, no login |
| Column | `HeavySequence` for VH |

**Download URL**:
```
https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv
```

Already working in current code. Keep it.

### OAS (Observed Antibody Space) — Winner (Negatives)

**URL**: `https://opig.stats.ox.ac.uk/webapps/oas/`

| What | Details |
|------|---------|
| Content | 2.4B+ antibody sequences from NGS studies |
| Species | Human, mouse, rabbit, etc. |
| Format | .csv.gz files (JSON metadata + CSV data) |
| Access | Free bulk download via wget |

**How to get data**:
1. POST to `/webapps/oas/oas_unpaired/` with filters
2. Response contains wget commands for .csv.gz files
3. Each file: first line = JSON metadata, then CSV

**Human VH data**: 13,265 data units available.

```python
# Get download URLs
import requests
resp = requests.post(
    "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/",
    data={"Species": "human", "Chain": "Heavy"}
)
# Parse wget commands from response
```

**CSV columns**: `sequence`, `productive`, `v_call`, `j_call`, etc.

### Also-Rans

| Database | Why Not |
|----------|---------|
| IMGT | License required for bulk |
| abYsis | Limited access, ~50k sequences |
| SAbDab | Contains therapeutics (contamination) |
| DrugBank | Sequences incomplete |

## Recommended Approach

Following Raybould et al. 2019:

1. **Positives**: TheraSAbDab approved + Phase III (~500 sequences)
2. **Negatives**: OAS human VH, sample ~2000-5000 sequences
3. **Filter**: `productive == True`, deduplicate
4. **Remove overlap**: Exclude any OAS sequences matching therapeutics

## Key Numbers

| Class | Source | Sequences |
|-------|--------|-----------|
| Positive | TheraSAbDab | ~500-1000 |
| Negative | OAS human VH | Sample ~2000 |

Class imbalance handled by stratified splits and weighted loss.

## References

1. Raybould et al. 2019. "Predicting Antibody Developability from Sequence." PNAS.
2. Raybould et al. 2020. "Thera-SAbDab." NAR.
3. Olsen et al. 2022. "Observed Antibody Space." Protein Science.
