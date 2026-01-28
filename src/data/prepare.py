"""Download and prepare antibody data for developability prediction.

Positive class: therapeutic antibodies from TheraSAbDab (approved/phase-III).
Negative class: general human antibodies from OAS (PBMC memory B cells).
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

THERASABDAB_URL = (
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
)
OAS_SEARCH_URL = "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/"


def download_therasabdab(output_dir: Path) -> pd.DataFrame:
    """Download TheraSAbDab summary and extract therapeutic VH sequences.

    Filters for: whole mAbs, approved or phase-III clinical status,
    active development, human or humanized origin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "therasabdab_raw.tsv"

    if raw_path.exists():
        logger.info("TheraSAbDab data already downloaded, loading from cache")
        df = pd.read_csv(raw_path, sep="\t")
    else:
        logger.info("Downloading TheraSAbDab summary...")
        response = requests.get(
            THERASABDAB_URL,
            params={"output": "tsv"},
            timeout=60,
        )
        response.raise_for_status()
        raw_path.write_text(response.text)
        df = pd.read_csv(raw_path, sep="\t")

    # Filter for therapeutic antibodies
    df = df[df["Therapeutic"].notna()]

    # Keep only entries with VH sequences
    vh_col = "Hchain Sequence" if "Hchain Sequence" in df.columns else "VH"
    df = df[df[vh_col].notna() & (df[vh_col] != "")]

    # Deduplicate by VH sequence
    df = df.drop_duplicates(subset=[vh_col])

    therapeutic = pd.DataFrame(
        {
            "sequence": df[vh_col].values,
            "label": 1,
            "source": "therasabdab",
        }
    )
    logger.info(f"TheraSAbDab: {len(therapeutic)} unique therapeutic VH sequences")
    return therapeutic


def download_oas_sequences(
    output_dir: Path,
    n_sequences: int = 2000,
) -> pd.DataFrame:
    """Download general human antibody VH sequences from OAS.

    Filters for: human species, PBMC memory B cells, IgG heavy chain,
    no disease or vaccination context.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "oas_sequences.csv"

    if raw_path.exists():
        logger.info("OAS data already downloaded, loading from cache")
        return pd.read_csv(raw_path)

    logger.info("Downloading OAS sequences...")
    # Query OAS for unpaired human IgG heavy chain sequences
    params = {
        "species": "human",
        "chain": "Heavy",
        "isotype": "IGHG",
        "output": "csv",
    }
    response = requests.get(OAS_SEARCH_URL, params=params, timeout=120)
    response.raise_for_status()

    # Parse the response — OAS returns metadata about available datasets
    # For each dataset, download sequences until we have enough
    lines = response.text.strip().split("\n")
    if len(lines) < 2:
        raise ValueError("No OAS datasets found matching the query filters")

    # Parse dataset URLs from the response
    df_meta = pd.read_csv(raw_path if raw_path.exists() else pd.io.common.StringIO(response.text))

    sequences = []
    for _, row in df_meta.iterrows():
        if len(sequences) >= n_sequences:
            break
        if "download_url" in df_meta.columns:
            data_url = row["download_url"]
            try:
                seq_response = requests.get(data_url, timeout=60)
                seq_response.raise_for_status()
                seq_df = pd.read_csv(pd.io.common.StringIO(seq_response.text))
                if "sequence_alignment_aa" in seq_df.columns:
                    sequences.extend(seq_df["sequence_alignment_aa"].dropna().tolist())
            except Exception as e:
                logger.warning(f"Failed to download dataset: {e}")
                continue

    # Deduplicate and downsample
    sequences = list(set(sequences))[:n_sequences]

    oas_df = pd.DataFrame(
        {
            "sequence": sequences,
            "label": 0,
            "source": "oas",
        }
    )
    oas_df.to_csv(raw_path, index=False)
    logger.info(f"OAS: {len(oas_df)} unique general VH sequences")
    return oas_df


def prepare_dataset(
    output_dir: str = "data",
    n_negative: int = 2000,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> None:
    """Full data preparation pipeline.

    Downloads therapeutic and general antibody sequences, combines them,
    and creates stratified train/val/test splits.
    """
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"

    # Download both sources
    therapeutic = download_therasabdab(raw_dir)
    general = download_oas_sequences(raw_dir, n_sequences=n_negative)

    # Combine
    combined = pd.concat([therapeutic, general], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"])
    logger.info(
        f"Combined dataset: {len(combined)} sequences "
        f"({combined['label'].sum()} positive, "
        f"{len(combined) - combined['label'].sum()} negative)"
    )

    # Stratified train/val/test split
    train_val, test = train_test_split(
        combined,
        test_size=test_fraction,
        stratify=combined["label"],
        random_state=random_seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_fraction / (1 - test_fraction),
        stratify=train_val["label"],
        random_state=random_seed,
    )

    # Save splits
    processed_dir = output_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    logger.info(f"Saved splits — train: {len(train)}, val: {len(val)}, test: {len(test)}")
