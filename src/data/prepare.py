"""Download and prepare antibody data for developability prediction.

Positive class: therapeutic antibodies from TheraSAbDab.
Negative class: general human antibodies from OAS.
"""

import gzip
import io
import logging
import random
import re
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

THERASABDAB_URL = (
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/"
    "TheraSAbDab_SeqStruc_OnlineDownload.csv"
)
OAS_SEARCH_URL = "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/"


def download_therasabdab(output_dir: Path) -> pd.DataFrame:
    """Download TheraSAbDab and extract therapeutic VH sequences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "therasabdab_raw.csv"

    if raw_path.exists():
        logger.info("TheraSAbDab cached, loading...")
        df = pd.read_csv(raw_path)
    else:
        logger.info("Downloading TheraSAbDab...")
        resp = requests.get(THERASABDAB_URL, timeout=60)
        resp.raise_for_status()
        raw_path.write_text(resp.text)
        df = pd.read_csv(raw_path)

    vh_col = "HeavySequence"
    df = df[df[vh_col].notna() & (df[vh_col] != "") & (df[vh_col] != "na")]
    df = df.drop_duplicates(subset=[vh_col])

    therapeutic = pd.DataFrame({"sequence": df[vh_col].values, "label": 1, "source": "therasabdab"})
    logger.info(f"TheraSAbDab: {len(therapeutic)} unique VH sequences")
    return therapeutic


def download_oas(
    output_dir: Path,
    therapeutic_sequences: set[str],
    n_sequences: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Download human VH sequences from OAS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "oas_sequences.csv"

    if cache_path.exists():
        logger.info("OAS cached, loading...")
        return pd.read_csv(cache_path)

    logger.info("Fetching OAS download URLs...")
    resp = requests.post(OAS_SEARCH_URL, data={"Species": "human", "Chain": "Heavy"}, timeout=60)
    urls = re.findall(r"wget (https://[^\"]+\.csv\.gz)", resp.text)
    logger.info(f"Found {len(urls)} OAS data units")

    random.seed(random_seed)
    random.shuffle(urls)

    sequences = set()
    for url in urls:
        if len(sequences) >= n_sequences:
            break
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            content = gzip.decompress(r.content).decode("utf-8")
            lines = content.strip().split("\n")
            df = pd.read_csv(io.StringIO("\n".join(lines[1:])))
            productive = df[df["productive"] == "T"]["sequence"].dropna().tolist()
            # Filter out therapeutics and add up to 500 per unit
            clean = [s for s in productive if s not in therapeutic_sequences and s not in sequences]
            sequences.update(clean[:500])
            logger.info(f"OAS: {len(sequences)}/{n_sequences} sequences")
        except Exception:
            continue

    oas_df = pd.DataFrame({"sequence": list(sequences)[:n_sequences], "label": 0, "source": "oas"})
    oas_df.to_csv(cache_path, index=False)
    logger.info(f"OAS: {len(oas_df)} unique VH sequences")
    return oas_df


def prepare_dataset(
    output_dir: str = "data",
    n_negative: int = 2000,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> None:
    """Download antibody data and create train/val/test splits."""
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"

    therapeutic = download_therasabdab(raw_dir)
    therapeutic_set = set(therapeutic["sequence"])
    general = download_oas(
        raw_dir, therapeutic_set, n_sequences=n_negative, random_seed=random_seed
    )

    combined = pd.concat([therapeutic, general], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence"])
    n_pos, n_neg = combined["label"].sum(), len(combined) - combined["label"].sum()
    logger.info(f"Combined: {len(combined)} ({n_pos} pos, {n_neg} neg)")

    train_val, test = train_test_split(
        combined, test_size=test_fraction, stratify=combined["label"], random_state=random_seed
    )
    val_size = val_fraction / (1 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["label"], random_state=random_seed
    )

    processed_dir = output_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    logger.info(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")
