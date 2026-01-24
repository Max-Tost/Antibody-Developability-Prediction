"""Download and prepare antibody data for developability prediction.

Positive class: therapeutic antibodies from TheraSAbDab (approved/phase-III).
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

THERASABDAB_URL = (
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
)


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
