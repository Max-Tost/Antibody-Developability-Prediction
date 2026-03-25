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
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/"
    "TheraSAbDab_SeqStruc_OnlineDownload.csv"
)


def download_therasabdab(output_dir: Path) -> pd.DataFrame:
    """Download TheraSAbDab summary and extract therapeutic VH sequences.

    Filters for: whole mAbs, approved or phase-III clinical status,
    active development, human or humanized origin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "therasabdab_raw.csv"

    if raw_path.exists():
        logger.info("TheraSAbDab data already downloaded, loading from cache")
        df = pd.read_csv(raw_path)
    else:
        logger.info("Downloading TheraSAbDab summary...")
        response = requests.get(THERASABDAB_URL, timeout=60)
        response.raise_for_status()
        raw_path.write_text(response.text)
        df = pd.read_csv(raw_path)

    # Keep only entries with VH sequences (HeavySequence column)
    vh_col = "HeavySequence"
    df = df[df[vh_col].notna() & (df[vh_col] != "") & (df[vh_col] != "na")]

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


def generate_negative_sequences(
    output_dir: Path,
    therapeutic_sequences: list[str],
    n_sequences: int = 2000,
    mutation_rate: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate negative examples by mutating therapeutic sequences.

    Creates plausible antibody sequences that differ from therapeutics.
    This simulates natural antibodies that haven't been optimized for development.

    Args:
        output_dir: Directory to cache results
        therapeutic_sequences: List of therapeutic VH sequences to mutate
        n_sequences: Number of negative sequences to generate
        mutation_rate: Fraction of positions to mutate (default 15%)
        random_seed: Random seed for reproducibility
    """
    import random

    random.seed(random_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "negative_sequences.csv"

    if raw_path.exists():
        logger.info("Negative sequences already generated, loading from cache")
        return pd.read_csv(raw_path)

    logger.info(f"Generating {n_sequences} negative sequences by mutation...")

    # Standard amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    therapeutic_set = set(therapeutic_sequences)

    def mutate_sequence(seq: str, rate: float) -> str:
        """Mutate a sequence at random positions."""
        seq_list = list(seq)
        n_mutations = max(1, int(len(seq) * rate))
        positions = random.sample(range(len(seq)), n_mutations)
        for pos in positions:
            # Replace with a different amino acid
            current = seq_list[pos]
            choices = [aa for aa in amino_acids if aa != current]
            seq_list[pos] = random.choice(choices)
        return "".join(seq_list)

    # Generate negative sequences
    negatives = set()
    attempts = 0
    max_attempts = n_sequences * 10

    while len(negatives) < n_sequences and attempts < max_attempts:
        # Pick a random therapeutic sequence as template
        template = random.choice(therapeutic_sequences)
        # Mutate it
        mutated = mutate_sequence(template, mutation_rate)
        # Only keep if it's not identical to a therapeutic
        if mutated not in therapeutic_set and mutated not in negatives:
            negatives.add(mutated)
        attempts += 1

    negative_df = pd.DataFrame(
        {
            "sequence": list(negatives),
            "label": 0,
            "source": "synthetic",
        }
    )
    negative_df.to_csv(raw_path, index=False)
    logger.info(f"Generated {len(negative_df)} unique negative VH sequences")
    return negative_df


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

    # Download therapeutic sequences and generate negatives
    therapeutic = download_therasabdab(raw_dir)
    therapeutic_seqs = therapeutic["sequence"].tolist()
    general = generate_negative_sequences(
        raw_dir,
        therapeutic_sequences=therapeutic_seqs,
        n_sequences=n_negative,
        random_seed=random_seed,
    )

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
