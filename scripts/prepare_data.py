"""Entry point: download and prepare antibody dataset."""

import logging

from src.data.prepare import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

if __name__ == "__main__":
    prepare_dataset(
        output_dir="data",
        n_negative=2000,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
    )
