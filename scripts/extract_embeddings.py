"""Entry point: extract ESM-2 embeddings for all dataset splits."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.models.embedder import ESM2Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    embedder = ESM2Embedder(
        model_name=cfg.embedder.model_name,
        device=cfg.embedder.device,
    )

    data_dir = Path(cfg.data.processed_dir)
    emb_dir = Path(cfg.data.embeddings_dir)

    for split in ["train", "val", "test"]:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"Skipping {split}: {csv_path} not found")
            continue

        split_emb_dir = emb_dir / split
        logger.info(f"Extracting embeddings for {split}...")
        embedder.embed_dataset(
            csv_path=str(csv_path),
            output_dir=str(split_emb_dir),
        )


if __name__ == "__main__":
    main()
