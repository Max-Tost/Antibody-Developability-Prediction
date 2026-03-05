"""Entry point: train antibody developability classifier with Hydra config."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.trainer import run_kfold_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    run_kfold_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
