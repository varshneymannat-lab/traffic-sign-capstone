from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer, logger=False)
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()

