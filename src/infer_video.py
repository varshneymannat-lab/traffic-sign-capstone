from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.utils.video_pipeline import annotate_video


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    annotate_video(
        input_video=cfg.input_video,
        output_video=cfg.output_video,
        classifier_ckpt=cfg.classifier_ckpt,
        sign_classes=list(cfg.data.sign_classes),
        image_size=cfg.data.image_size,
        yolo_model=cfg.yolo_model,
        device=cfg.device,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        max_frames=cfg.max_frames,
    )


if __name__ == "__main__":
    main()
