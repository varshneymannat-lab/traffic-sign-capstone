from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from src.models.sign_classifier import TrafficSignClassifierLitModule


TRAFFIC_SIGN_LIKE_CLASS_IDS = {9, 11}


def load_classifier(
    checkpoint_path: str | Path,
    sign_classes: list[str],
    device: str,
) -> TrafficSignClassifierLitModule:
    model = TrafficSignClassifierLitModule(sign_classes=sign_classes)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_crop(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.inference_mode()
def classify_crop(
    crop_bgr,
    model: TrafficSignClassifierLitModule,
    transform: transforms.Compose,
    sign_classes: list[str],
    device: str,
) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    sign_logits = model(tensor)
    return sign_classes[sign_logits.argmax(1).item()]


def annotate_video(
    input_video: str | Path,
    output_video: str | Path,
    classifier_ckpt: str | Path,
    sign_classes: list[str],
    image_size: int,
    yolo_model: str,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    max_frames: int | None = None,
) -> None:
    detector = YOLO(yolo_model)
    classifier = load_classifier(classifier_ckpt, sign_classes, device)
    transform = preprocess_crop(image_size)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        result = detector.predict(
            source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False, device=device
        )[0]
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in TRAFFIC_SIGN_LIKE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            sign_label = classify_crop(
                crop, classifier, transform, sign_classes, device
            )
            label = sign_label.replace("_", " ")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 220, 80),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
