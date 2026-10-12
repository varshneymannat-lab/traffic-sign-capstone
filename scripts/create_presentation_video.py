from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.utils.video_pipeline import (
    TRAFFIC_SIGN_LIKE_CLASS_IDS,
    annotate_video,
    classify_crop,
    load_classifier,
    preprocess_crop,
)
from ultralytics import YOLO


def writer(path: Path, fps: int, dimensions: tuple[int, int]):
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, dimensions)
    if not video.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return video


def inspect_video(path: Path, expected_frames: int) -> dict:
    video = cv2.VideoCapture(str(path))
    frames = 0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            raise RuntimeError("Video dimensions changed during decoding.")
        frames += 1
    video.release()
    if frames != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} frames, decoded {frames}.")
    return {"decoded_frames": frames, "width": width, "height": height, "fps": fps}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Make a labelled synthetic model demonstration.")
    parser.add_argument("--demo-dir", default="outputs/presentation_demo")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    demo = PROJECT / args.demo_dir
    checkpoint = demo / "models/checkpoints/best.ckpt"
    labels = ["stop", "speed_limit", "traffic_light", "pedestrian_crossing", "no_entry"]
    data = pd.read_csv(demo / "data/processed/annotations_split.csv")
    test = data[data["split"] == "test"]
    transform = preprocess_crop(224)
    model = load_classifier(checkpoint, labels, "cpu")
    test_results = []
    for row in test.itertuples():
        image = Image.open(demo / "data/raw" / row.image_path).convert("RGB")
        tensor = transform(image.crop((row.xmin, row.ymin, row.xmax, row.ymax)))
        prediction = labels[int(model(tensor.unsqueeze(0)).argmax(1).item())]
        test_results.append({"image": row.image_path, "actual": row.sign_label, "predicted": prediction})

    selected = [test[test.sign_label == label].iloc[0] for label in labels]
    fps, segment_seconds = 10, 4
    segment_frames = fps * segment_seconds
    frame_total = len(selected) * segment_frames
    input_path = demo / "synthetic_input.mp4"
    annotated_path = demo / "annotated_output.mp4"
    presentation_path = demo / "traffic_sign_video_demo.mp4"
    source = writer(input_path, fps, (640, 456))
    examples = []
    for row in selected:
        image = cv2.imread(str(demo / "data/raw" / row.image_path))
        if image is None:
            raise FileNotFoundError(row.image_path)
        frame = cv2.resize(image, (640, 456), interpolation=cv2.INTER_LINEAR)
        examples.append(frame)
        for _ in range(segment_frames):
            source.write(frame)
    source.release()

    start = time.perf_counter()
    annotate_video(
        input_video=input_path,
        output_video=annotated_path,
        classifier_ckpt=checkpoint,
        sign_classes=labels,
        image_size=224,
        yolo_model=str(PROJECT / args.yolo_model),
        device="cpu",
        conf_threshold=0.35,
        iou_threshold=0.5,
    )
    elapsed = time.perf_counter() - start

    detector = YOLO(str(PROJECT / args.yolo_model))
    segment_predictions = []
    for frame, row in zip(examples, selected):
        detections = []
        result = detector.predict(frame, device="cpu", conf=0.35, iou=0.5, verbose=False)[0]
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in TRAFFIC_SIGN_LIKE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(639, x2), min(455, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            predicted = classify_crop(frame[y1:y2, x1:x2], model, transform, labels, "cpu")
            detections.append({"detector_category": detector.names[class_id], "classifier_category": predicted, "detector_confidence": float(box.conf.item())})
        segment_predictions.append({"image": row.image_path, "annotated_category": row.sign_label, "detections": detections})

    left = cv2.VideoCapture(str(input_path))
    right = cv2.VideoCapture(str(annotated_path))
    combined = writer(presentation_path, fps, (1280, 720))
    screenshots = []
    for index in range(frame_total):
        ok_left, input_frame = left.read()
        ok_right, output_frame = right.read()
        if not ok_left or not ok_right:
            raise RuntimeError("Missing frames in presentation source.")
        segment = index // segment_frames
        evidence = segment_predictions[segment]
        canvas = np.full((720, 1280, 3), 248, dtype=np.uint8)
        cv2.putText(canvas, "Traffic sign prototype: synthetic video demo", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (35, 35, 35), 2)
        cv2.putText(canvas, "Generated images. Actual model predictions. No real-road accuracy claim.", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (85, 85, 85), 1)
        cv2.putText(canvas, "INPUT", (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 2)
        cv2.putText(canvas, "PROCESSED OUTPUT", (660, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 55, 55), 2)
        canvas[130:586, :640] = input_frame
        canvas[130:586, 640:] = output_frame
        expected = evidence["annotated_category"].replace("_", " ")
        cv2.putText(canvas, f"Example category from annotation: {expected}", (30, 621), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 45, 45), 2)
        if evidence["detections"]:
            details = "; ".join(f"YOLO: {x['detector_category']} / classifier: {x['classifier_category'].replace('_', ' ')}" for x in evidence["detections"])
        else:
            details = "No accepted detection. The classifier receives no crop."
        cv2.putText(canvas, details, (30, 659), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (55, 95, 65), 2)
        cv2.putText(canvas, "Demo classifier trained separately. Other sign categories need detector training.", (30, 698), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (85, 85, 85), 1)
        combined.write(canvas)
        if index % segment_frames == segment_frames // 2:
            screenshot = demo / f"preview_{segment + 1}.jpg"
            cv2.imwrite(str(screenshot), canvas)
            screenshots.append(str(screenshot))
    left.release()
    right.release()
    combined.release()

    verification = {p.name: inspect_video(p, frame_total) for p in [input_path, annotated_path, presentation_path]}
    correct = sum(x["actual"] == x["predicted"] for x in test_results)
    report = {
        "dataset": "Synthetic drawings only. Not real road footage.",
        "training": {"samples": len(data[data.split == "train"]), "epochs": 15, "pretrained_classifier": False},
        "validation_samples": len(data[data.split == "val"]),
        "test": {"samples": len(test_results), "correct": correct, "crop_accuracy": correct / len(test_results), "results": test_results},
        "checkpoint": str(checkpoint),
        "video_duration_seconds": frame_total / fps,
        "processing_seconds_including_model_load": elapsed,
        "processing_fps_including_model_load": frame_total / elapsed,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 2},
        "segments": segment_predictions,
        "verification": verification,
        "limitations": ["Static generated examples assembled into video, not driving footage.", "Only stop-sign and traffic-light detector classes pass the current filter.", "Synthetic cropped-image accuracy is separate from detector or video accuracy.", "No detector mAP evaluation.", "Video uses a separately trained PyTorch classifier, not ONNX runtime."],
    }
    (demo / "video_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"video": str(presentation_path), "synthetic_test_correct": correct, "synthetic_test_count": len(test_results), "segments": segment_predictions, "processing_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
