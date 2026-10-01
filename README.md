# Real-Time Traffic Sign Detection and Classification

This is an original graduation capstone project inspired by the example project structure, but with a simpler and more student-friendly topic. The system detects traffic-sign-like objects in road images or videos and classifies each crop into a sign category.

The project follows the requested modern machine-learning workflow: Hydra configuration, PyTorch Lightning training, TensorBoard logging, DVC data/model tracking, Docker reproducibility, and ONNX export.

## Why I Selected This Project

I selected traffic sign detection because it is practical but manageable for a first computer-vision graduation project. Traffic signs are visually clear, have meaningful categories, and connect naturally to road safety and driver-assistance systems.

Compared with a more complex vehicle type/color project, this scope is easier to explain and defend:

- The model learns one clear label: the traffic sign category.
- The demo dataset is small enough to reproduce quickly.
- The project still demonstrates detection, classification, evaluation, and deployment.
- The topic has a real-world purpose without being too advanced.

## Project Goal

Build an end-to-end system that can:

1. Detect traffic signs or traffic-sign-like regions in road frames.
2. Crop each detected sign region.
3. Classify the crop as one of:
   - `stop`
   - `speed_limit`
   - `traffic_light`
   - `pedestrian_crossing`
   - `no_entry`
4. Write an annotated output video with bounding boxes and labels.

## Method

The project uses a simple two-stage design:

1. **Detection stage:** Ultralytics YOLO is used during video inference to locate sign-like objects. The default COCO model can detect classes such as traffic lights and stop signs. For a real final dataset, YOLO can be fine-tuned on custom traffic-sign bounding boxes.
2. **Classification stage:** A ResNet-18 classifier predicts the sign category from cropped sign images.

This design is easier to understand than modifying a detector architecture directly, and it lets detection and classification be improved separately.

## Repository Structure

```text
configs/                 Hydra experiment configs
data/                    DVC-tracked data folders
models/                  DVC-tracked model artifacts
src/
  data/                  Dataset and Lightning DataModule
  models/                Traffic sign classifier
  utils/                 Video annotation pipeline
  train.py               Train classifier
  eval.py                Evaluate classifier
  infer_video.py         Run video inference
  export_onnx.py         Export classifier to ONNX
scripts/                 Dataset preparation and demo-data generation
tests/                   Unit/smoke tests
report/                  Technical report and mentor defense notes
dvc.yaml                 Reproducible pipeline stages
Dockerfile               Reproducible environment
```

## Dataset Format

Place annotated data in `data/raw/annotations.csv`:

```csv
image_path,xmin,ymin,xmax,ymax,sign_label
images/frame_0001.jpg,82,31,145,94,stop
images/frame_0002.jpg,94,25,156,87,speed_limit
```

`image_path` is relative to `data/raw/` unless it is an absolute path. Bounding boxes use pixel coordinates.

Supported labels are configured in [configs/data/traffic_signs.yaml](/Users/mannatvarshney/Documents/New%20project/traffic_sign_capstone/configs/data/traffic_signs.yaml).

## Quick Demo

The repository includes a small synthetic traffic-sign dataset generator. This is useful for presenting the workflow without downloading a large dataset.

```bash
make demo-data
make prepare
```

Or run the DVC pipeline:

```bash
dvc repro
```

The generated data is for demonstration and smoke testing. A real project submission can replace it with a public traffic sign dataset such as GTSDB/GTSRB or a small custom annotated dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 -m src.train
```

TensorBoard logs are written to `logs/tensorboard/`.

```bash
tensorboard --logdir logs/tensorboard
```

## Evaluate

```bash
python3 -m src.eval ckpt_path=models/checkpoints/best.ckpt
```

## Export ONNX

```bash
python3 -m src.export_onnx ckpt_path=models/checkpoints/best.ckpt output_path=models/traffic_sign_classifier.onnx
```

## Video Inference

```bash
python3 -m src.infer_video \
  input_video=data/raw/sample.mp4 \
  output_video=outputs/annotated_sample.mp4 \
  classifier_ckpt=models/checkpoints/best.ckpt
```

## DVC Pipeline

```bash
dvc repro
```

Pipeline stages:

- `demo_data`: creates a tiny synthetic traffic sign dataset.
- `prepare`: validates and splits annotations.
- `train`: runs a one-epoch smoke training job.
- `export`: exports the classifier to ONNX.

The default DVC run is intentionally lightweight so it can be reproduced during review. For a more serious experiment, use a real dataset and train longer:

```bash
python3 -m src.train model.pretrained=true trainer.max_epochs=20
```

## Docker

```bash
docker build -t traffic-sign-capstone .
docker run --rm -it -v "$PWD:/workspace" traffic-sign-capstone python3 -m src.train
```

## Mentor Defense Talking Points

- I chose traffic signs because they are visually clear and beginner-friendly, but still useful in driver-assistance systems.
- I used a two-stage pipeline because it is easier to understand: first detect a sign, then classify the crop.
- I used ResNet-18 because it is a standard CNN backbone and not too large.
- I used Hydra and Lightning so experiments are organized and reproducible.
- I used DVC so data and model artifacts can be tracked separately from Git code.
- I exported to ONNX because it shows how a trained model can be prepared for deployment.

## Notes

The included ONNX/checkpoint artifacts are smoke-demo artifacts, not final research-quality models. For final presentation results, replace the synthetic data with real traffic sign images, run training, and report accuracy, precision, recall, and F1-score.
