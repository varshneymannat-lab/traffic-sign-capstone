# Real-Time Traffic Sign Detection and Classification

## Abstract

This project presents a beginner-friendly computer-vision system for detecting and classifying traffic signs in road images or video. The system uses a two-stage design: a YOLO detector locates sign-like objects, and a ResNet-18 classifier predicts the sign category from each cropped region. The repository includes Hydra configuration, PyTorch Lightning training, TensorBoard logging, DVC pipeline tracking, Docker reproducibility, and ONNX export.

## 1. Introduction

Traffic signs are important for road safety and driver-assistance systems. A model that can recognize signs such as stop signs, speed-limit signs, traffic lights, pedestrian crossings, and no-entry signs can support basic intelligent transportation applications.

This project was selected because it is practical but not too complex for a first computer-vision graduation project. The task is easier to explain than multi-attribute vehicle classification, while still demonstrating object detection, image classification, evaluation, and deployment.

## 2. Methodology

### 2.1 Dataset

The expected dataset contains images with bounding boxes around traffic signs and one label per sign. The required annotation format is:

```csv
image_path,xmin,ymin,xmax,ymax,sign_label
```

The repository also includes a synthetic data generator for smoke testing. It creates simple sign-like images for five categories: stop, speed limit, traffic light, pedestrian crossing, and no entry. This synthetic data is not meant to replace a real dataset; it exists to demonstrate the pipeline.

### 2.2 Preprocessing

The preprocessing script validates annotation columns, checks bounding-box coordinates, and creates train, validation, and test splits. The dataset class crops each sign region and applies image transforms.

Training augmentation includes resizing, horizontal flipping, color jitter, and small rotations. Evaluation uses deterministic resizing and ImageNet normalization.

### 2.3 Detection

The video pipeline uses an Ultralytics YOLO model for real-time-style detection. The default pretrained model can identify COCO classes such as traffic lights and stop signs. In a larger version of the project, YOLO can be fine-tuned on custom traffic sign bounding boxes.

### 2.4 Classification

The classifier is a ResNet-18 model implemented with PyTorch Lightning. It predicts one traffic sign category from each cropped image. The model is trained with cross-entropy loss.

Classification metrics include accuracy, macro precision, macro recall, and macro F1-score.

## 3. Experiments

The repository supports TensorBoard tracking for training and validation curves. A lightweight DVC pipeline runs a one-epoch smoke training job so the full workflow can be reproduced quickly.

For final experimentation, the recommended comparisons are:

- synthetic demo data vs. real traffic sign data,
- ResNet-18 vs. MobileNetV3,
- crop sizes of 160, 224, and 256 pixels,
- PyTorch inference vs. ONNX inference.

## 4. Reproducibility

The project uses:

- Hydra for experiment configuration,
- PyTorch Lightning for training,
- TensorBoard for logs,
- DVC for data/model pipeline tracking,
- Docker for environment reproducibility,
- ONNX for deployment export.

These tools make the project easier to rerun, explain, and improve.

## 5. Discussion

The two-stage approach is useful for a student project because each part is understandable. YOLO handles localization, and the classifier handles category prediction. The main limitation is that the included synthetic dataset is simple. Real road images introduce challenges such as blur, lighting changes, small signs, occlusion, and unusual camera angles.

Future work should include training or fine-tuning on a real traffic sign dataset and measuring detection mAP in addition to classification metrics.

## 6. Conclusion

This project delivers a complete, reproducible traffic sign detection and classification pipeline. It is simple enough to present clearly as a graduation project while still demonstrating important deep-learning and MLOps concepts.
