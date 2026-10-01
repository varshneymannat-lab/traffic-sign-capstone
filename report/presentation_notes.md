# Presentation Notes

## Project Choice

I selected **Real-Time Traffic Sign Detection and Classification for Driver Assistance** because it is practical, understandable, and appropriate for someone learning computer vision for the first time. It is related to intelligent transportation, but it is simpler than a project that detects many vehicle types and colors.

## Why This Is My Own Project

The example project involved detecting cars and classifying vehicle type and color. My project uses a similar technical idea, but the topic, labels, dataset design, and explanation are different. Instead of cars, my system focuses on traffic signs such as stop signs, speed-limit signs, traffic lights, pedestrian crossings, and no-entry signs.

## Technical Approach

The project uses a two-stage pipeline:

1. YOLO detects traffic-sign-like objects in a frame.
2. A ResNet-18 classifier predicts the sign category from the cropped region.

This is a good beginner design because each part has a clear role. If the final prediction is wrong, I can check whether the detection stage missed the sign or whether the classifier mislabeled the crop.

## Why YOLO

YOLO is widely used for real-time object detection. It is suitable for video because it is fast and can be tested with pretrained weights. In this project, YOLO is used as the detection component, and it can later be fine-tuned on a custom traffic sign dataset.

## Why ResNet-18

ResNet-18 is a standard convolutional neural network. It is powerful enough for traffic sign classification but still small enough for a student project and laptop experiments.

## Reproducibility

The project includes:

- Hydra for configuration.
- PyTorch Lightning for training structure.
- TensorBoard for experiment logs.
- DVC for data and model pipeline tracking.
- Docker for environment reproducibility.
- ONNX export for deployment.

## Evaluation Plan

The classifier is evaluated using accuracy, precision, recall, and F1-score. For a full real-data version, the detector can also be evaluated using mAP.

## Limitations

The included synthetic dataset is only for demonstration. It proves the code and pipeline work, but final model performance should be measured on a real traffic sign dataset or a small custom annotated dataset.
