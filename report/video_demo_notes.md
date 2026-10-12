# Synthetic Video Demonstration

## What to open

Open `outputs/presentation_demo/traffic_sign_demo_playback.mp4` if available. The original presentation video is `outputs/presentation_demo/traffic_sign_video_demo.mp4`. It lasts 20 seconds and compares input with actual processed output. Open it in QuickTime Player and pause after each example if you want to explain the results.

The input consists of static synthetic images assembled into video. It is not driving footage. The boxes and prediction labels come from YOLO and the classifier, not from the known annotations. The annotation category shown below each example is a separate reference label.

## What to say before playing

"I will show a short synthetic demonstration. The left side is the generated input, and the right side is the output from the detector and classifier. This uses a separate classifier trained for the video demo. It does not establish accuracy on real roads. The clip also shows missed detections, so I can explain the current limitations."

## What to explain during playback

| Time | Example | Observed output | Explanation |
| --- | --- | --- | --- |
| 0-4 seconds | Stop | YOLO finds a stop sign and the classifier predicts stop | Both stages give the expected category on this example. |
| 4-8 seconds | Speed limit | No accepted detection | The classifier receives no crop. Correct crop classification does not guarantee video recognition. |
| 8-12 seconds | Traffic light | YOLO finds a traffic light and the classifier predicts traffic light | The pipeline adds the expected label on this example. It does not identify light state. |
| 12-16 seconds | Pedestrian crossing | No accepted detection | The detector needs suitable training and matching category handling. |
| 16-20 seconds | No entry | YOLO calls it a stop sign, while the classifier predicts no entry | The classifier corrects this particular category error. This does not prove that YOLO reliably detects no-entry signs. |

## Separate training run

The original ten-image dataset, original model checkpoints, and original ONNX model were preserved. The video uses a new checkpoint under `outputs/presentation_demo/models/checkpoints/best.ckpt`.

- 120 generated images with 24 examples per category.
- 84 training examples, 18 validation examples, and 18 test examples.
- ResNet-18 trained from scratch for 15 epochs on CPU.
- The selected checkpoint correctly classified 18 of 18 held-out annotated crops in a separate direct evaluation.
- These crops come from the same simple image generator. This result does not measure real-world generalisation or full-pipeline accuracy.
- The video still uses the general pretrained YOLO detector and its existing two-category filter.
- Video inference uses the PyTorch checkpoint. The separately trained demo classifier has not been exported here as a replacement for the original ONNX model.
- The demo files are under an ignored output directory. They have not been published to GitHub or added to the DVC pipeline.

The machine-readable details are in `outputs/presentation_demo/video_results.json`. All 200 frames in each generated video were decoded to verify frame count and dimensions. In this run, processing the 20-second input took approximately 9.91 seconds including model loading, about 20.18 frames per second, on CPU with two PyTorch threads. This is a measurement on repeated synthetic scenes, not a claim about real-road processing speed.

## What to say after playing

"This demonstrates that the program can read a video, produce detections, classify the detected crops, and save annotated output. It also shows why the detector needs improvement: two examples never reach the classifier. My next step would be real-data detector training and a representative evaluation."

## Reproducing the demo

Run these commands from the repository with its dependencies installed:

```bash
python3 scripts/create_demo_dataset.py --output-dir outputs/presentation_demo/data/raw --samples-per-class 24 --seed 123
python3 scripts/prepare_dataset.py --input outputs/presentation_demo/data/raw/annotations.csv --output outputs/presentation_demo/data/processed/annotations_split.csv --seed 123
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m src.train paths.data_dir=outputs/presentation_demo/data paths.model_dir=outputs/presentation_demo/models paths.log_dir=outputs/presentation_demo/logs paths.output_dir=outputs/presentation_demo/training model.pretrained=false trainer.max_epochs=15 trainer.accelerator=cpu data.batch_size=16 data.num_workers=0 test_after_training=true trainer.log_every_n_steps=1
python3 scripts/create_presentation_video.py
```

The first YOLO run needs to download `yolo11n.pt` if it is not already present. A compatible MP4 playback conversion can be performed separately on macOS using `avconvert`.
