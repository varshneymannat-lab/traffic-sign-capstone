# Synthetic Video Results

[Open or download the demonstration video](../demo/traffic_sign_demo.mp4).

The 20-second clip compares generated input images with actual detector and classifier output. The input consists of static synthetic images, not driving footage. The reference category below each example comes from annotations and is separate from the model predictions.

| Time | Reference category | Detector output | Classifier output |
| --- | --- | --- | --- |
| 0-4 seconds | Stop | Stop sign | Stop |
| 4-8 seconds | Speed limit | No accepted detection | No crop |
| 8-12 seconds | Traffic light | Traffic light | Traffic light |
| 12-16 seconds | Pedestrian crossing | No accepted detection | No crop |
| 16-20 seconds | No entry | Stop sign | No entry |

The video classifier was trained separately on 120 generated images for 15 epochs, using 84 training, 18 validation, and 18 test examples. Its best checkpoint correctly classified 18 of 18 held-out annotated crops. This measures synthetic crop classification, not complete video accuracy or real-road performance. The original model checkpoints and original ONNX export were preserved.

The detector retains its original two-category filter. The missed examples require detector improvements. The classifier's correction of the no-entry example does not establish reliable detection of that category.

Processing the 200-frame input took approximately 9.91 seconds including model loading, about 20.18 frames per second, on macOS arm64 using CPU and two PyTorch threads. All generated frames were decoded to check their count and dimensions. The tracked presentation copy uses H.264 at 1280 by 720 pixels and ten frames per second.

The [technical report](technical_report.pdf) discusses the experiment and limitations. Per-image crop predictions, detector observations, and timing details are preserved in [experiment_results.json](experiment_results.json). Generated data, checkpoints, and logs are local outputs under `outputs/presentation_demo/`, outside the original DVC pipeline.

## Reproduction

With the project dependencies installed, run from the repository:

```bash
python3 scripts/create_demo_dataset.py --output-dir outputs/presentation_demo/data/raw --samples-per-class 24 --seed 123
python3 scripts/prepare_dataset.py --input outputs/presentation_demo/data/raw/annotations.csv --output outputs/presentation_demo/data/processed/annotations_split.csv --seed 123
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m src.train paths.data_dir=outputs/presentation_demo/data paths.model_dir=outputs/presentation_demo/models paths.log_dir=outputs/presentation_demo/logs paths.output_dir=outputs/presentation_demo/training model.pretrained=false trainer.max_epochs=15 trainer.accelerator=cpu data.batch_size=16 data.num_workers=0 test_after_training=true trainer.log_every_n_steps=1
python3 scripts/create_presentation_video.py
```

The detector weights download on first use if absent. Playback conversion to H.264 is a separate encoding step and does not change model predictions.
