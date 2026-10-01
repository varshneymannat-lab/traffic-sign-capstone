.PHONY: install demo-data prepare train eval export infer test tensorboard

install:
	python3 -m pip install -r requirements.txt

demo-data:
	python3 scripts/create_demo_dataset.py

prepare:
	python3 scripts/prepare_dataset.py

train:
	python3 -m src.train

eval:
	python3 -m src.eval

export:
	python3 -m src.export_onnx

infer:
	python3 -m src.infer_video

test:
	pytest -q

tensorboard:
	tensorboard --logdir logs/tensorboard
