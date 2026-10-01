import torch

from src.models.sign_classifier import TrafficSignClassifierLitModule


def test_traffic_sign_classifier_forward_shapes():
    model = TrafficSignClassifierLitModule(
        sign_classes=["stop", "speed_limit", "traffic_light"],
        pretrained=False,
    )
    sign_logits = model(torch.randn(2, 3, 224, 224))
    assert sign_logits.shape == (2, 3)
