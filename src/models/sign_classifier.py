from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision.models import ResNet18_Weights, resnet18


class TrafficSignClassifierLitModule(LightningModule):
    def __init__(
        self,
        sign_classes: list[str],
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.sign_head = nn.Linear(feature_dim, len(sign_classes))
        self.loss_fn = nn.CrossEntropyLoss()

        self.sign_acc = MulticlassAccuracy(num_classes=len(sign_classes))
        self.sign_f1 = MulticlassF1Score(num_classes=len(sign_classes), average="macro")
        self.sign_precision = MulticlassPrecision(num_classes=len(sign_classes), average="macro")
        self.sign_recall = MulticlassRecall(num_classes=len(sign_classes), average="macro")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.sign_head(features)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        sign_logits = self(batch["image"])
        loss = self.loss_fn(sign_logits, batch["sign_label"])

        sign_preds = sign_logits.argmax(dim=1)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc", self.sign_acc(sign_preds, batch["sign_label"]), on_epoch=True)
        self.log(f"{stage}/f1", self.sign_f1(sign_preds, batch["sign_label"]), on_epoch=True)
        self.log(f"{stage}/precision", self.sign_precision(sign_preds, batch["sign_label"]), on_epoch=True)
        self.log(f"{stage}/recall", self.sign_recall(sign_preds, batch["sign_label"]), on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
