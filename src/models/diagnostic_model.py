"""Frozen diagnostic classifier wrapper (ResNet50 fine-tuned on BUSI)."""
import torch
import torch.nn as nn
import timm


NUM_CLASSES = 3  # normal, benign, malignant


class DiagnosticModel(nn.Module):
    def __init__(self, checkpoint_path: str | None = None, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state["model_state"])
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (class_indices, probabilities)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return preds, probs
