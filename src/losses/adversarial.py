"""Least-Squares GAN (LSGAN) adversarial loss."""
import torch
import torch.nn as nn


class LSGANLoss(nn.Module):
    """
    LSGAN replaces the log-loss with MSE, giving smoother gradients and
    avoiding the vanishing-gradient problem in the early training stages.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return self.mse(prediction, target)
