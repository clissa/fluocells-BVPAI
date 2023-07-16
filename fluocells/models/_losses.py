"""
This module contains implementations of custom loss functions for segmentation.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-07-16
License: Apache License 2.0
"""

import sys
import inspect
from pathlib import Path

SCRIPT_PATH = inspect.getfile(inspect.currentframe())
FLUOCELLS_PATH = Path(SCRIPT_PATH).parent.absolute()

sys.path.append(str(FLUOCELLS_PATH))

from fastai.vision.all import *
from fastai.losses import FocalLossFlat


# This implementation is taken from https://github.com/TIO-IKIM/CellViT/blob/main/base_ml/base_loss.py
# Slight adaptations were necessary to integrate it with c-ResUnet and fastai. They are indicated as `# ADAPTATION`
class FocalTverskyLoss(nn.Module):
    """FocalTverskyLoss

    PyTorch implementation of the Focal Tversky Loss Function for multiple classes
    doi: 10.1109/ISBI.2019.8759329
    Abraham, N., & Khan, N. M. (2019).
    A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation.
    In International Symposium on Biomedical Imaging. https://doi.org/10.1109/isbi.2019.8759329

    @ Fabian Hörst, fabian.hoerst@uk-essen.de
    Institute for Artifical Intelligence in Medicine,
    University Medicine Essen

    Args:
        alpha_t (float, optional): Alpha parameter for tversky loss (multiplied with false-negatives). Defaults to 0.7.
        beta_t (float, optional): Beta parameter for tversky loss (multiplied with false-positives). Defaults to 0.3.
        gamma_f (float, optional): Gamma Focal parameter. Defaults to 4/3.
        smooth (float, optional): Smooting factor. Defaults to 0.000001.
    """

    def __init__(
        self,
        alpha_t: float = 0.7,
        beta_t: float = 0.3,
        gamma_f: float = 4 / 3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.gamma_f = gamma_f
        self.smooth = smooth
        self.num_classes = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Predictions, logits (without Softmax). Shape: (batch-size, H, W, num_classes)
            target (torch.Tensor): Targets, either flattened (Shape: (batch.size, H, W) or as one-hot encoded (Shape: (batch-size, H, W, num_classes)).

        Raises:
            ValueError: Error if there is a shape missmatch

        Returns:
            torch.Tensor: FocalTverskyLoss (weighted)
        """
        input = input.permute(
            0, 2, 3, 1
        )  # ADAPTATION: fastai returns (batch-size, num_classes, H, W)
        if input.shape[-1] != self.num_classes:
            raise ValueError(
                "Predictions must be a logit tensor with the last dimension shape beeing equal to the number of classes"
            )
        if len(target.shape) != len(input.shape):
            # convert the targets to onehot
            target = F.one_hot(target, num_classes=self.num_classes)

        # flatten
        target = target.view(-1)
        input = torch.softmax(input, dim=-1).view(-1)

        # calculate true positives, false positives and false negatives
        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()

        Tversky = (tp + self.smooth) / (
            tp + self.alpha_t * fn + self.beta_t * fp + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma_f

        return FocalTversky

    # ADAPTATION
    def decodes(self, x):
        return x.argmax(dim=1)

    # ADAPTATION
    def activation(self, x):
        return F.softmax(x, dim=1)


class CombinedLoss:
    """Combined loss function. This include three terms that focus on complementary aspects:
    - BCE: better handles noisy labels
    - Dice: help ensuring better performance on object boundaries
    - Focal: gives more weights to challenging examples
    """

    def __init__(
        self, axis=1, w_bce=0.3, w_dice=0.3, w_focal=0.4, smooth=1e-6, gamma=2.0
    ):
        store_attr()
        self.name = "CombinedLoss"
        # self.bce = BCEWithLogitsLossFlat(axis=axis)
        self.bce_loss = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)
        self.focal_loss = FocalLossFlat(axis=axis, gamma=gamma)

    def __call__(self, preds, targets):
        return (
            self.w_bce * self.bce_loss(preds, targets)
            + self.w_dice * self.dice_loss(preds, targets)
            + self.w_focal * self.focal_loss(preds, targets)
        )

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


class CombinedFTLoss:
    """Combined loss function. This include three terms that focus on complementary aspects:
    - BCE: better handles noisy labels
    - Dice: help ensuring better performance on object boundaries
    - FocalTversky: gives more weights to challenging examples, but over-suppress misclassifications if class accuracy is high
    """

    def __init__(
        self,
        axis=1,
        w_bce=0.3,
        w_dice=0.3,
        w_focal=0.4,
        smooth=1e-6,
        gamma=4 / 3,
        beta=0.3,
    ):
        store_attr()
        self.name = "CombinedFTLoss"
        self.bce_loss = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha_t=1 - beta, beta_t=beta, gamma_f=gamma, smooth=smooth
        )

    def __call__(self, preds, targets):
        return (
            self.w_bce * self.bce_loss(preds, targets)
            + self.w_dice * self.dice_loss(preds, targets)
            + self.w_focal * self.focal_tversky_loss(preds, targets)
        )

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)
