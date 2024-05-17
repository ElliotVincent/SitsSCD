"""
Code from
https://github.com/clcarwin/focal_loss_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, x, y):
        """
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
        Returns:
            torch.Tensor: Focal loss between x and y: torch.Tensor([B])
        """
        prediction = x["logits"]
        target = y["gt"]
        if prediction.dim() > 2:
            prediction = prediction.contiguous().view(prediction.size(0)*prediction.size(1), prediction.size(2), -1)  # B,T,K,H,W => B*T,K,H*W
            prediction = prediction.transpose(1, 2)  # N,K,H*W => N,H*W,K
            prediction = prediction.contiguous().view(-1, prediction.size(2))  # N,H*W,K => N*H*W,K
        target = target.contiguous().flatten()

        if self.ignore_index is not None:
            prediction = prediction[target != self.ignore_index]
            target = target[target != self.ignore_index]

        target = target[:, None]
        logpt = F.log_softmax(prediction, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != prediction.data.type():
                self.alpha = self.alpha.type_as(prediction.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


LOSSES = {
    "focal": FocalLoss,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, ignore_index=None):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.ignore_index = ignore_index
        self.init_losses(mix)

    def init_losses(self, mix):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                self.loss[m] = (LOSSES[m](ignore_index=self.ignore_index), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        losses = {n: AVERAGE[average](f(x, y)) for n, (f, _) in self.loss.items()}
        losses["loss"] = sum([losses[n] * w for n, (_, w) in self.loss.items()])
        return losses