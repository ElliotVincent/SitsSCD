import pytorch_lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate


class SitsScdModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.network.instance)
        self.loss = instantiate(cfg.loss.instance)
        self.ignore_index = self.loss.ignore_index
        self.val_metrics = {0: instantiate(cfg.val_metrics), 1: instantiate(cfg.val_metrics)}
        self.test_metrics = {0: instantiate(cfg.test_metrics), 1: instantiate(cfg.test_metrics)}

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = self.loss(pred, batch, average=True)
        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        loss = self.loss(pred, batch, average=True)["loss"]
        self.val_metrics[dataloader_idx].update(pred["pred"], batch["gt"])
        self.log("val/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        for dataloader_idx in range(2):
            metrics = self.val_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"val/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx):
        pred = self.model(batch)
        pred["pred"] = torch.argmax(pred["logits"], dim=2)
        self.test_metrics[dataloader_idx].update(pred["pred"], batch["gt"])

    def on_test_epoch_end(self):
        for dataloader_idx in range(2):
            metrics = self.test_metrics[dataloader_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"test/{metric_name}_{dataloader_idx}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

    def configure_optimizers(self):
        if self.cfg.optimizer.exclude_ln_and_biases_from_weight_decay:
            parameters_names_wd = get_parameter_names(self.model, [nn.LayerNorm])
            parameters_names_wd = [
                name for name in parameters_names_wd if "bias" not in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in parameters_names_wd
                    ],
                    "weight_decay": self.cfg.optimizer.optim.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in parameters_names_wd
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = instantiate(
                self.cfg.optimizer.optim, optimizer_grouped_parameters
            )
        else:
            optimizer = instantiate(self.cfg.optimizer.optim, self.model.parameters())
        scheduler = instantiate(self.cfg.lr_scheduler)(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.global_step)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    Taken from HuggingFace transformers.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result