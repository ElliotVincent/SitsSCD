import pytorch_lightning as L
from torch.utils.data import DataLoader
import time


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset_out,
        val_dataset_in,
        test_dataset_out,
        test_dataset_in,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val_out": val_dataset_out,
            "val_in": val_dataset_in,
            "test_out": test_dataset_out,
            "test_in": test_dataset_in,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        else:
            return self._builders["train"]().num_classes

    def setup(self, stage=None):
        """Setup the datamodule.
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
        """
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset_out = self._builders["val_out"]()
            self.val_dataset_in = self._builders["val_in"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Out-of-domain val dataset size: {len(self.val_dataset_out)}")
            print(f"In-domain val dataset size: {len(self.val_dataset_in)}")
        else:
            self.test_dataset_out = self._builders["test_out"]()
            self.test_dataset_in = self._builders["test_in"]()
            print(f"Out-of-domain test dataset size: {len(self.test_dataset_out)}")
            print(f"In-domain test dataset size: {len(self.test_dataset_in)}")
        end_time = time.time()
        print(f"Setup took {(end_time - start_time):.2f} seconds")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset_out,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_out.collate_fn,
        ), DataLoader(
            self.val_dataset_in,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_in.collate_fn,
        )
        ]

    def test_dataloader(self):
        return [DataLoader(
            self.test_dataset_out,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_out.collate_fn,
        ), DataLoader(
            self.test_dataset_in,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_in.collate_fn,
        )
        ]
