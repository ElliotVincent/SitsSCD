import pytorch_lightning as L
from torch.utils.data import DataLoader
import time


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset_domain_shift,
        val_dataset_no_domain_shift,
        test_dataset_domain_shift,
        test_dataset_no_domain_shift,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val_domain_shift": val_dataset_domain_shift,
            "val_no_domain_shift": val_dataset_no_domain_shift,
            "test_domain_shift": test_dataset_domain_shift,
            "test_no_domain_shift": test_dataset_no_domain_shift,
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
            self.val_dataset_shift = self._builders["val_domain_shift"]()
            self.val_dataset_no_shift = self._builders["val_no_domain_shift"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset with domain shift size: {len(self.val_dataset_shift)}")
            print(f"Val dataset without domain shift size: {len(self.val_dataset_no_shift)}")
        else:
            self.test_dataset_shift = self._builders["test_domain_shift"]()
            self.test_dataset_no_shif = self._builders["test_no_domain_shift"]()
            print(f"Test dataset with domain shift size: {len(self.test_dataset_shift)}")
            print(f"Test dataset without domain shiftsize: {len(self.test_dataset_no_shif)}")
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
            self.val_dataset_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_shift.collate_fn,
        ), DataLoader(
            self.val_dataset_no_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset_no_shift.collate_fn,
        )
        ]

    def test_dataloader(self):
        return [DataLoader(
            self.test_dataset_shift,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_shift.collate_fn,
        ), DataLoader(
            self.test_dataset_no_shif,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset_no_shif.collate_fn,
        )
        ]
