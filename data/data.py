import numpy as np
import torch
import json
from os.path import join
import random
import torchvision
from datetime import date
import pandas as pd
from torch.utils.data import Dataset
from .transforms import random_resize_crop, random_rotate, random_crop, random_flipv, random_fliph
import time


class SitsDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 domain_shift,
                 num_channels,
                 num_classes,
                 img_size,
                 true_size,
                 train_length):
        super(SitsDataset, self).__init__()
        self.path = path
        self.image_folder, self.gt_folder = join(path, split if domain_shift else 'train'), join(path, 'labels')
        self.split = split
        self.domain_shift = domain_shift
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.true_size = true_size
        self.train_length = train_length
        self.monthly_dates = get_monthly_dates_dict()
        self.gt, self.sits_ids = self.load_ground_truth(split)
        self.collate_fn = collate_fn
        self.mean, self.std, self.month_list = None, None, None  # Needs to be defined in subclass

    def __len__(self):
        if self.split == 'train':
            return len(self.sits_ids) * ((self.true_size // self.img_size) ** 2 - 4)
        elif self.domain_shift:
            return len(self.sits_ids) * (self.true_size // self.img_size) ** 2
        else:
            return len(self.sits_ids) * 2

    def __getitem__(self, i):
        """Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "data", "gt", "positions", "idx"
        Shapes:
            data: T x C x H x W
            gt: T x H x W
            positions: T
            idx: 1
        """
        if self.split == 'train':
            num_patches_per_sits = (self.true_size // self.img_size) ** 2 - 4
            sits_number = i // num_patches_per_sits
            patch_loc_i, patch_loc_j = None, None
            months = self.get_random_months(sits_number)
        elif self.domain_shift:
            num_patches_per_sits = (self.true_size // self.img_size) ** 2
            sits_number = i // num_patches_per_sits
            patch_loc_i = (i % num_patches_per_sits) // (self.true_size // self.img_size)
            patch_loc_j = (i % num_patches_per_sits) % (self.true_size // self.img_size)
            months = list(range(24))
        else:
            num_patches_per_sits = 2
            sits_number = i // num_patches_per_sits
            patch_loc_i, patch_loc_j = self.get_loc_per_split(i % num_patches_per_sits)
            months = list(range(24))
        sits_id = self.sits_ids[sits_number]
        curr_sits_path = join(self.image_folder, sits_id)
        gt = self.gt[sits_number, months]
        data, days = self.load_data(sits_number, sits_id, months, curr_sits_path)
        data, gt = self.transform(data, gt, patch_loc_i, patch_loc_j)
        data = self.normalize(data)
        positions = torch.tensor(days, dtype=torch.long)
        output = {"data": data, "gt": gt.long(), "positions": positions, "idx": sits_number}
        return output

    def load_ground_truth(self, split):
        """Returns the ground truth label of the given split."""
        start_time = time.time()
        if self.domain_shift:
            sits_ids = json.load(open(join(self.path, 'split.json')))[split]
        else:
            sits_ids = json.load(open(join(self.path, 'split.json')))['train']
        sits_ids.sort()
        num_sits = len(sits_ids)
        gt = torch.zeros((num_sits, 24, 1024, 1024), dtype=torch.int8)
        for sits in range(num_sits):
            gt[sits] = torch.tensor(np.load(join(self.gt_folder, f'{sits_ids[sits]}.npy')), dtype=torch.int8)
        end_time = time.time()
        print(f"Loading {split} ground truth took {(end_time - start_time):.2f} seconds")
        return gt, sits_ids

    def transform(self, data, gt=None, patch_loc_i=None, patch_loc_j=None):
        if self.split == 'train':
            data, gt = random_crop(data, gt, self.img_size, self.true_size)
            data, gt = random_fliph(data, gt)
            data, gt = random_flipv(data, gt)
            data, gt = random_rotate(data, gt)
            data, gt = random_resize_crop(data, gt)
        else:
            data = data[..., patch_loc_i * self.img_size: (patch_loc_i + 1) * self.img_size,
                        patch_loc_j * self.img_size: (patch_loc_j + 1) * self.img_size]
            gt = gt[..., patch_loc_i * self.img_size: (patch_loc_i + 1) * self.img_size,
                    patch_loc_j * self.img_size: (patch_loc_j + 1) * self.img_size]
        return data, gt

    def normalize(self, data):
        return (data - self.mean) / self.std

    def get_loc_per_split(self, i):
        return {'val': [self.true_size // self.img_size - 1 - i, self.true_size // self.img_size - 1 - i],
                'test': [self.true_size // self.img_size - 1 - i, self.true_size // self.img_size - 2 + i]
                }[self.split]

    def get_random_months(self, sits_number):
        months = self.month_list[sits_number]
        random.shuffle(months)
        months = sorted(months[:self.train_length])
        return months

    def load_data(self, sits_number, sits_id, months, curr_sits_path):
        """Needs to be implemented in subclass."""
        data, days = None, None
        return data, days


class Muds(SitsDataset):
    def __init__(
            self,
            path,
            split="train",
            domain_shift=False,
            num_channels=3,
            num_classes=3,
            img_size=128,
            true_size=1024,
            train_length=6,
    ):
        super(Muds, self).__init__(path=path,
                                   split=split,
                                   domain_shift=domain_shift,
                                   num_channels=num_channels,
                                   num_classes=num_classes,
                                   img_size=img_size,
                                   true_size=true_size,
                                   train_length=train_length)
        """Initializes the dataset.
        Args:
            path (str): path to the dataset
            split (str): split to use (train, val, test)
            domain_shift (bool): if val/test, whether we are in a domain shift setting or not
        """
        month_list = [(((self.gt[k].float() - 2) ** 2).mean((1, 2)) == 0).int().numpy().tolist() for k in range(self.gt.shape[0])]
        self.month_list = [[j for j in range(24) if month_list[i][j] == 0] for i in range(len(month_list))]
        self.mean = torch.tensor([119.9347, 105.3608, 77.5125], dtype=torch.float16).reshape(3, 1, 1)
        self.std = torch.tensor([59.5921, 48.2708, 44.7296], dtype=torch.float16).reshape(3, 1, 1)
        self.month_string = [f'{year}_{month}' for year, month in zip(
            [2018 for _ in range(12)] + [2019 for _ in range(12)],
            [f'0{m}' for m in range(1, 10)] + ['10', '11', '12'] + [f'0{m}' for m in range(1, 10)] + ['10', '11',
                                                                                                      '12'])]

    def load_data(self, sits_number, sits_id, months, curr_sits_path):
        data = torch.zeros((len(months), self.num_channels, self.true_size, self.true_size), dtype=torch.float16)
        name_rgb = [f'{sits_id}_{self.month_string[month]}.jpg' for month in months]
        for month_id, month in enumerate(months):
            if month in self.month_list[sits_number]:
                data[month_id] = torchvision.io.read_image(join(curr_sits_path, name_rgb[month_id]))
        days = [self.monthly_dates[month] for month in months]
        return data, days


class DynamicEarthNet(SitsDataset):
    def __init__(
            self,
            path,
            split="train",
            domain_shift=False,
            num_channels=4,
            num_classes=7,
            img_size=128,
            true_size=1024,
            train_length=6,
            date_aug_range=2
    ):

        super(DynamicEarthNet, self).__init__(path=path,
                                              split=split,
                                              domain_shift=domain_shift,
                                              num_channels=num_channels,
                                              num_classes=num_classes,
                                              img_size=img_size,
                                              true_size=true_size,
                                              train_length=train_length)
        """Initializes the dataset.
        Args:
            path (str): path to the dataset
            split (str): split to use (train, val, test)
            domain_shift (bool): if val/test, whether we are in a domain shift setting or not
        """
        self.date_aug_range = date_aug_range
        self.monthly_dates = get_monthly_dates_dict()
        self.gt, self.sits_ids = self.load_ground_truth(split)
        self.month_list = [list(range(24)) for _ in range(self.gt.shape[0])]
        self.mean = torch.tensor([83.1029, 80.7615, 69.3328, 133.8648], dtype=torch.float16).reshape(4, 1, 1)
        self.std = torch.tensor([33.2714, 25.5288, 23.9868, 30.5591], dtype=torch.float16).reshape(4, 1, 1)

    def load_data(self, sits_number, sits_id, months, curr_sits_path):
        data = torch.zeros((len(months), self.num_channels, self.true_size, self.true_size), dtype=torch.float16)
        days = [self.random_date_augmentation(month) for month in months]
        name_rgb = [f'{sits_id}_{day}_rgb.jpeg' for day in days]
        name_infra = [f'{sits_id}_{day}_infra.jpeg' for day in days]
        for d, (n_rgb, n_infra) in enumerate(zip(name_rgb, name_infra)):
            data[d, :3] = torchvision.io.read_image(join(curr_sits_path, n_rgb))
            data[d, 3] = torchvision.io.read_image(join(curr_sits_path, n_infra))
        return data, days

    def random_date_augmentation(self, month):
        if self.split == 'train':
            return max(0, random.randint(0, self.date_aug_range * 2) - self.date_aug_range + self.monthly_dates[month])
        else:
            return self.monthly_dates[month]


def collate_fn(batch):
    """Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "data", "gt", "positions" and "idx"
    Returns:
        dict: dictionary with keys "data", "gt", "positions" and "idx"
    """
    keys = list(batch[0].keys())
    idx = [x["idx"] for x in batch]
    output = {"idx": idx}
    keys.remove("idx")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


def get_monthly_dates_dict():
    s_date = date(2018, 1, 1)
    e_date = date(2019, 12, 31)
    dates_monthly = [f'{year}-{month}-01' for year, month in zip(
        [2018 for _ in range(12)] + [2019 for _ in range(12)],
        [f'0{m}' for m in range(1, 10)] + ['10', '11', '12'] + [f'0{m}' for m in range(1, 10)] + ['10', '11', '12']
    )]
    dates_daily = pd.date_range(s_date, e_date, freq='d').strftime('%Y-%m-%d').tolist()
    monthly_dates = []
    i, j = 0, 0
    while i < 730 and j < 24:
        if dates_monthly[j] == dates_daily[i]:
            monthly_dates.append(i)
            j += 1
        i += 1
    return monthly_dates
