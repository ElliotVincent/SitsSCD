from random import randint
import torch
import torchvision.transforms as transforms


def random_flipv(data, label):
    flip = randint(0, 1)
    if flip:
        if label is not None:
            return transforms.functional.vflip(data), transforms.functional.vflip(label)
        else:
            return transforms.functional.vflip(data), None
    else:
        return data, label


def random_fliph(data, label):
    flip = randint(0, 1)
    if flip:
        if label is not None:
            return transforms.functional.hflip(data), transforms.functional.hflip(label)
        else:
            return transforms.functional.hflip(data), None
    else:
        return data, label


def random_rotate(data, label):
    rotate = randint(0, 3)
    if label is not None:
        return torch.rot90(data, rotate, (2, 3)), torch.rot90(label, rotate, (1, 2))
    else:
        return torch.rot90(data, rotate, (2, 3)), None


def random_crop(data, label, img_size, true_size):
    i = randint(0, true_size - img_size - 1)
    j = randint(0, true_size - (1 + 2 * int(i >= true_size - 3 * img_size)) * img_size - 1)
    data = transforms.functional.crop(data, i, j, img_size, img_size)
    if label is not None:
        label = transforms.functional.crop(label, i, j, img_size, img_size)
        return data, label
    else:
        return data


def random_resize_crop(data, label):
    height, width = data.shape[-2:]
    top, left = randint(0, height//2-1), randint(0, width//2-1)
    new_height, new_width = randint(height//2, height-top), randint(width//2, width-left)
    data = transforms.functional.resized_crop(data,
                                              top=top, left=left, height=new_height, width=new_width,
                                              size=[height, width],
                                              interpolation=transforms.InterpolationMode.BILINEAR,
                                              antialias=True)
    if label is not None:
        label = transforms.functional.resized_crop(label,
                                                   top=top, left=left, height=new_height, width=new_width,
                                                   size=[height, width],
                                                   interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                                                   antialias=True)
        return data, label
    else:
        return data, None
