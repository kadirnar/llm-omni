import random
import warnings
from random import choice

import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.datasets as dset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor


def get_pokemon_dataset():
    """
    Get image-caption pokemon dataset.

    use this as val
    """
    return load_dataset("lambdalabs/pokemon-blip-captions")


def get_coco_dataset(mode='train'):
    """Get COCO dataset from HuggingFace."""
    # Map 'val' to 'validation' for compatibility
    if mode == 'val':
        mode = 'validation'

    dataset = load_dataset("HuggingFaceM4/COCO", split=mode, trust_remote_code=True, cache_dir="./coco2017")
    return HF_COCO_Wrapper(dataset)


class HF_COCO_Wrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.totensor = ToTensor()
        self.resize = Resize((256, 256), antialias=True)
        self.len = len(dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Get and transform items."""
        item = self.dataset[idx]
        image = item['image']

        # Convert to RGB if the image is not already in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.totensor(image)
        #image = self.resize(image)

        # Get the caption from the sentences dictionary
        caption = item['sentences']['raw']

        sample = {'image': image, 'caption': caption}

        return sample
