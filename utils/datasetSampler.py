"""
    This file contains the implementation of a simple dataset class and a sampler wrapper for PyTorch.
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

class SimpleDataset(Dataset):
    """
    A simple Dataset class to load images from a directory and apply transformations.
    
    Args:
        dir_path (str): Directory containing image files.
        transforms (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, dir_path, transforms=None):
        super(SimpleDataset, self).__init__()
        
        # Check if the directory path is valid
        if not os.path.isdir(dir_path):
            raise ValueError(f"The path {dir_path} is not a valid directory.")
        
        self.dir_path = dir_path
        self.img_paths = os.listdir(self.dir_path)
        
        # Use default transform if none provided
        self.transforms = transforms if transforms is not None else T.ToTensor()

    def __getitem__(self, index):
        """Loads and transforms an image from the dataset by index."""
        file_name = self.img_paths[index]
        img_path = os.path.join(self.dir_path, file_name)
        
        # Open image and apply transformations
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        
        return img

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.img_paths)


class InfiniteSampler:
    """
    A generator that returns indices from 0 to n-1, indefinitely in random order.
    
    Args:
        n (int): Total number of items to sample from.
    """
    def __init__(self, n):
        self.n = n
        self.order = np.random.permutation(n)
        self.i = 0

    def __iter__(self):
        """Returns the next random index, cycling through the order indefinitely."""
        while True:
            yield self.order[self.i]
            self.i += 1
            if self.i >= self.n:
                np.random.seed()  # Re-seed and shuffle
                self.order = np.random.permutation(self.n)
                self.i = 0


class InfiniteSamplerWrapper(Sampler):
    """
    A PyTorch Sampler wrapper that uses InfiniteSampler to generate indices.
    
    Args:
        data_source (Dataset): The dataset from which to sample.
    """
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        """Returns an iterator for the sampling process."""
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        """Returns a very large number, as the sampling is infinite."""
        return 2 ** 31  # Large enough number to simulate infinity
