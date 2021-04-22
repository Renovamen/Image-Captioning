import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Union
import h5py
import json
import os

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Parameters
    ----------
    data_folder : str
        Path to the folder where data files are stored

    data_name : str
        Base name of processed datasets

    split : str
        'train', 'val', 'test'

    transform : Callable, optional
        Image transform pipeline
    """
    def __init__(
        self,
        data_folder: str,
        data_name: str,
        split: str,
        transform: Optional[Callable] = None
    ) -> None:
        self.split = split
        assert self.split in {'train', 'val', 'test'}

        # read images from .hdf5 file
        self.h = h5py.File(os.path.join(data_folder, self.split + '_images_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # captions_per_image
        self.cpi = self.h.attrs['captions_per_image']

        # captions after one-hot encoding
        with open(os.path.join(data_folder, self.split + '_captions_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # lengths of captions
        with open(os.path.join(data_folder, self.split + '_caplength_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # size of dataset
        self.dataset_size = len(self.captions)

    def __getitem__(
        self, i: int
    ) -> Union[
        Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor],
        Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]
    ]:
        # remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'train':
            return img, caption, caplen
        else:
            # for validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self) -> int:
        return self.dataset_size
