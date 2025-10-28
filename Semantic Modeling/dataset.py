import os
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import namedtuple


class CityscapesDataset(Dataset):
  def __init__(self, root_dir: str, mode: str, tf=None):
    self.tf = tf
    self.images_dir = os.path.join(root_dir, 'images', mode)
    self.images_src = [os.path.join(self.images_dir, img_src) for img_src in os.listdir(self.images_dir)]

    self.target_dir = os.path.join(root_dir, 'groundtruth', mode)
    self.targets_src = [os.path.join(self.target_dir, trgt_src) for trgt_src in os.listdir(self.target_dir)]

  def __len__(self):
    return len(self.images_src)

  def __getitem__(self, index):
    image = Image.open(self.images_src[index])
    image = self.tf(image)

    target = Image.open(self.targets_src[index])
    target = np.array(target)
    target[target == 255] = 19
    target = torch.from_numpy(target).long()

    return image, target


def data_loader(rootDir):
  data = CityscapesDataset(rootDir, mode='train', tf=preprocess)
  test_set = CityscapesDataset(rootDir, mode='val', tf=preprocess)

  total_count = len(data)
  train_count = int(0.8 * total_count)
  train_set, val_set = random_split(data, (train_count, total_count - train_count),
          generator=torch.Generator().manual_seed(1))

  train_loader = DataLoader(train_set, batch_size=1,shuffle=True)
  test_loader = DataLoader(test_set, batch_size=1,shuffle=False)
  val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

  return train_loader, test_loader, val_loader
