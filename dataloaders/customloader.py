from torch.utils.data import DataLoader
from losses import MultiClassDiceLoss,PixelPrototypeCELoss

import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2


class AbdominalDataset(Dataset):
    def __init__(self, root_dir, mode='ct', transform=None, applied_types=None, new_size=(256,256)):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.applied_types = applied_types
        self.new_size = new_size
        self.file_paths = self._get_file_paths()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
      file_path = self.file_paths[idx]
      data = np.load(file_path)
      img = data['arr_0'].astype(np.float32)
      seg = data['arr_1']

      if self.transform:
          transformed = self.transform(image=img, mask=seg)
          img = transformed['image']


      img = torch.tensor(img, dtype=torch.float)
      seg = torch.tensor(seg, dtype=torch.float)
      # seg = seg.unsqueeze(0)
      seg = seg.to(torch.long)
      return img, seg


    def _get_file_paths(self):
        file_paths = []
        mode_dir = os.path.join(self.root_dir, self.mode)
        for root, dirs, files in os.walk(mode_dir):
            for file in files:
                if file.endswith('.npz'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

def get_transform(applied_types=None, new_size=(256, 256)):
    if applied_types is None:
        data_transforms = albumentations.Compose([
            albumentations.Resize(new_size[0], new_size[1]),
            ToTensorV2()
        ])
    elif applied_types == "train":
        data_transforms = albumentations.Compose([
            albumentations.Resize(new_size[0], new_size[1]),
            albumentations.RandomResizedCrop(height=new_size[0], width=new_size[1], scale=(0.9, 1.1), ratio=(0.9, 1.1), p=0.25),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=15, p=0.25),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
            albumentations.RandomGamma(gamma_limit=(60, 140), p=0.25),
            ToTensorV2()
        ])
    elif applied_types == "val" or applied_types == "test":
        data_transforms = albumentations.Compose([
            albumentations.Resize(new_size[0], new_size[1]),
            ToTensorV2()
        ])
    return data_transforms