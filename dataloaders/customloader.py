import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#Should be in transformations.py
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

#Actual dataloader
class AbdominalDataset(Dataset):
    def __init__(self, root_dir, mode='CT_npy', transform=None, applied_types=None, new_size=(256,256)):
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
        data = np.load(file_path)['arr_0']  
        
        if self.transform:
            data = self.transform(image=data)['image']
        
        return torch.tensor(data, dtype=torch.float)

    def _get_file_paths(self):
        file_paths = []
        mode_dir = os.path.join(self.root_dir, self.mode)
        for root, dirs, files in os.walk(mode_dir):
            for file in files:
                if file.endswith('.npz'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

if __name__ == '__main__':
	root_directory = '/content/Dataset/abdominalDATA'
	batch_size = 32
	applied_types = "train" 
	transform = get_transform(applied_types=applied_types, new_size=(256,256))
	mode = 'ct'

	if mode=='ct':
		modepath = 'CT_npy'
	if mode=='mri':
		modepath = 'MR_T2_npy'

	abdominal_dataset = AbdominalDataset(root_directory, mode=modepath, transform=transform)
	dataloader = DataLoader(abdominal_dataset, batch_size=batch_size, shuffle=True, drop_last=True)