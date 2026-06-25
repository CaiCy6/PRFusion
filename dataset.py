# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

from sklearn.model_selection import KFold

class Fusion_dataset(Dataset):
    def __init__(self, split, data_dir_vis, data_dir_ir, length=0, fold=0, num_folds=5):
        super(Fusion_dataset, self).__init__()
        assert split in ['train', 'val'], 'split must be "train"|"val"'

        self.filepath_ir = []
        self.filenames_ir = []
        self.filepath_vis = []
        self.filenames_vis = []
        self.length = length
        self.fold = fold
        self.num_folds = num_folds
        self.split = split
        
        
        mri_files = glob.glob(os.path.join(data_dir_vis, "*.png"))
        pet_files = glob.glob(os.path.join(data_dir_ir, "*.png"))
        
        # Sort files for consistency
        mri_files.sort()
        pet_files.sort()

        # Combine the lists
        all_files = list(zip(mri_files, pet_files))

        # Split the data into k-folds for cross-validation
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        all_indices = list(range(len(all_files)))
        split_indices = list(kf.split(all_indices))
        train_indices, val_indices = split_indices[self.fold]

        # Separate the data into training and validation sets
        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]

        # Set the file paths for the current fold
        if self.split == 'train':
            self.filepath_vis = [file[0] for file in train_files]
            self.filepath_ir = [file[1] for file in train_files]
        elif self.split == 'val':
            self.filepath_vis = [file[0] for file in val_files]
            self.filepath_ir = [file[1] for file in val_files]

        # Set the corresponding filenames
        self.filenames_vis = [os.path.basename(f) for f in self.filepath_vis]
        self.filenames_ir = [os.path.basename(f) for f in self.filepath_ir]
        
    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        image_vis = cv2.imread(vis_path)
        image_inf = cv2.imread(ir_path, 0)

        image_vis = (
            np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose(
                (2, 0, 1)
            )
            / 255.0
        )
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)

        name = self.filenames_vis[index]
        return (
            torch.tensor(image_vis),
            torch.tensor(image_ir),
            name,
        )

    def __len__(self):
        return len(self.filepath_ir)
