import os
import json

import numpy as np
import SimpleITK as sitk

import torch
import torch.utils.data as data
import torch.nn.functional as F

from Visualisations import CreateSlicers
import matplotlib.pyplot as plt
class NiftiDataset(data.Dataset):
    def __init__(self, root_dir, transform_info_file, size, resize=None):
        self.root_dir = root_dir
        self.resize = resize
        self.size = size
        with open(transform_info_file, 'r') as f:
            self.transform_info = json.load(f)

    def __len__(self):
        return len(self.transform_info)

    def __getitem__(self, idx):
        patient_info = self.transform_info[idx]
#        scout_nii_path = os.path.join(self.root_dir, patient_info['pathOrigScoutnii'])
        scout_nii_path = os.path.join(patient_info['Paths'])
        transform_matrix = np.array(patient_info['A']['AugmentedA']) # transform_matrix = np.array(patient_info['TransformSurvToSA']['A'])

        # Load NIFTI data
        scout_nii = sitk.ReadImage(scout_nii_path)
        scout_data = torch.FloatTensor(sitk.GetArrayFromImage(scout_nii))
        if scout_data.shape[0] > int(self.size[0]):
            scout_data = scout_data[:int(self.size[0]), :, :]
        if scout_data.shape[1] != int(self.size[1]):
            scout_data = scout_data[:, :int(self.size[1]), :int(self.size[1])]
        scout_data = scout_data.unsqueeze(0)
        scout_data = scout_data.unsqueeze(0)
        if self.resize is not None:
            scout_data = F.interpolate(scout_data, scale_factor=(self.resize, self.resize, self.resize))

        # Ground truth values
        ground_truth = torch.FloatTensor(transform_matrix[:3, :3].flatten())

        return scout_data.squeeze(0), ground_truth


#if __name__ == "__main__":
    # Example usage:
#    root_dir = r'C:\Users\42060\Desktop\DP3\dataset'
#    transform_info_file = os.path.join(root_dir, "train_info.json")
#    size = (220, 220, 200)
#    dataset = NiftiDataset(root_dir=root_dir, transform_info_file=transform_info_file, size=size, resize=128/size[0])
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over the dataloader
#    for batch_idx, (data, ground_truth) in enumerate(dataloader):
#        print("Batch:", batch_idx)
#        print("Data shape:", data.shape)
#        print("Ground truth:", ground_truth)
#        print(data[0, 0, :, :, :].shape)
#        fig = CreateSlicers(data[0, 0, :, :, :], manual=True)
#        plt.show()
#        break
        