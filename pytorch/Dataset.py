from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from utils import gen_gaussian_noise, calculate_iou, cropping
from environment import warp
from torchvision.transforms import transforms

class TrackData_SL(Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_bboxes = np.loadtxt(os.path.join(root_dir, 'target'))
        self.imgpaths = np.loadtxt(os.path.join(root_dir, 'imglist'), dtype=np.str)

    def __len__(self):
        return self.gt_bboxes.shape[0]

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: patch, target_action, target_label
        '''

        img_path = os.path.join(self.root_dir, self.imgpaths[idx])
        img = Image.open(img_path)
        gt_bbox = self.gt_bboxes[idx]
        noisy_bbox = gen_gaussian_noise(gt_bbox)

        target_label = 0
        if calculate_iou(noisy_bbox, gt_bbox) > 0.7:
            target_label = 1

        # search the best action
        action_iou = np.zeros(10)
        for i in range(10):
            warpped_bbox = noisy_bbox + warp[i]
            action_iou[i] = calculate_iou(warpped_bbox, gt_bbox)

        target_action = np.argmax(action_iou)

        # patch
        patch = cropping(img, noisy_bbox)
        if self.transform:
            patch = self.transform(patch)

        return patch, target_action, target_label


class TrackData_RL(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.gt_bboxes = np.loadtxt(os.path.join(root_dir, 'target'))
        self.imgpaths = np.loadtxt(os.path.join(root_dir, 'imglist'), dtype=np.str)

    def __len__(self):
        return self.gt_bboxes.shape[0]

    def __getitem__(self, idx):
        '''

        :param idx:
        :return: img, gt_box
        '''

        img_path = os.path.join(self.root_dir, self.imgpaths[idx])
        img = Image.open(img_path)
        gt_bbox = self.gt_bboxes[idx]

        return np.array(img), gt_bbox



