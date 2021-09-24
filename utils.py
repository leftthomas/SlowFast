import glob
import os

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.io import read_video


class ConvertTHWCtoTCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ConvertTCHWtoCTHW(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2, 3)


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([ConvertTHWCtoTCHW(), transforms.Resize(112), transforms.RandomCrop(112),
                                   transforms.RandomHorizontalFlip(), transforms.ConvertImageDtype(torch.float32),
                                   transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
                                   ConvertTCHWtoCTHW()])
    else:
        return transforms.Compose([ConvertTHWCtoTCHW(), transforms.Resize(112), transforms.CenterCrop(112),
                                   transforms.ConvertImageDtype(torch.float32),
                                   transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
                                   ConvertTCHWtoCTHW()])


class VideoDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(VideoDataset, self).__init__()

        self.videos = sorted(glob.glob(os.path.join(data_root, data_name, split, '*', '*.mp4')))
        self.transform = get_transform(split)

        self.labels, self.classes = [], {}
        i = 0
        for video in self.videos:
            label = os.path.dirname(video).split('/')[-1]
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])

    def __getitem__(self, index):
        video, audio, info = read_video(self.videos[index])
        video = self.transform(video)
        label = self.labels[index]
        return video, label

    def __len__(self):
        return len(self.videos)
