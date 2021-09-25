import torch
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, RandomShortSideScale, \
    ShortSideScale, Normalize
from torch import nn
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop

side_size = 256
max_size = 320
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
clip_duration = (num_frames * sampling_rate) / frames_per_second
num_classes = 400


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames):
        fast_pathway = frames
        # perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(frames, 1, torch.linspace(0, frames.shape[1] - 1,
                                                                    frames.shape[1] // self.alpha).long(), )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


train_transform = ApplyTransformToKey(key="video", transform=Compose(
    [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
     RandomShortSideScale(min_size=side_size, max_size=max_size), RandomCrop(crop_size), RandomHorizontalFlip(),
     PackPathway()]))
test_transform = ApplyTransformToKey(key="video", transform=Compose(
    [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
     ShortSideScale(size=side_size), CenterCrop(crop_size), PackPathway()]))
