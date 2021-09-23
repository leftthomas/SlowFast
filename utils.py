import glob
import os
import random

from PIL import Image
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, precision_at_k
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        images = []
        for classes in os.listdir(os.path.join(data_root, data_name, split, 'sketch')):
            sketches = glob.glob(os.path.join(data_root, data_name, split, 'sketch', str(classes), '*.jpg'))
            photos = glob.glob(os.path.join(data_root, data_name, split, 'photo', str(classes), '*.jpg'))
            # only consider the classes which photo images >= 400 for tuberlin dataset
            if len(photos) < 400 and data_name == 'tuberlin' and split == 'val':
                pass
            else:
                images += sketches
                # only append sketches for train
                if split == 'val':
                    images += photos
        self.images = sorted(images)
        self.transform = get_transform(split)

        self.domains, self.labels, self.classes = [], [], {}
        i = 0
        for img in self.images:
            domain, label = os.path.dirname(img).split('/')[-2:]
            self.domains.append(0 if domain == 'photo' else 1)
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])
        # store photos for each class to easy sample for sketch in training period
        if split == 'train':
            self.refs = {}
            for key, value in self.classes.items():
                self.refs[value] = glob.glob(os.path.join(data_root, data_name, split, 'photo', key, '*.jpg'))

        self.split = split

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transform(img)
        label = self.labels[index]
        if self.split == 'val':
            domain = self.domains[index]
            return img, domain, label
        else:
            ref = Image.open(random.choice(self.refs[label]))
            ref = self.transform(ref)
            return img, ref, label

    def __len__(self):
        return len(self.images)


class MetricCalculator(AccuracyCalculator):
    def calculate_precision_at_100(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 100, self.avg_of_avgs, self.label_comparison_fn)

    def calculate_precision_at_200(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(knn_labels, query_labels[:, None], 200, self.avg_of_avgs, self.label_comparison_fn)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_100", "precision_at_200"]


def compute_metric(vectors, domains, labels):
    calculator_200 = MetricCalculator(include=['mean_average_precision', 'precision_at_100', 'precision_at_200'], k=200)
    calculator_all = MetricCalculator(include=['mean_average_precision'])
    acc = {}

    photo_vectors = vectors[domains == 0]
    sketch_vectors = vectors[domains == 1]
    photo_labels = labels[domains == 0]
    sketch_labels = labels[domains == 1]
    map_200 = calculator_200.get_accuracy(sketch_vectors, photo_vectors, sketch_labels, photo_labels, False)
    map_all = calculator_all.get_accuracy(sketch_vectors, photo_vectors, sketch_labels, photo_labels, False)

    acc['P@100'] = map_200['precision_at_100']
    acc['P@200'] = map_200['precision_at_200']
    acc['mAP@200'] = map_200['mean_average_precision']
    acc['mAP@all'] = map_all['mean_average_precision']
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@100'] + acc['P@200'] + acc['mAP@200'] + acc['mAP@all']) / 4
    return acc

