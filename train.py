import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorchvideo.data import RandomClipSampler, UniformClipSampler, labeled_video_dataset
from pytorchvideo.models import create_slowfast
from pytorchvideo.transforms import create_video_transform
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(model, data_loader, train_optimizer):
    model.train()
    total_loss, total_acc, total_num, train_bar = 0.0, 0, 0, tqdm(data_loader, dynamic_ncols=True)
    for batch in train_bar:
        video, label = batch['video'].cuda(), batch['label'].cuda()
        train_optimizer.zero_grad()
        pred = model(video)
        loss = loss_criterion(pred, label)
        total_loss += loss.item() * video.size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum()
        loss.backward()
        train_optimizer.step()

        total_num += video.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.4f}'
                                  .format(epoch, epochs, total_loss / total_num, total_acc / total_num))

    return total_loss / total_num


# val for one epoch
def val(backbone, encoder, data_loader):
    backbone.eval()
    encoder.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            img = img.cuda()
            photo = img[domain == 0]
            sketch = img[domain == 1]
            if photo.size(0) != 0:
                photo_emb = backbone(photo)
            if sketch.size(0) != 0:
                sketch_emb = backbone(encoder(sketch))
            if photo.size(0) == 0:
                emb = sketch_emb
            if sketch.size(0) == 0:
                emb = photo_emb
            if photo.size(0) != 0 and sketch.size(0) != 0:
                emb = torch.cat((photo_emb, sketch_emb), dim=0)
            vectors.append(emb.cpu())
            photo_label = label[domain == 0]
            sketch_label = label[domain == 1]
            label = torch.cat((photo_label, sketch_label), dim=0)
            labels.append(label)
            photo_domain = domain[domain == 0]
            sketch_domain = domain[domain == 1]
            domain = torch.cat((photo_domain, sketch_domain), dim=0)
            domains.append(domain)
        vectors = torch.cat(vectors, dim=0)
        domains = torch.cat(domains, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = compute_metric(vectors, domains, labels)
        results['P@100'].append(acc['P@100'] * 100)
        results['P@200'].append(acc['P@200'] * 100)
        results['mAP@200'].append(acc['mAP@200'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@100:{:.1f}% | P@200:{:.1f}% | mAP@200:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@100'] * 100, acc['P@200'] * 100, acc['mAP@200'] * 100,
                      acc['mAP@all'] * 100))
    return acc['precise'], vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of videos in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, batch_size, epochs, save_root = args.data_root, args.batch_size, args.epochs, args.save_root

    # data prepare
    train_transform = create_video_transform(mode='train', video_key='video', num_samples=8,
                                             convert_to_float=False, video_mean=(114.75, 114.75, 114.75),
                                             video_std=(57.375, 57.375, 57.375))
    test_transform = create_video_transform(mode='val', video_key='video', num_samples=8,
                                            convert_to_float=False, video_mean=(114.75, 114.75, 114.75),
                                            video_std=(57.375, 57.375, 57.375))
    train_data = labeled_video_dataset('{}/train'.format(data_root), RandomClipSampler(clip_duration=2),
                                       transform=train_transform, decode_audio=False)
    test_data = labeled_video_dataset('{}/test'.format(data_root), UniformClipSampler(clip_duration=2),
                                      transform=test_transform, decode_audio=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    # model define, loss setup and optimizer config
    slow_fast = create_slowfast(model_num_class=5).cuda()
    loss_criterion = CrossEntropyLoss()
    optimizer = Adam(slow_fast.parameters(), lr=1e-1)

    # training loop
    results = {'loss': [], 'acc': [], 'top-1': [], 'top-5': []}
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(slow_fast, train_loader, optimizer)
        results['loss'].append(train_loss)
        results['acc'].append(train_acc * 100)
        top_1, top_5 = val(slow_fast, test_loader)
        results['top-1'].append(top_1 * 100)
        results['top-5'].append(top_5 * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/metrics.csv'.format(save_root), index_label='epoch')

        if top_1 > best_acc:
            best_acc = top_1
            torch.save(slow_fast.state_dict(), '{}/slow_fast.pth'.format(save_root))
