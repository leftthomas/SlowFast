import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.models import create_slowfast
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import train_transform, test_transform, clip_duration, num_classes

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(model, data_loader, train_optimizer):
    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    train_bar = tqdm(data_loader, total=math.ceil(train_data.num_videos / batch_size), dynamic_ncols=True)
    for batch in train_bar:
        video, label = [i.cuda() for i in batch['video']], batch['label'].cuda()
        train_optimizer.zero_grad()
        pred = model(video)
        loss = loss_criterion(pred, label)
        total_loss += loss.item() * video[0].size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        loss.backward()
        train_optimizer.step()

        total_num += video[0].size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                  .format(epoch, epochs, total_loss / total_num, total_acc * 100 / total_num))

    return total_loss / total_num, total_acc / total_num


# test for one epoch
def val(model, data_loader):
    model.eval()
    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        test_bar = tqdm(data_loader, total=math.ceil(test_data.num_videos / batch_size), dynamic_ncols=True)
        for batch in test_bar:
            video, label = [i.cuda() for i in batch['video']], batch['label'].cuda()
            pred = model(video)
            total_top_1 += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
            total_top_5 += torch.any(torch.eq(pred.topk(k=5, dim=-1).indices, label.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)
            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, epochs, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))
    return total_top_1 / total_num, total_top_5 / total_num


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
    train_data = labeled_video_dataset('{}/train'.format(data_root), make_clip_sampler('random', clip_duration),
                                       transform=train_transform, decode_audio=False)
    test_data = labeled_video_dataset('{}/test'.format(data_root),
                                      make_clip_sampler('constant_clips_per_video', clip_duration, 1),
                                      transform=test_transform, decode_audio=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    # model define, loss setup and optimizer config
    slow_fast = create_slowfast(model_num_class=num_classes).cuda()
    # slow_fast = torch.hub.load('facebookresearch/pytorchvideo:main', model='slowfast_r50', pretrained=True)
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
