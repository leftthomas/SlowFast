import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Extractor, Discriminator, Generator
from utils import DomainDataset, compute_metric

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(backbone, data_loader, train_optimizer):
    backbone.train()
    generator.train()
    discriminator.train()
    total_extractor_loss, total_generator_loss, total_discriminator_loss = 0.0, 0.0, 0.0
    total_num, train_bar = 0, tqdm(data_loader, dynamic_ncols=True)
    for sketch, photo, label in train_bar:
        sketch, photo, label = sketch.cuda(), photo.cuda(), label.cuda()

        # generator #
        optimizer_generator.zero_grad()
        fake = generator(sketch)
        pred_fake = discriminator(fake)

        # generator loss
        target_fake = torch.ones(pred_fake.size(), device=pred_fake.device)
        generators_loss = adversarial_criterion(pred_fake, target_fake)
        total_generator_loss += generators_loss.item() * sketch.size(0)

        # extractor #
        train_optimizer.zero_grad()
        photo_proj = backbone(photo)
        sketch_proj = backbone(fake)

        # extractor loss
        class_loss = class_criterion(photo_proj, label) + class_criterion(sketch_proj, label)
        total_extractor_loss += class_loss.item() * sketch.size(0)

        loss = generators_loss + class_loss
        loss.backward()
        train_optimizer.step()
        optimizer_generator.step()

        # discriminator loss #
        optimizer_discriminator.zero_grad()
        pred_real = discriminator(photo)
        target_real = torch.ones(pred_real.size(), device=pred_real.device)
        pred_fake = discriminator(fake.detach())
        target_fake = torch.zeros(pred_fake.size(), device=pred_fake.device)
        adversarial_loss = adversarial_criterion(pred_real, target_real) + adversarial_criterion(pred_fake, target_fake)
        adversarial_loss.backward()
        optimizer_discriminator.step()
        total_discriminator_loss += adversarial_loss.item() * photo.size(0)

        total_num += sketch.size(0)

        e_loss = total_extractor_loss / total_num
        g_loss = total_generator_loss / total_num
        d_loss = total_discriminator_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] E-Loss: {:.4f} G-Loss: {:.4f} D-Loss: {:.4f}'
                                  .format(epoch, epochs, e_loss, g_loss, d_loss))

    return e_loss, g_loss, d_loss


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
    parser.add_argument('--data_root', default='/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--warmup', default=1, type=int, help='Number of warmups over the extractor to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, emb_dim = args.data_root, args.data_name, args.backbone_type, args.emb_dim
    batch_size, epochs, warmup, save_root = args.batch_size, args.epochs, args.warmup, args.save_root

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size // 2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model define
    extractor = Extractor(backbone_type, emb_dim).cuda()
    generator = Generator(in_channels=8).cuda()
    discriminator = Discriminator(in_channels=8).cuda()

    # loss setup
    class_criterion = NormalizedSoftmaxLoss(len(train_data.classes), emb_dim).cuda()
    adversarial_criterion = nn.MSELoss()
    # optimizer config
    optimizer_extractor = Adam([{'params': extractor.parameters()}, {'params': class_criterion.parameters(),
                                                                     'lr': 1e-1}], lr=1e-5)
    optimizer_generator = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # training loop
    results = {'extractor_loss': [], 'generator_loss': [], 'discriminator_loss': [], 'precise': [],
               'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):

        # warmup, not update the parameters of extractor, except the final fc layer
        for param in list(extractor.backbone.parameters())[:-2]:
            param.requires_grad = False if epoch <= warmup else True

        extractor_loss, generator_loss, discriminator_loss = train(extractor, train_loader, optimizer_extractor)
        results['extractor_loss'].append(extractor_loss)
        results['generator_loss'].append(generator_loss)
        results['discriminator_loss'].append(discriminator_loss)
        precise, features = val(extractor, generator, val_loader)
        results['precise'].append(precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if precise > best_precise:
            best_precise = precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))
            torch.save(generator.state_dict(), '{}/{}_generator.pth'.format(save_root, save_name_pre))
            torch.save(discriminator.state_dict(), '{}/{}_discriminator.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
