import argparse
import logging
import os
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from Dataloader.Dataloader_HCI import HCI_dataset
from video_depth_anything.video_depth import VideoDepthAnything
from f_utils.loss import MaskedMSELoss
from f_utils.metric import eval_depth
from f_utils.tools import init_log, ensure_folder_exists

parser = argparse.ArgumentParser(description='Focus Depth Anything for HCI 10')
parser.add_argument('--encoder', default='vits', choices=['vits', 'vitl'])
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--pretrained_from', default="checkpoints/video_depth_anything_vits.pth", type=str)
parser.add_argument('--save_path', default="record/vits_temp/", type=str)


def main():
    args = parser.parse_args()
    ensure_folder_exists(args.save_path)
    # logger
    logger = init_log('global', args.save_path, logging.INFO)
    logger.propagate = 0
    all_args = {**vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    # tensorboard
    ensure_folder_exists(args.save_path + "tensorboard")
    writer = SummaryWriter(args.save_path + "tensorboard")

    cudnn.enabled = True
    cudnn.benchmark = True

    data_root = "/mnt/data1/**********************/PublicDataset/4D-Light-Field-Dataset/HCI_FS_trainval_10.h5"
    train_dataset = HCI_dataset(data_root, 10, "stack_train", "disp_train")
    valid_dataset = HCI_dataset(data_root, 10, "stack_val", "disp_val")
    train_loader = DataLoader(train_dataset, batch_size=args.bs, pin_memory=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, num_workers=4)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model = VideoDepthAnything(**model_configs[args.encoder])
    if args.pretrained_from:
        model.load_state_dict(torch.load(args.pretrained_from, map_location='cpu'), strict=True)
    for name, param in model.named_parameters():
        if 'scratch' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model = nn.DataParallel(model).cuda()
    criterion = MaskedMSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    previous_best = {'MSE': 100, 'RMSE': 100, 'AbsRel': 100, 'SqRel': 100, 'ACC_1': 0, 'ACC_2': 0, 'ACC_3': 0}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_samples = 0
        for i, samples in enumerate(train_loader):
            optimizer.zero_grad()
            FS, gt, focus_dists, mask = samples
            FS = FS.cuda()
            gt = gt.cuda()
            focus_dists = focus_dists.cuda()
            mask = mask.cuda()
            pred = model(FS, focus_dists)
            loss = criterion(pred, gt, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_samples += 1
        writer.add_scalar('train/loss', total_loss / train_samples, epoch)
        model.eval()
        results = {'MSE': torch.tensor([0.0]).cuda(), 'RMSE': torch.tensor([0.0]).cuda(),
                   'AbsRel': torch.tensor([0.0]).cuda(), 'SqRel': torch.tensor([0.0]).cuda(),
                   'ACC_1': torch.tensor([0.0]).cuda(), 'ACC_2': torch.tensor([0.0]).cuda(),
                   'ACC_3': torch.tensor([0.0]).cuda()}
        val_samples = torch.tensor([0.0]).cuda()
        pth_record_bool = False
        for i, samples in enumerate(val_loader):
            FS, gt, focus_dists, mask = samples
            FS = FS.cuda()
            gt = gt.cuda()
            focus_dists = focus_dists.cuda()
            mask = mask.cuda()
            with torch.no_grad():
                pred = model(FS, focus_dists)
            H, W = gt.shape[1:]
            pred = pred[:, :H, :W]
            cur_results = eval_depth(pred, gt, mask)
            for k in results.keys():
                results[k] += cur_results[k]
            val_samples += 1

        logger.info('Epoch: {:}/{:}'.format(epoch, args.epochs))
        logger.info('==========================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}'.format(
            *tuple([(v / val_samples).item() for v in results.values()])))
        logger.info('==========================================================================================')

        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / val_samples).item(), epoch)

        for k in results.keys():
            if k in ['ACC_1', 'ACC_2', 'ACC_3']:
                previous_best[k] = max(previous_best[k], (results[k] / val_samples).item())
            elif k in ['MSE']:
                if (results[k] / val_samples).item() < previous_best['MSE']:
                    pth_record_bool = True
                previous_best[k] = min(previous_best[k], (results[k] / val_samples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / val_samples).item())
        if pth_record_bool:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': previous_best,
            }
            torch.save(checkpoint, args.save_path + "Best.pth")

    logger.info("===== ******* =====")

    for handler in logger.handlers:
        handler.close()


if __name__ == '__main__':
    main()
