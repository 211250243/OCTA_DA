import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--model-file', type=str, default='./logs_train/Domain1/20260325_094606.319125/checkpoint_200.pth.tar')
parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--lr-decrease-rate', type=float, default=0.95, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='../framework/datasets/OCTA-500')
parser.add_argument('--dataset', type=str, default='Domain2')
parser.add_argument('--model-source', type=str, default='Domain1')
parser.add_argument('--batch-size', type=int, default=8)

parser.add_argument('--model-ema-rate', type=float, default=0.999)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.5)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from dataloaders import octa_dataloader
from dataloaders import custom_transforms as trans
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import cv2
import torch.backends.cudnn as cudnn
import random
import glob
import sys

seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels


def init_feature_pred_bank(model, loader):
    feature_bank = {}
    pred_bank = {}

    model.eval()

    with torch.no_grad():
        for sample in loader:
            data = sample['image']
            img_name = sample['img_name']
            data = data.cuda()

            pred, feat = model(data)
            pred = torch.sigmoid(pred)

            for i in range(data.size(0)):
                feature_bank[img_name[i]] = feat[i].detach().clone()
                pred_bank[img_name[i]] = pred[i].detach().clone()

    model.train()

    return feature_bank, pred_bank


def adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=None):
    model_t.eval()
    model_s.train()
    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']
        img_name = sample_w['img_name']
        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()

        # model predict
        predictions_stu_s, features_stu_s = model_s(imgs_s)
        with torch.no_grad():
            predictions_tea_w, features_tea_w = model_t(imgs_w)

        predictions_stu_s_sigmoid = torch.sigmoid(predictions_stu_s)
        predictions_tea_w_sigmoid = torch.sigmoid(predictions_tea_w)

        # get hard pseudo label
        pseudo_labels = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)

        # 置信度掩码：只在 Teacher 高置信度区域（>0.8 或 <0.2）计算 loss
        confidence_mask = ((predictions_tea_w_sigmoid > 0.8) | (predictions_tea_w_sigmoid < 0.2)).float()

        # 如果整个 batch 没有高置信度像素，退化为全局 loss
        if confidence_mask.sum() < 10:
            confidence_mask = torch.ones_like(confidence_mask)

        # BCE with confidence mask
        bceloss_fn = torch.nn.BCELoss(reduction='none')
        loss_bce_pixel = bceloss_fn(predictions_stu_s_sigmoid, pseudo_labels)
        loss_bce = (loss_bce_pixel * confidence_mask).sum() / confidence_mask.sum()

        # Dice Loss on confident region only
        masked_pred = predictions_stu_s_sigmoid * confidence_mask
        masked_target = pseudo_labels * confidence_mask
        loss_dice = DiceLoss(masked_pred, masked_target)

        loss = loss_bce + loss_dice

        loss.backward()
        optim.step()
        optim.zero_grad()

        # update teacher
        for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
            param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)

        # update feature/pred bank
        for idx in range(len(img_name)):
            feature_bank[img_name[idx]] = features_tea_w[idx].detach().clone()
            pred_bank[img_name[idx]] = predictions_tea_w_sigmoid[idx].detach().clone()


def eval(model, data_loader, dice_threshold=0.5):
    model.eval()

    val_dice = []
    val_assd = np.array([])

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            data = data.cuda()
            predictions, _ = model(data)
            dice = dice_coeff_binary(predictions, target_map, threshold=dice_threshold)
            val_dice.append(dice)

            assd = assd_compute(predictions, target_map, threshold=dice_threshold)
            val_assd = np.append(val_assd, assd[:, 0])

        avg_dice = float(np.mean(val_dice))
        std_dice = float(np.std(val_dice))
        val_assd = np.delete(val_assd, np.where(val_assd == -1))
        if val_assd.size == 0:
            avg_assd = float('nan')
            std_assd = float('nan')
        else:
            avg_assd = float(np.mean(val_assd))
            std_assd = float(np.std(val_assd))

    model.train()

    return avg_dice, std_dice, avg_assd, std_assd


def main():
    now = datetime.now()
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, now.strftime('%Y%m%d_%H%M%S.%f')+'.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    # dataset
    # 弱增强（给 Teacher）：仅仅改变大小，归一化
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])

    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.add_salt_pepper_noise(amount=0.01, prob=0.7),
        trans.GaussianNoise(std=15, prob=0.7),
        trans.eraser(),
        trans.adjust_light(),
        trans.NormalizeOCTA(),
        trans.ToTensorOCTA()
    ])

    split_train = 'train'
    split_test = 'value'

    dataset_train = octa_dataloader.OCTASegmentation_2transform(base_dir=args.data_dir, dataset=args.dataset,
                                                                split=split_train,
                                                                transform_weak=composed_transforms_test,
                                                                transform_strong=composed_transforms_train)
    dataset_train_weak = octa_dataloader.OCTASegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                          split=split_train,
                                                          transform=composed_transforms_test)
    dataset_test = octa_dataloader.OCTASegmentation(base_dir=args.data_dir, dataset=args.dataset, split=split_test,
                                     transform=composed_transforms_test)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    model_s = netd.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=1, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)


    if torch.cuda.is_available():
        model_s = model_s.cuda()
        model_t = model_t.cuda()
    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_s.load_state_dict(checkpoint['model_state_dict'])
    model_t.load_state_dict(checkpoint['model_state_dict'])

    if (args.gpu).find(',') != -1:
        model_s = torch.nn.DataParallel(model_s, device_ids=[0, 1])
        model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False


    feature_bank, pred_bank = init_feature_pred_bank(model_s, train_loader_weak)

    avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
    log_str = ("initial dice: %.4f+-%.4f, assd: %.4f+-%.4f" % (avg_dice, std_dice, avg_assd, std_assd))
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    best_dice = 0.0
    patience_counter = 0
    patience = 6

    for epoch in range(args.epoch):

        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=1.0)

        scheduler.step()

        avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
        log_str = ("teacher dice: %.4f+-%.4f, assd: %.4f+-%.4f" % (avg_dice, std_dice, avg_assd, std_assd))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        avg_dice_s, std_dice_s, avg_assd_s, std_assd_s = eval(model_s, test_loader)
        log_str = ("student dice: %.4f+-%.4f, assd: %.4f+-%.4f" % (avg_dice_s, std_dice_s, avg_assd_s, std_assd_s))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        if avg_dice > best_dice:
            best_dice = avg_dice
            patience_counter = 0
            torch.save({'model_state_dict': model_t.state_dict()},
                       args.out + '/best_teacher.pth.tar')
            log_str = "  => best teacher saved (dice=%.4f)" % best_dice
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_str = "  => early stopping at epoch %d (best dice=%.4f)" % (epoch+1, best_dice)
                print(log_str)
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                break

    torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/after_adaptation.pth.tar')


if __name__ == '__main__':
    main()

