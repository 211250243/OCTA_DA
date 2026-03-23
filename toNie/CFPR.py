import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--model-file', type=str, default='./logs_train/Domain3/source_model.pth.tar')
parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data_dir', default='E:/dataset/Medical Image Dataset/Fundus/')
parser.add_argument('--dataset', type=str, default='Domain2')
parser.add_argument('--model-source', type=str, default='Domain3')
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--model_ema_rate', type=float, default=0.98)
parser.add_argument('--pseudo_label_threshold', type=float, default=0.82)
parser.add_argument('--mean_loss_calc_bound_ratio', type=float, default=0.2)

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
from dataloaders import fundus_dataloader
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
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from sklearn.mixture import GaussianMixture
from skimage import filters

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

def psl_from_Gauss(features_tea_w, n_clusters=3):
    predictions_tea_w= features_tea_w.cpu().numpy()
    gmm_psl_list = []
    for i in range(len(predictions_tea_w)):
        predictions_reshape = np.reshape(predictions_tea_w[i], (256,-1))
        feature_t = predictions_reshape.T
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(feature_t)
        psl_gmm = gmm.predict(feature_t)
        
        # 确定少数类和多数类
        unique_labels, counts = np.unique(psl_gmm, return_counts=True)
        sorted_indices = np.argsort(counts)
        v0_index = sorted_indices[0]
        v1_index = sorted_indices[1]
        v2_index = sorted_indices[2]
        
        v0 = unique_labels[v0_index] # 少数类
        v1 = unique_labels[v1_index] # 中间类
        v2 = unique_labels[v2_index] # 多数类
        
        psl_tem_cup =  np.where(psl_gmm == v0, 1, np.where((psl_gmm == v1) | (psl_gmm == v2), 0, psl_gmm))
        psl_tem_cup = np.reshape(psl_tem_cup, (128, 128)).T
        psl_tem_cup_zoom = zoom(psl_tem_cup, (4, 4), order=1)
        #psl_tem_disk = np.where(psl_gmm == v2, 0, np.where((psl_gmm == v0) | (psl_gmm == v1), 1, psl_gmm))
        psl_tem_disk = np.where(psl_gmm == v1, 1, np.where((psl_gmm == v0) | (psl_gmm == v2), 0, psl_gmm))
        psl_tem_disk = np.reshape(psl_tem_disk, (128, 128)).T
        psl_tem_disk_zoom = zoom(psl_tem_disk , (4, 4), order=1)
        
        psl_tem = np.stack((psl_tem_cup_zoom, psl_tem_disk_zoom), axis=0)
        
        gmm_psl_list.append(psl_tem)
        #kmeans_psl_list = np.stack((kmeans_psl, psl_kmeans), 2)
    gmm_psl = torch.from_numpy(np.stack(gmm_psl_list, axis=0))
      
    return gmm_psl

def psl_from_kmeans(features_tea_w, n_clusters=3):
    features_tea_w = features_tea_w.cpu().numpy()
    kmeans_psl_list = []
    for i in range(len(features_tea_w)):
        feature_reshape = np.reshape(features_tea_w[i], (256,-1))
        feature_t = feature_reshape.T
        kmeans = KMeans(n_clusters, init='k-means++', random_state=42, n_init=10, tol=0.0001)
        kmeans.fit(feature_t)
        psl_kmeans = kmeans.labels_
        
        # 确定少数类和多数类
        unique_labels, counts = np.unique(psl_kmeans, return_counts=True)
        sorted_indices = np.argsort(counts)
        v0_index = sorted_indices[0]
        v1_index = sorted_indices[1]
        v2_index = sorted_indices[2]
        
        v0 = unique_labels[v0_index] # 少数类/cup
        v1 = unique_labels[v1_index] # 中间类/disc
        v2 = unique_labels[v2_index] # 多数类/bg
        
        psl_tem_cup =  np.where(psl_kmeans == v0, 1, np.where((psl_kmeans == v1) | (psl_kmeans == v2), 0, psl_kmeans))
        psl_tem_cup = np.reshape(psl_tem_cup, (128, 128)).T   
        psl_tem_cup_zoom = zoom(psl_tem_cup, (4, 4), order = 1)
        
        # psl_tem_disc = np.where(psl_kmeans == v2, 0, np.where((psl_kmeans == v0) | (psl_kmeans == v1), 1, psl_kmeans))
        psl_tem_disc = np.where(psl_kmeans == v1, 1, np.where((psl_kmeans == v2), 0, psl_kmeans))
        psl_tem_disc = np.reshape(psl_tem_disc, (128, 128)).T
        psl_tem_disc_zoom = zoom(psl_tem_disc, (4, 4), order = 1)
        
        psl_tem = np.stack((psl_tem_cup_zoom, psl_tem_disc_zoom), axis=0)
        kmeans_psl_list.append(psl_tem)

    kmeans_psl = torch.from_numpy(np.stack(kmeans_psl_list, axis=0))
      
    return kmeans_psl

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

#w: weak, s: strong, stu: student, tea: teacher
def adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=None, epoch = None):
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
                  
        Cluster_psl = psl_from_kmeans(features_tea_w, n_clusters=3).cuda()

        predictions_stu_s_sigmoid = torch.sigmoid(predictions_stu_s)
        predictions_tea_w_sigmoid = torch.sigmoid(predictions_tea_w)

        # get hard pseudo label
        # pseudo_label_threshold = (1 - epoch / 50) * pseudo_label_threshold + (epoch / 50) * torch.max(predictions_stu_s)  / 50
        #pseudo_labels_net = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)
        pseudo_labels_net = torch.zeros_like(predictions_tea_w_sigmoid)
        
        # cup_thresh = 0
        # for i in range(len(predictions_tea_w_sigmoid)):
        #     cup_ps1 = predictions_tea_w_sigmoid[i, 0, :,:].cpu().numpy()
        #     cup_thresh_tem = filters.threshold_otsu(cup_ps1)
        #     # if cup_thresh < cup_thresh_tem:
        #     #     cup_thresh = cup_thresh_tem
        #     # cup_pseudo_labels = cup_ps1 > cup_thresh_tem
        #     print('cup_thresh:', cup_thresh_tem)  
        #     pseudo_labels_net[i,0,:,:]  = seg_soft_label_to_hard(predictions_tea_w_sigmoid[i,0,:,:], pseudo_label_threshold = cup_thresh_tem)
                       
        #     dice_ps1 = predictions_tea_w_sigmoid[i,1,:,:].cpu().numpy()
        #     dice_thresh_tem = filters.threshold_otsu(dice_ps1)
        #     # dice_pseudo_labels = dice_ps1 > dice_thresh_tem
        #     print('dice_thresh:', dice_thresh_tem)  
        #     pseudo_labels_net[i,1,:,:] = seg_soft_label_to_hard(predictions_tea_w_sigmoid[i,1,:,:], pseudo_label_threshold = dice_thresh_tem)
            
        # print('cup_thresh:', cup_thresh)   
        # pseudo_labels_net[:,0,:,:] = soft_label_to_hard(predictions_tea_w_sigmoid[:,0,:,:], pseudo_label_threshold = 0.4)
        # pseudo_labels_net[:,1,:,:] = soft_label_to_hard(predictions_tea_w_sigmoid[:,1,:,:], pseudo_label_threshold = 0.8)
        
        pseudo_labels_net = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)
        pseudo_labels = torch.where(pseudo_labels_net == Cluster_psl, pseudo_labels_net, predictions_tea_w_sigmoid)
        
        # pseudo_labels = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)
        
        # for idx in range(len(img_name)):
        #     file_name = os.path.basename(img_name[idx])
        #     cup_pl = pseudo_labels_net[idx,0,:,:].cpu().numpy()
        #     disc_pl = pseudo_labels_net[idx,1,:,:].cpu().numpy()
        #     np.save(f'E:/paper/Medical/unnamed2/TMI/results/PL/cup/{epoch}/{file_name}.npy', cup_pl)
        #     np.save(f'E:/paper/Medical/unnamed2/TMI/results/PL/disc/{epoch}/{file_name}.npy', disc_pl)
            
        #     cup_fpl = pseudo_labels[idx,0,:,:].cpu().numpy()
        #     disc_fpl = pseudo_labels[idx,1,:,:].cpu().numpy()
        #     np.save(f'E:/paper/Medical/unnamed2/TMI/results/FPL/cup/{epoch}/{file_name}.npy', cup_fpl)
        #     np.save(f'E:/paper/Medical/unnamed2/TMI/results/FPL/disc/{epoch}/{file_name}.npy', disc_fpl)
        
        bceloss = torch.nn.BCELoss(reduction='none')
        loss_seg_pixel = bceloss(predictions_stu_s_sigmoid, pseudo_labels)
        
        mean_loss_weight_mask = torch.ones(pseudo_labels.size()).cuda()
        #0:bg, 1:fg ??
        mean_loss_weight_mask[:, 0, ...][pseudo_labels[:, 0, ...] == 0] = loss_weight
        loss_mask = mean_loss_weight_mask

        loss1 = torch.sum(loss_seg_pixel * loss_mask) / torch.sum(loss_mask)
        
        #add
        # gamma = 1.5
        # factor = torch.zeros_like(predictions_stu_s_sigmoid)
        # for c in range(2):
        #     factor[:, c, :, :] = (1 - predictions_stu_s_sigmoid[:, c, :, :]) ** gamma
        # factor0 = factor[:,0,:,:]
        # factor1 = factor[:,1,:,:]
        
        pred_cup = predictions_stu_s_sigmoid[:,0,:,:]
        cup_gt = pseudo_labels[:,0,:,:]
        m1 = 2. * torch.sum(pred_cup * cup_gt, axis = (1,2))
        m2 = torch.sum(pred_cup, axis = (0,1,2)) + torch.sum(cup_gt, axis = (0,1,2))
        # m1 = 2. * torch.sum(factor0 * pred_cup * cup_gt, axis = (1,2))
        # m2 = torch.sum(factor0 * pred_cup, axis = (0,1,2)) + torch.sum(factor0 * cup_gt, axis = (0,1,2))
        # dice_loss1 = 1 - torch.mean((m1 + 1e-6) / (m2 + 1e-6))
        # dice_loss1 = torch.mean(1 - (m1 + 1e-6) / (m2 + 1e-6))
        dice_loss1 = torch.mean((m1 + 1e-6) / (m2 + 1e-6))
        
        pred_disc = predictions_stu_s_sigmoid[:,1,:,:]
        disc_gt = pseudo_labels[:,1,:,:]
        n1 = 2. * torch.sum(pred_disc * disc_gt, axis = (1,2))
        n2 = torch.sum(pred_disc, axis = (0,1,2)) + torch.sum(disc_gt, axis = (0,1,2))
        # n1 = 2. * torch.sum(factor1 * pred_disc * disc_gt, axis = (1,2))
        # n2 = torch.sum(factor1 * pred_disc, axis = (0,1,2)) + torch.sum(factor1 * disc_gt, axis = (0,1,2))
        # dice_loss2 = 1 - torch.mean((n1 + 1e-6) / (n2 + 1e-6))
        # dice_loss2 = torch.mean(1 - (n1 + 1e-6) / (n2 + 1e-6))
        dice_loss2 = torch.mean((n1 + 1e-6) / (n2 + 1e-6))
        
        cup_num = (cup_gt == 1).sum().item()
        disc_num = (disc_gt == 1).sum().item()
        w1 = (disc_num / (cup_num + disc_num)) ** 0.8
                     
        #w1 = 0.9
        w2 = 0.3
        loss2 = 1 - (w1 * dice_loss1 + (1 - w1) * dice_loss2)
        # loss2 = 1 - (dice_loss1 + dice_loss2)/2
        # loss = w2 * loss1 + (1 - w2) * (w1 * dice_loss1 + (1 - w1) * dice_loss2)
        loss = w2 * loss1 + (1 - w2) * loss2
        #loss = loss1
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        #update teacher
        for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
            param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)
            #param_t.data = param_t.data.clone() * model_ema_rate + param_s.data.clone() * (1.0 - model_ema_rate)

        # update feature/pred bank
        for idx in range(len(img_name)):
            feature_bank[img_name[idx]] = features_tea_w[idx].detach().clone()
            pred_bank[img_name[idx]] = predictions_tea_w_sigmoid[idx].detach().clone()


def eval(model, data_loader):
    model.eval()

    val_dice = {'cup': np.array([]), 'disc': np.array([])}
    val_assd = {'cup': np.array([]), 'disc': np.array([])}

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            data = data.cuda()
            predictions, _ = model(data)
            
            dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
            val_dice['cup'] = np.append(val_dice['cup'], dice_cup)
            val_dice['disc'] = np.append(val_dice['disc'], dice_disc)

            assd = assd_compute(predictions, target_map)
            val_assd['cup'] = np.append(val_assd['cup'], assd[:, 0])
            val_assd['disc'] = np.append(val_assd['disc'], assd[:, 1])

        avg_dice = [0.0, 0.0, 0.0, 0.0]
        std_dice = [0.0, 0.0, 0.0, 0.0]
        avg_assd = [0.0, 0.0, 0.0, 0.0]
        std_assd = [0.0, 0.0, 0.0, 0.0]
        avg_dice[0] = np.mean(val_dice['cup'])
        avg_dice[1] = np.mean(val_dice['disc'])
        std_dice[0] = np.std(val_dice['cup'])
        std_dice[1] = np.std(val_dice['disc'])
        val_assd['cup'] = np.delete(val_assd['cup'], np.where(val_assd['cup'] == -1))
        val_assd['disc'] = np.delete(val_assd['disc'], np.where(val_assd['disc'] == -1))
        avg_assd[0] = np.mean(val_assd['cup'])
        avg_assd[1] = np.mean(val_assd['disc'])
        std_assd[0] = np.std(val_assd['cup'])
        std_assd[1] = np.std(val_assd['disc'])

    model.train()

    return avg_dice, std_dice, avg_assd, std_assd

def eval_disc(model, data_loader, path=None):
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            data = data.cuda()
            predictions, _ = model(data)
            
            for idx in range(len(sample['img_name'])):
                disc_name = os.path.basename(sample['img_name'][idx])
                disc = predictions[idx,1,:,:]         
                disc = torch.sigmoid(disc)
                disc = disc.cpu().numpy()
                disc[disc > 0.5] = 1
                disc[disc < 0.5] = 0
                disc = disc * 255
            
                cv2.imwrite(f'E:/code/Medical Image/CCMT_temporary/results/D2/CF_Dice/{path}/{disc_name}', disc)
    model.train()
                
def eval_cup(model, data_loader, path=None):
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            data = data.cuda()
            predictions, _ = model(data)
            
            for idx in range(len(sample['img_name'])):              
                cup_name = os.path.basename(sample['img_name'][idx])          
                cup = predictions[idx,0,:,:]
                cup = torch.sigmoid(cup)
                cup = cup.cpu().numpy()
                cup[cup > 0.5] = 1
                cup[cup < 0.5] = 0
                cup = cup * 255

                cv2.imwrite(f'E:/code/Medical Image/CCMT_temporary/results/D2/CF_Dice/{path}/{cup_name}', cup)
    model.train()

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
    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.adjust_contrast(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])

    dataset_train = fundus_dataloader.FundusSegmentation_2transform(base_dir=args.data_dir, dataset=args.dataset,
                                                                    split='train/ROIs',
                                                                    transform_weak=composed_transforms_test,
                                                                    transform_strong=composed_transforms_train)
    dataset_train_weak = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                              split='train/ROIs',
                                                              transform=composed_transforms_test)
    dataset_test = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test//ROIs_bad',
                                         transform=composed_transforms_test)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    model_s = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
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
    log_str = ("initial dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
            avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    
   
    # writer = SummaryWriter('./logs')


    for epoch in range(args.epoch):

        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        not_cup_loss_sum = torch.FloatTensor([0]).cuda()
        cup_loss_sum = torch.FloatTensor([0]).cuda()
        not_cup_loss_num = 0
        cup_loss_num = 0
        #lower_bound, upper_bound: Filter predictions that are too samll or too large, i.e. highly confident
        lower_bound = args.pseudo_label_threshold * args.mean_loss_calc_bound_ratio
        upper_bound = 1 - ((1 - args.pseudo_label_threshold) * args.mean_loss_calc_bound_ratio)

        for pred_i in pred_bank.values():
            not_cup_loss_sum += torch.sum(
                -torch.log(1 - pred_i[0, ...][(pred_i[0, ...] < args.pseudo_label_threshold) * (pred_i[0, ...] > lower_bound)]))
            not_cup_loss_num += torch.sum((pred_i[0, ...] < args.pseudo_label_threshold) * (pred_i[0, ...] > lower_bound))
            cup_loss_sum += (torch.sum(-torch.log(pred_i[0, ...][(pred_i[0, ...] > args.pseudo_label_threshold) * (pred_i[0, ...] < upper_bound)])))
            cup_loss_num += torch.sum((pred_i[0, ...] > args.pseudo_label_threshold) * (pred_i[0, ...] < upper_bound))
        loss_weight = (cup_loss_sum.item() / cup_loss_num) / (not_cup_loss_sum.item() / not_cup_loss_num)
      
        #adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=loss_weight, model_ema_rate=model_ema_rate)
        adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=loss_weight, epoch = epoch)

        scheduler.step()

        t_avg_dice, t_std_dice, t_avg_assd, t_std_assd = eval(model_t, test_loader)
        log_str = ("teacher dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            t_avg_dice[0], t_std_dice[0], t_avg_dice[1], t_std_dice[1], (t_avg_dice[0] + t_avg_dice[1]) / 2.0,
            t_avg_assd[0], t_std_assd[0], t_avg_assd[1], t_std_assd[1], (t_avg_assd[0] + t_avg_assd[1]) / 2.0))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        s_avg_dice, s_std_dice, s_avg_assd, s_std_assd = eval(model_s, test_loader)
        log_str = ("student dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
                s_avg_dice[0], s_std_dice[0], s_avg_dice[1], s_std_dice[1], (s_avg_dice[0] + s_avg_dice[1]) / 2.0,
                s_avg_assd[0], s_std_assd[0], s_avg_assd[1], s_std_assd[1], (s_avg_assd[0] + s_avg_assd[1]) / 2.0))
        print(log_str)

        
        
                                                
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
  

    # writer.close()
    torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/after_adaptation.pth.tar')
 
if __name__ == '__main__':
    main()