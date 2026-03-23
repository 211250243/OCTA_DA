import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from tqdm import tqdm
from loaddata import BMPDataset

# from models.DGNet import *
# from models.PIENet import *
# from models.PIENet import *
# from models.frnet import *
from models.UNet import *
# from models.CSNet import *
# from models.OCT2former import*
# from models.utnet import *
# from models.CEnet import *
# from models.swinuent import *
# from models.Vesselnet import *

from dice import DiceLoss
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from datetime import datetime,timedelta
import random
import os

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        """
        Args:
            patience (int): 验证集损失不再下降时等待的epoch数
            delta (float): 被视为有改善的最小变化量
            verbose (bool): 如果为True，打印早停信息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model,modelname, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,modelname, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, modelname, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, epoch):
        """保存当前最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), f"./{modelname}_temp_checkpoint.pth")
        else:
            torch.save(model.state_dict(),f"./{modelname}_temp_checkpoint.pth")
        self.val_loss_min = val_loss
        self.best_epoch = epoch

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            # images = images.repeat(1, 3, 1, 1)
            images = images.to(device)
            labels = labels.to(device)
            # labels_down1 = torch.max_pool2d(labels, kernel_size=2,stride=2)
            # labels_down2 = torch.max_pool2d(labels, kernel_size=4,stride=4)
            outputs = model(images)
            # out1, out2, out3, out4 = model(images)
            # outputs = model(images)
            criterion1 = DiceLoss().cuda()
            criterion2 = nn.BCELoss().cuda()
            # criterion2 = nn.MSELoss().cuda()
            # loss = criterion2(out1, labels) + criterion2(out2, labels) + criterion2(out3, labels_down1)/3 + 2*criterion2(out4, labels_down2)/3
            loss =  criterion1(outputs, labels) + criterion2(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def calculate_metrics(model, test_loader, device, threshold=0.5,crop_size = None):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            # images = images.repeat(1, 3, 1, 1)
            images = images.to(device)
            labels = labels.to(device)
            # out1, out2, out3, out4 = model(images)
            outputs = model(images)
            probs = outputs
              # 二值化 [B, 1, H, W]

            if crop_size is not None:
                B,C,H1,W1 = images.shape
                H, W = crop_size
                pad_height = max(H1 - H, 0)
                pad_width = max(W1 - W, 0)  
                h_start = pad_height//2
                w_start = pad_width//2
                probs = probs[:, :, h_start:H+ h_start, w_start:W+w_start]
                labels = labels[:, :, h_start:H+ h_start, w_start:W+w_start]
            
            preds_binary = probs > threshold

            # 展平
            labels_flat = labels.flatten().cpu().numpy()
            probs_flat = probs.flatten().cpu().numpy()
            preds_flat = preds_binary.flatten().cpu().numpy()

            # 统计 TP, FP, TN, FN
            tp = np.sum((preds_flat == 1) & (labels_flat == 1))
            fp = np.sum((preds_flat == 1) & (labels_flat == 0))
            tn = np.sum((preds_flat == 0) & (labels_flat == 0))
            fn = np.sum((preds_flat == 0) & (labels_flat == 1))

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

            # 收集所有预测概率和标签
            all_preds.append(probs_flat)
            all_labels.append(labels_flat)

    # 拼接成一维数组
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels).astype(int)  # 保证是 int 类型

    # 计算各类指标
    eps = 1e-8
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    sen = total_tp / (total_tp + total_fn + eps)
    spe = total_tn / (total_tn + total_fp + eps)
    fdr = total_fp / (total_tp + total_fp + eps)
    acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn + eps)

    # AUC 与 Kappa
    auc = roc_auc_score(all_labels, all_preds)
    preds_binary_for_kappa = (all_preds > threshold).astype(int)
    kappa = cohen_kappa_score(all_labels, preds_binary_for_kappa)

    metrics = {
        'Dice': dice,
        'Sen': sen,
        'Spe': spe,
        'FDR': fdr,
        'Acc': acc,
        'AUC': auc,
        'Kappa': kappa,
        'TP': total_tp,
        'FP': total_fp,
        'TN': total_tn,
        'FN': total_fn
    }

    return metrics


def train(model, train_loader, val_loader, test_loader, optimizer, device, num_epochs, modelname, scheduler=None, patience=10):
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    criterion1 = DiceLoss().cuda()
    criterion2 = nn.BCELoss().cuda()
    # criterion2 = nn.MSELoss().cuda()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch_idx, (images, labels, filenames) in loop:
            # images = images.repeat(1, 3, 1, 1)
            images = images.to(device)
            
            labels = labels.to(device)
            # labels_down1 = torch.max_pool2d(labels, kernel_size=2,stride=2)
            # labels_down2 = torch.max_pool2d(labels, kernel_size=4,stride=4)
            outputs = model(images)
            # out1, out2, out3, out4 = model(images)
            # loss = criterion2(out1, labels) + criterion2(out2, labels) + criterion2(out3, labels_down1)/3 + 2*criterion2(out4, labels_down2)/3
            loss =  criterion1(outputs, labels) + criterion2(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

        if scheduler is not None:
            scheduler.step()

        early_stopping(avg_val_loss, model, modelname, epoch+1)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            print(f"Best model was at epoch {early_stopping.best_epoch} with val loss {early_stopping.val_loss_min:.4f}")
            break

    # 加载最佳模型（自动适配单卡 / 多卡）
    checkpoint = torch.load(f'{modelname}_temp_checkpoint.pth')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:   
        model.load_state_dict(checkpoint)

    # 删除临时文件
    if os.path.exists(f'{modelname}_temp_checkpoint.pth'):
        os.remove(f'{modelname}_temp_checkpoint.pth')
        print(f"临时 checkpoint 已加载并删除: {modelname}_temp_checkpoint.pth")
    else:
        print(f"警告: 临时 checkpoint 文件不存在: {modelname}_temp_checkpoint.pth")
    
    # 训练全部完成后再评估测试集
    if test_loader is not None:
        print("\nEvaluating best model on test set...")
        test_metrics = calculate_metrics(model, test_loader, device)

        result_str = f"\nTest Set Metrics (Best Model from Epoch {early_stopping.best_epoch}):\n"
        result_str += f"Dice: {test_metrics['Dice']:.6f}\n"
        result_str += f"Sensitivity: {test_metrics['Sen']:.6f}\n"
        result_str += f"Specificity: {test_metrics['Spe']:.6f}\n"
        result_str += f"FDR: {test_metrics['FDR']:.6f}\n"
        result_str += f"Accuracy: {test_metrics['Acc']:.6f}\n"
        result_str += f"AUC: {test_metrics['AUC']:.6f}\n"
        result_str += f"Kappa: {test_metrics['Kappa']:.6f}\n"
        result_str += f"Confusion Matrix: TP={test_metrics['TP']}, FP={test_metrics['FP']}, TN={test_metrics['TN']}, FN={test_metrics['FN']}\n"

        print(result_str)

        os.makedirs('./results', exist_ok=True)
        with open(f'./results/{modelname}_test_results.txt', 'a') as f:
            f.write(result_str)

        print(f"Test results saved to './results/{modelname}_test_results.txt'")

    return model

    


if __name__ == "__main__":
    # 参数设置
    
    # dataname = '3mm'
    # image_dir = "./datasets/OCTA-500/3mm/train/images"
    # label_dir = "./datasets/OCTA-500/3mm/train/labels"
    # val_image_dir = "./datasets/OCTA-500/3mm/value/images"  # 验证集图像路径
    # val_label_dir = "./datasets/OCTA-500/3mm/value/labels"  # 验证集标签路径
    # test_image_dir = "./datasets/OCTA-500/3mm/test/images"  # 测试集图像路径
    # test_label_dir = "./datasets/OCTA-500/3mm/test/labels"  # 测试集标签路径

    dataname = '6mm'
    image_dir = "./datasets/OCTA-500/6mm/train/images"
    label_dir = "./datasets/OCTA-500/6mm/train/labels"
    val_image_dir = "./datasets/OCTA-500/6mm/value/images"  # 验证集图像路径
    val_label_dir = "./datasets/OCTA-500/6mm/value/labels"  # 验证集标签路径
    test_image_dir = "./datasets/OCTA-500/6mm/test/images"  # 测试集图像路径
    test_label_dir = "./datasets/OCTA-500/6mm/test/labels"  # 测试集标签路径

    # dataname = 'ROSE'
    # image_dir = "./datasets/ROSE/train/images"
    # label_dir = "./datasets/ROSE/train/labels"
    # val_image_dir = "./datasets/ROSE/value/images"  # 验证集图像路径
    # val_label_dir = "./datasets/ROSE/value/labels"  # 验证集标签路径
    # test_image_dir = "./datasets/ROSE/test/images"  # 测试集图像路径
    # test_label_dir = "./datasets/ROSE/test/labels"  # 测试集标签路径

    # dataname = 'ROSSA'
    # image_dir = "./datasets/ROSSA/train/images"
    # label_dir = "./datasets/ROSSA/train/labels"
    # val_image_dir = "./datasets/ROSSA/value/images"  # 验证集图像路径
    # val_label_dir = "./datasets/ROSSA/value/labels"  # 验证集标签路径
    # test_image_dir = "./datasets/ROSSA/test/images"  # 测试集图像路径
    # test_label_dir = "./datasets/ROSSA/test/labels"  # 测试集标签路径

    batch_size = 2
    num_epochs = 1
    learning_rate = 5e-4
    patience = 100  # 早停等待的epoch数
    power = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据加载
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),  
                                    transforms.RandomVerticalFlip(p=0.5),   
                                    transforms.RandomRotation(degrees=30), ])
    # transform = transforms.Compose([transforms.RandomRotation(degrees=10)])
    
    # 训练集
    train_dataset = BMPDataset(image_dir, label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 验证集
    val_dataset = BMPDataset(val_image_dir, val_label_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 测试集
    test_dataset = BMPDataset(test_image_dir, test_label_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型、损失、优化器
    # model = FRNet(ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock)
    # modelname = "FRNet"
    # model = PIENet(n_channels=1, n_classes=1, patchsize=20)
    # modelname = "PIENet"
    model = UNet(n_channels=1, n_classes=1)
    modelname = "UNet"
    # model = CSNet(channels=1, classes=1)
    # model = UTNet(in_chan=1, num_classes=1)
    # modelname = "UTNet"
    # modelname = "CSNet"
    # model = OCT2Former(in_chans=1, num_classes=1)
    # modelname = "OCT2Former"
    # model = CE_Net_OCT(num_classes=1,num_channels=1)
    # modelname = "CEnet"
    # model = SwinTransformerSys(img_size=448,in_chans=1,num_classes=1)
    # modelname = "swinunet"
    # model = DGNet(inp_c=1,input_resolution=(320,320))
    # modelname = "DGNet"
    # model = VesselNet(in_channels=1)
    # modelname = "Vesselnet"
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate , weight_decay=0.0001)#FR0.001 DG0.0001
    # optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad ], lr=learning_rate , weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=1e-6,last_epoch=-1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** power)
    # scheduler = None
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # 开始训练
    model = train(model, train_loader, val_loader, test_loader,  optimizer, device, num_epochs, modelname, scheduler, patience)


# 获取当前时间并格式化为字符串
    utc_now = datetime.now()
    beijing_now = utc_now + timedelta(hours=8)
    timestamp = beijing_now.strftime("%Y%m%d_%H%M")

    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), f"./checkpoints/{dataname}_{modelname}_{timestamp}.pth")
    else:
        torch.save(model.state_dict(), f"./checkpoints/{dataname}_{modelname}_{timestamp}.pth")
