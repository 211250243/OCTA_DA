import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import os
from tqdm import tqdm
from loaddata import BMPDataset
from models.PIENet import PIENet  # 根据您的模型选择导入

def calculate_metrics(model, test_loader, device, threshold=0.5, crop_size=None):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = outputs

            if crop_size is not None:
                B, C, H1, W1 = images.shape
                H, W = crop_size
                pad_height = max(H1 - H, 0)
                pad_width = max(W1 - W, 0)  
                h_start = pad_height // 2
                w_start = pad_width // 2
                probs = probs[:, :, h_start:H + h_start, w_start:W + w_start]
                labels = labels[:, :, h_start:H + h_start, w_start:W + w_start]
            
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
    all_labels = np.concatenate(all_labels).astype(int)

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

def run_inference(model_path, model_class, test_loader, device, modelname, threshold=0.5, crop_size=None):
    """
    加载模型权重并计算测试集指标
    
    Args:
        model_path: 模型权重文件路径
        model_class: 模型类
        test_loader: 测试数据加载器
        device: 设备
        modelname: 模型名称
        threshold: 二值化阈值
        crop_size: 裁剪尺寸 (H, W)
    """
    # 初始化模型
    model = model_class
    model.to(device)
    model.eval()
    
    # 加载权重
    if os.path.exists(model_path):
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print(f"错误: 模型权重文件不存在: {model_path}")
        return
    
    # 计算指标
    print("开始计算测试集指标...")
    test_metrics = calculate_metrics(model, test_loader, device, threshold, crop_size)

    # 打印结果
    result_str = f"\nTest Set Metrics:\n"
    result_str += f"Dice: {test_metrics['Dice']:.6f}\n"
    result_str += f"Sensitivity: {test_metrics['Sen']:.6f}\n"
    result_str += f"Specificity: {test_metrics['Spe']:.6f}\n"
    result_str += f"FDR: {test_metrics['FDR']:.6f}\n"
    result_str += f"Accuracy: {test_metrics['Acc']:.6f}\n"
    result_str += f"AUC: {test_metrics['AUC']:.6f}\n"
    result_str += f"Kappa: {test_metrics['Kappa']:.6f}\n"
    result_str += f"Confusion Matrix: TP={test_metrics['TP']}, FP={test_metrics['FP']}, TN={test_metrics['TN']}, FN={test_metrics['FN']}\n"

    print(result_str)

    # 保存结果
    os.makedirs('./results', exist_ok=True)
    result_filename = f"./results/{modelname}_inference_results.txt"
    with open(result_filename, 'a') as f:
        f.write(result_str)

    print(f"测试结果已保存至: {result_filename}")

if __name__ == "__main__":
    # 参数设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 数据集配置（根据您的实际数据集修改）
    dataname = '6mm'  # 或 '3mm', 'ROSE', 'ROSSA'
    test_image_dir = f"./datasets/OCTA-500/{dataname}/test/images"
    test_label_dir = f"./datasets/OCTA-500/{dataname}/test/labels"
     # 模型初始化（根据您的模型选择）
    model = PIENet(n_channels=1, n_classes=1, patchsize=20)
    modelname = "PIENet"

    # 模型权重路径
    model_path = f"./checkpoints/{modelname}/6mm_PIENet_20250813_0414.pth"  # 替换为您的实际模型路径
    
    # 测试集
    test_dataset = BMPDataset(test_image_dir, test_label_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
   
    # 运行推理
    run_inference(
        model_path=model_path,
        model_class=model,
        test_loader=test_loader,
        device=device,
        modelname=modelname,
        threshold=0.5,
        crop_size=None  # 如果需要裁剪，设置为 (H, W)
    )