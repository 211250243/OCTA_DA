import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from loaddata import BMPDataset
# from models.DGNet import *
# from models.PIENet import *
# from models.PIENet import *
# from models.frnet import *
# from models.UNet import *
# from models.CSNet import *
# from models.OCT2former import*
# from models.utnet import *
# from models.CEnet import *
# from models.swinuent import *
from models.Vesselnet import *
import torch.nn.functional as F

def save_prediction(pred_tensor, save_path,crop_size = None):
    """
    将模型输出转换为灰度 BMP 图像（0-255）
    """
    pred_np = pred_tensor.squeeze().cpu().numpy()  # [H, W]
    pred_np = (pred_np * 255).astype(np.uint8)     # 0~1 → 0~255
    img = Image.fromarray(pred_np)
    if crop_size is not None:
        H1,W1 = img.size
        H, W = crop_size
        pad_height = max(H1 - H, 0)
        pad_width = max(W1 - W, 0)  
        h_start = pad_height//2
        w_start = pad_width//2
        img = img.crop((w_start, h_start, w_start + W, h_start + H))
    img.save(save_path)

def run_inference(model_path, save_dir="./output",
                #   test_image_dir="./datasets/OCTA-500/3mm/test/images",
                #   test_label_dir="./datasets/OCTA-500/3mm/test/labels",
                #   test_image_dir="./datasets/OCTA-500/6mm/test/images",
                #   test_label_dir="./datasets/OCTA-500/6mm/test/labels",
                #   test_image_dir="./datasets/ROSE/test/images",
                #   test_label_dir="./datasets/ROSE/test/labels",
                  test_image_dir="./datasets/ROSSA/test/images",
                  test_label_dir="./datasets/ROSSA/test/labels",
                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                  threshold=0.5):
    
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    test_dataset = BMPDataset(test_image_dir, test_label_dir, transform=None, target_size=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 创建模型
    # model = FRNet(ch_in=1, ch_out=1, cls_init_block=RRCNNBlock, cls_conv_block=RecurrentConvNeXtBlock)
    # modelname = "FRNet"
    # model = PIENet(n_channels=1, n_classes=1, patchsize=15)
    # modelname = "PIENet_pro"
    # model = UNet(n_channels=1, n_classes=1)
    # modelname = "UNet"
    # model = UTNet(in_chan=1, num_classes=1)
    # modelname = "UTNet"
    # model = CSNet(channels=1, classes=1)
    # modelname = "CSNet"
    # model = OCT2Former(in_chans=1, num_classes=1)
    # modelname = "OCT2Former"
    # model = CE_Net_OCT(num_classes=1,num_channels=1)
    # modelname = "CEnet"
    # model = SwinTransformerSys(img_size=448,in_chans=1,num_classes=1)
    # modelname = "swinunet"
    # model = DGNet(inp_c=1,input_resolution=(320,320))
    # modelname = "DGNet"
    model = VesselNet(in_channels=1)
    # modelname = "Vesselnet"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"开始推理并保存结果至 {save_dir} ...")

    with torch.no_grad():
        for idx, (image, label, filename) in enumerate(test_loader):
            # image = image.repeat(1, 3, 1, 1)
            image = image.to(device)
            out1, out2, out3, out4 = model(image)
            # output = model(image)                   # (1, 1, H, W)
            # output = torch.sigmoid(output)          # Sigmoid → 0~1
            output_bin = (out1 > threshold).float()  # 二值化
            
            filename = os.path.splitext(filename[0])[0]
            save_path = os.path.join(save_dir, f"{filename}.png")

            save_prediction(output_bin[0, 0], save_path, crop_size=None)

    print("所有预测已保存完成！")

if __name__ == "__main__":
    model_path = "./checkpoints/Vesselnet/ROSSA_Vesselnet_20250815_0451.pth"
    run_inference(model_path=model_path)
