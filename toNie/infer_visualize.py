"""
使用 train_target 保存的 best_teacher.pth.tar（或任意同结构 checkpoint）做推理并导出可视化。
在 toNie 目录下运行: python infer_visualize.py --checkpoint <path/to/best_teacher.pth.tar> --dataset Domain2 --split test -g 0
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders import custom_transforms as trans
from dataloaders import octa_dataloader
import networks.deeplabv3 as netd
from utils.metrics import dice_coeff_binary


def load_state_dict_flexible(model, checkpoint_path, map_location="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def input_tensor_to_gray_uint8(img_t):
    """[3,H,W] float 0~1 -> HxW uint8"""
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    if x.shape[2] == 3:
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    else:
        gray = x[:, :, 0]
    return gray


def label_tensor_to_mask_uint8(lbl_t, threshold=0.5):
    """[1,H,W] -> HxW 0/255"""
    m = lbl_t.detach().cpu().numpy()[0]
    return (m > threshold).astype(np.uint8) * 255


def prob_to_mask_uint8(logits, threshold=0.5):
    prob = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]
    return (prob > threshold).astype(np.uint8) * 255


def blend_pred_overlay(gray_uint8, pred_mask_uint8, bgr=(0, 0, 255), alpha=0.45):
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    m = pred_mask_uint8 > 127
    overlay[m] = bgr
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)


def concat_h(images_bgr):
    h = min(im.shape[0] for im in images_bgr)
    resized = []
    for im in images_bgr:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            w = int(im.shape[1] * scale)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(im)
    return np.hstack(resized)


def main():
    parser = argparse.ArgumentParser(description="OCTA DeepLab 推理与可视化")
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="best_teacher.pth.tar 或含 model_state_dict 的权重",
    )
    parser.add_argument("--data-dir", type=str, default="../framework/datasets/OCTA-500")
    parser.add_argument("--dataset", type=str, default="Domain2")
    parser.add_argument("--split", type=str, default="test", help="train / value(val) / test")
    parser.add_argument("--out-dir", type=str, default="", help="默认同目录下 infer_vis_<dataset>_<split>")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-stride", type=int, default=16)
    parser.add_argument("--sync-bn", type=bool, default=True)
    parser.add_argument("--freeze-bn", type=bool, default=False)
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值，与 eval 中 dice 一致")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-prob", action="store_true", help="额外保存概率图 prob_*.png (0~255)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    here = osp.dirname(osp.abspath(__file__))
    if not args.out_dir:
        args.out_dir = osp.join(
            here,
            "infer_vis_{}_{}".format(args.dataset, args.split.replace("/", "_")),
        )
    os.makedirs(args.out_dir, exist_ok=True)

    composed = transforms.Compose(
        [
            trans.Resize(512),
            trans.NormalizeOCTA(),
            trans.ToTensorOCTA(),
        ]
    )
    dataset = octa_dataloader.OCTASegmentation(
        base_dir=args.data_dir,
        dataset=args.dataset,
        split=args.split,
        transform=composed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = netd.DeepLab(
        num_classes=1,
        backbone="mobilenet",
        output_stride=args.out_stride,
        sync_bn=args.sync_bn,
        freeze_bn=args.freeze_bn,
    )
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    load_state_dict_flexible(model, args.checkpoint, map_location=map_loc)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    dice_list = []
    with torch.no_grad():
        for sample in loader:
            data = sample["image"]
            target = sample["label"]
            names = sample["img_name"]
            if torch.cuda.is_available():
                data = data.cuda()
            logits, _ = model(data)
            for i in range(data.size(0)):
                d = dice_coeff_binary(
                    logits[i : i + 1], target[i : i + 1], threshold=args.threshold
                )
                dice_list.append(d)

                gray = input_tensor_to_gray_uint8(data[i])
                pred_m = prob_to_mask_uint8(logits[i : i + 1], threshold=args.threshold)
                gt_m = label_tensor_to_mask_uint8(target[i], threshold=0.5)
                overlay = blend_pred_overlay(gray, pred_m)

                base = osp.splitext(names[i])[0]
                out_base = osp.join(args.out_dir, base)
                cv2.imwrite(out_base + "_input.png", gray)
                cv2.imwrite(out_base + "_pred.png", pred_m)
                cv2.imwrite(out_base + "_gt.png", gt_m)
                cv2.imwrite(out_base + "_overlay.png", overlay)
                panel = concat_h(
                    [
                        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(pred_m, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(gt_m, cv2.COLOR_GRAY2BGR),
                        overlay,
                    ]
                )
                cv2.imwrite(out_base + "_panel.png", panel)

                if args.save_prob:
                    prob = torch.sigmoid(logits[i, 0]).detach().cpu().numpy()
                    prob_u8 = (prob * 255.0).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(out_base + "_prob.png", prob_u8)

    mean_dice = float(np.mean(dice_list)) if dice_list else float("nan")
    summary_path = osp.join(args.out_dir, "dice_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("checkpoint: {}\n".format(args.checkpoint))
        f.write("dataset: {} split: {}\n".format(args.dataset, args.split))
        f.write("n_images: {}\n".format(len(dice_list)))
        f.write("mean_dice (per-image batch mean, th={}): {:.6f}\n".format(args.threshold, mean_dice))
    print("Saved visualizations to: {}".format(args.out_dir))
    print("Mean dice: {:.4f} (see {})".format(mean_dice, summary_path))


if __name__ == "__main__":
    main()
