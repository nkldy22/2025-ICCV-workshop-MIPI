import argparse
import cv2

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import os
from tqdm import tqdm
import torch
from fewlens.utils import img2tensor, tensor2img, imwrite
from fewlens.archs.MIMO_TLC_arch import *
import numpy as np

ccm = np.array([[1.93994141, -0.73925781, -0.20068359],
                [-0.28857422, 1.59741211, -0.30883789],
                [-0.0078125, -0.45654297, 1.46435547]])

ccm_inv = np.array([0.560585587345286, 0.299436361600599, 0.139978051054115, \
                    0.108381485800569, 0.724058690188644, 0.167559824010787, \
                    0.036780946522526, 0.227337731614535, 0.735881321862938]).reshape(3, 3)


def apply_ccm(img, ccm, inverse=False):
    """Applies a color correction matrix."""
    if inverse:
        ccm = np.linalg.inv(ccm)

    # Reshape img for matrix multiplication
    img_reshaped = img.reshape(-1, 3).T  # 转换为 3 x N 形状
    img_out = ccm @ img_reshaped  # 应用色彩校正矩阵
    img_out = img_out.T.reshape(img.shape)  # 恢复原始形状

    return img_out


def apply_wb(img, wb, inverse=False):
    """Applies white balance to the image."""
    if inverse:
        wb = 1.0 / wb

    img_out = np.stack((
        img[:, :, 0] * wb[0],
        img[:, :, 1] * wb[1],
        img[:, :, 2] * wb[2]
    ), axis=-1)

    return img_out


def test_selfensemble(lq, net_g, device='cuda'):
    def _transform(v, op):
        if v.dim() == 3:
            v = v.unsqueeze(0)
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(device)
        return ret

    if lq.dim() == 3:
        lq = lq.unsqueeze(0)

    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    net_g.eval()
    with torch.no_grad():
        out_list = []
        for aug in lq_list:
            print("input:", aug.shape)
            output = net_g(aug)
            if isinstance(output, (list, tuple)):
                output = output[0]  # 获取第一个输出
            if output.dim() == 3:
                output = output.unsqueeze(0)
            out_list.append(output)

    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')

    output = torch.cat(out_list, dim=0)
    result = output.mean(dim=0, keepdim=True)

    if lq.shape[0] == 1 and result.shape[0] == 1:
        result = result.squeeze(0)

    return result


def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        default='/mnt/sda/zsh/dataset/AberrationCorrection-MIPI-dataset/valid/lq',
                        help='Input image or folder')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='results_MIMO_ours',
                        help='results image folder')
    parser.add_argument(
        '-w',
        '--weight',
        type=str,
        default='/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/experiments/MIMO_ours/net_g_latest.pth',
        help='path for model weights')

    parser.add_argument('--suffix',
                        type=str,
                        default='',
                        help='Suffix of the restored image')
    parser.add_argument(
        '--max_size',
        type=int,
        default=2400,
        help=
        'Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wb_param = np.load('/mnt/sda/zsh/dataset/AberrationCorrection-MIPI-dataset/valid/wb_params.npz')

    weight_path = args.weight

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    sr_model = MIMOUNetLocal().to(device)

    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=True)
    sr_model.eval()

    total_files = 0
    # Walk through the directory
    for root, dirs, files in os.walk(args.input):
        total_files += len(files)  # Count the files in the current directory

    pbar = tqdm(total=total_files, unit='image')
    with torch.no_grad():
        for image in os.listdir(args.input):

            img_name = os.path.basename(image).split('.')[0]

            pbar.set_description(f'Test {img_name}')

            save_path = os.path.join(output_folder, f'{img_name}.png')
            print("img_name:", img_name)
            wb = wb_param[img_name]

            path = os.path.join(args.input, image)
            img_demosaiced = cv2.imread(path)
            img_demosaiced = cv2.cvtColor(img_demosaiced, cv2.COLOR_BGR2RGB) / 255.0
            [h, w, _] = img_demosaiced.shape

            h_range = np.arange(0, h, 1)
            w_range = np.arange(0, w, 1)
            img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)

            img_fld_h = ((img_fld_h - (h - 1) / 2) / ((h - 1) / 2)).astype(np.float32)
            img_fld_w = ((img_fld_w - (w - 1) / 2) / ((w - 1) / 2)).astype(np.float32)
            # print(img_fld_h)

            img_fld_h = np.expand_dims(img_fld_h, -1)
            img_fld_w = np.expand_dims(img_fld_w, -1)

            img_wz_fld = np.concatenate([img_demosaiced, img_fld_h, img_fld_w], 2)

            img_tensor = img2tensor(img_wz_fld).to(device)
            img_tensor = img_tensor.unsqueeze(0)

            b, c, h, w = img_tensor.shape
            if h * w < 2048 * 2048:
                # 使用 test_selfensemble 进行推理
                output = test_selfensemble(img_tensor, sr_model, device=device)
            else:
                patch_size = 1024
                stride = 896

                output = torch.zeros((b, 3, h, w)).to(device)
                weight_map = torch.zeros((b, 3, h, w)).to(device)

                for y in range(0, h, stride):
                    for x in range(0, w, stride):
                        y_end = min(y + patch_size, h)
                        x_end = min(x + patch_size, w)
                        y_start = max(y_end - patch_size, 0)
                        x_start = max(x_end - patch_size, 0)

                        patch = img_tensor[:, :, y_start:y_end, x_start:x_end]

                        # 使用 test_selfensemble 进行推理
                        patch_output = test_selfensemble(patch, sr_model, device=device)

                        output[:, :, y_start:y_end, x_start:x_end] += patch_output

                        weight_map[:, :, y_start:y_end, x_start:x_end] += 1

                output = output / weight_map

            output_img = output.cpu().numpy()[0].transpose(1, 2, 0)
            output_img = apply_wb(output_img, wb)

            output_img = apply_ccm(output_img, ccm, inverse=False)

            output_img = output_img.clip(0, 1)

            output_img = cv2.pow(output_img, 1 / 2.2) * 255
            output_img = output_img.clip(0, 255)

            output_img = output_img.astype(np.uint8)

            cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

            pbar.update(1)
        pbar.close()


if __name__ == '__main__':
    main()