import argparse
import cv2

import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
from fewlens.archs.ASTv2_arch import *
from fewlens.archs.fov_arch import *
from fewlens.archs.MIMO_arch import *
from fewlens.archs.restormer_arch import *

path_fovkpn='/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/experiments/fovkpn_perturbed_archived_20250602_134347/models/net_g_latest.pth'
path_MIMO='/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/experiments/mimo_perturbed_archived_20250602_134433/models/net_g_latest.pth'
path_restormer='/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/experiments/restormer_perturbed/models/net_g_latest.pth'
path_ASTv2='/home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main/experiments/ASTv2_perturbed/models/net_g_latest.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sr_model_AST = ASTv2().to(device)
sr_model_fovkpn = FOVKPN().to(device)
sr_model_MIMO = MIMOUNet().to(device)
sr_model_restormer = Restormer().to(device)

sr_model_restormer.load_state_dict(torch.load(path_restormer)['params'], strict=True)
print("restormer done")
sr_model_MIMO.load_state_dict(torch.load(path_MIMO)['params'], strict=True)
print("MIMO done")
sr_model_fovkpn.load_state_dict(torch.load(path_fovkpn)['params'], strict=True)
print("fovkpn done")
sr_model_AST.load_state_dict(torch.load(path_ASTv2)['params'], strict=True)
print("AST done")

