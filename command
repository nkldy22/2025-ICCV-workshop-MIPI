conda activate fewlens
cd /home/ubuntu13/zsh/code/Aberration_Correction_MIPI-main
训练指令
nohup sh train_mimo_pertubed.sh > baseline_1.txt 2>&1 &

nohup sh train_fovkpn_pertubed.sh > baseline2_fovkpn_pertubed.txt 2>&1 &

nohup sh train_restormer_pertubed.sh > baseline3_restormer_pertubed.txt 2>&1 &

nohup sh train_ASTv2_pertubed.sh  > ASTv2_72_dim_perturbed.txt 2>&1 &

nohup sh train_uformer_pertubed.sh  > baseline5_uformer_pertubed.txt 2>&1 &

nohup sh train_ConvIR_pertubed.sh  > baseline6_ConvIR_pertubed.txt 2>&1 &

nohup sh train_HINT_pertubed.sh  > baseline7_HINT_pertubed.txt 2>&1 &


nohup sh train_ASTv2_pertubed_tmp.sh  > ASTv2_48_dim_perturbed_[468]_30w.txt 2>&1 &


推理指令+测试指标
conda activate fewlens
export PYTHONPATH="./:${PYTHONPATH}"
python inference/inference_
cd ..
cd MANIQA-master
python inference.py
conda activate Uformer
cd ..
cd niqe-master
python niqe.py

