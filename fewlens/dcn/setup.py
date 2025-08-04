import sys
import os

# 获取当前 Conda 环境的 site-packages 路径
conda_env_path = os.path.dirname(sys.executable)  # /home/ubuntu13/anaconda3/envs/fewlens/bin
site_packages_path = os.path.join(conda_env_path, "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")

# 将路径添加到系统路径（确保构建环境能找到 torch）
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

# 验证是否能导入 torch（调试用，后续可删除）
try:
    import torch
    print(f"成功导入 torch，路径：{torch.__file__}")
except ImportError:
    print(f"仍然无法导入 torch，当前 sys.path：{sys.path}")
    sys.exit(1)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, sources):

    return CUDAExtension(
        name='{}'.format(name), sources=[p for p in sources], extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


setup(
    name='deform_conv', ext_modules=[
        make_cuda_ext(name='deform_conv_cuda',
                      sources=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu'])
    ], cmdclass={'build_ext': BuildExtension}, zip_safe=False)
