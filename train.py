"""
DiffusionFSCIL Training Script
简化的训练脚本 - 专注于扩散模型训练
"""
import argparse
import os
import random

import numpy as np
import torch

from models.logger import LOGGER
from models.my_vit.fscil_trainer import FSCILTrainer

log = LOGGER.LOGGER

DATA_DIR = 'data/'
PROJECT = 'diffusion_fscil'

import sys

sys.argv = [
    "program_name",
    "-project", PROJECT,
    "-dataset", "cifar100",
    "-gpu", "0",
    "-epochs_base", "200",
    "-num_diffusion_steps", "50",
    "-ddim_steps", "50",
    "-lr_diffusion", "1e-4",
    "-batch_size_diffusion", "256",
    "-batch_size_base", "128",
    "-test_batch_size", "100",
    "-num_workers", "8",
    "-seed", "1"
]

def display_device_info(PROJECT_NAME, DATASET, GPU_ID):
    # Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_cuda_devices}")

    # Iterate through each CUDA device to get its name
    for i in range(num_cuda_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"    GPU Device {i}: {device_name}")

    print(f"""
        ==================================================
          DiffusionFSCIL - CIFAR-100
        ==================================================
        Project: {PROJECT_NAME}
        Dataset: {DATASET}
        GPU name: {torch.cuda.get_device_name(0)}
        CUDA: {torch.version.cuda}
        GPU ID: {GPU_ID}
        ==================================================
        """
          )

def get_command_line_parser():
    parser = argparse.ArgumentParser(description='DiffusionFSCIL Training')

    # ========== 基础设置 ==========
    parser.add_argument('-project', type=str, default=PROJECT, help='Project name')
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['cifar100', 'mini_imagenet', 'cub200'],
                        help='Dataset name')
    parser.add_argument('-dataroot', type=str, default=DATA_DIR, help='Data root directory')

    # ========== FSCIL设置 ==========
    parser.add_argument('-way', type=int, default=5, help='Classes per incremental session')
    parser.add_argument('-shot', type=int, default=5, help='Shots per class')
    parser.add_argument('-sessions', type=int, default=9, help='Total number of sessions')
    parser.add_argument('-base_class', type=int, default=60, help='Number of base classes')

    # ========== 扩散模型设置 ==========
    parser.add_argument('-num_diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('-ddim_steps', type=int, default=50,
                        help='Number of DDIM sampling steps (default: 50)')
    parser.add_argument('-lr_diffusion', type=float, default=1e-4,
                        help='Learning rate for diffusion model')
    parser.add_argument('-batch_size_diffusion', type=int, default=256,
                        help='Batch size for diffusion training')

    # ========== 训练设置 ==========
    parser.add_argument('-epochs_base', type=int, default=100,
                        help='Training epochs for base session')
    parser.add_argument('-batch_size_base', type=int, default=128,
                        help='Batch size for base session data loading')
    parser.add_argument('-test_batch_size', type=int, default=100,
                        help='Batch size for testing')

    # ========== 环境设置 ==========
    parser.add_argument('-gpu', type=str, default='0', help='GPU device ID')
    parser.add_argument('-num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('-seed', type=int, default=1, help='Random seed')

    # ========== 其他设置 ==========
    parser.add_argument('-debug', action='store_true', help='Debug mode')
    parser.add_argument('-start_session', type=int, default=0,
                        help='Starting session (0=from scratch)')
    parser.add_argument('-resume', type=str, default=None,
                        help='Resume from checkpoint')

    # ========== 数据集特定参数（自动设置） ==========
    parser.add_argument('-image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('-num_classes', type=int, default=None,
                        help='Total number of classes (auto-set)')

    # ========== PKSampler参数 ==========
    parser.add_argument('-p', type=int, default=64, help='P for PKSampler')
    parser.add_argument('-k', type=int, default=8, help='K for PKSampler')

    # ========== 数据集特定参数（增量） ==========
    parser.add_argument('-dataset_seed', type=int, default=None, help='Dataset seed for reproducibility')
    parser.add_argument('-num_workers_new', type=int, default=0, help='Workers for incremental data loading')
    parser.add_argument('-batch_size_new', type=int, default=0, help='Batch size for incremental sessions (0=use all)')

    return parser


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_config_dataset(args):
    """自动配置数据集参数"""
    # way: number of different classes in each session
    # shot: number of additional samples in each session
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.base_class = 60
        args.way = 5
        args.shot = 5
        args.sessions = 9  # 60 + 8*5 = 100
        args.image_size = 224

    elif args.dataset == 'mini_imagenet':
        args.num_classes = 100
        args.base_class = 60
        args.way = 5
        args.shot = 5
        args.sessions = 9
        args.image_size = 224

    elif args.dataset == 'cub200':
        args.num_classes = 200
        args.base_class = 100
        args.way = 10
        args.shot = 5
        args.sessions = 11  # 100 + 10*10 = 200
        args.image_size = 224

    log.info(f"Dataset config: {args.dataset}")
    log.info(f"  - Total classes: {args.num_classes}")
    log.info(f"  - Base classes: {args.base_class}")
    log.info(f"  - Way: {args.way}")
    log.info(f"  - Shot: {args.shot}")
    log.info(f"  - Sessions: {args.sessions}")
    log.info(f"  - Image size: {args.image_size}")


def main():
    """主函数"""
    # 解析参数
    parser = get_command_line_parser()
    args = parser.parse_args()

    # 设置GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 设置随机种子
    set_seed(args.seed)
    log.info(f"Random seed set to {args.seed}")

    # 自动配置数据集
    auto_config_dataset(args)

    # 设置数据集加载器（导入Dataset模块）
    from dataloader.data_utils import set_up_datasets
    args = set_up_datasets(args)

    # 打印配置
    log.info("\n" + "=" * 60)
    log.info("DiffusionFSCIL Configuration")
    log.info("=" * 60)
    for key, value in vars(args).items():
        log.info(f"{key:30s}: {value}")
    log.info("=" * 60 + "\n")

    # 创建训练器
    trainer = FSCILTrainer(args)

    # 开始训练
    if args.start_session == 0:
        # 从头开始
        trainer.run()
    else:
        # 从某个session恢复
        log.info(f"Resuming from session {args.start_session}...")

        # 加载之前的checkpoint
        if args.resume:
            trainer.load_checkpoint(args.resume)
        else:
            # 尝试自动加载
            prev_session = args.start_session - 1
            trainer.load_checkpoint(f'session{prev_session}_diffusion.pth')

        # 继续训练
        for session in range(args.start_session, args.sessions):
            if session == 0:
                trainer.train_base()
            else:
                trainer.train_incremental(session)


if __name__ == '__main__':
    PROJECT_NAME, DATASET, GPU_ID = "diffusion_fscil_cifar100", "cifar100", "0"
    display_device_info(PROJECT_NAME, DATASET, GPU_ID)

    import itertools

    change_para = {}
    change_para["-num_diffusion_steps"] = [10, 50, 100, 200]
    change_para["-ddim_steps"] = [10, 30, 100]
    change_para["-lr_diffusion"] = [5e-5, 2e-4, 5e-4]

    # find existing results
    existing_results = list()
    results_collected_path = "./complementary/results_collected"
    for f in os.listdir(results_collected_path):
        if f.startswith("res_diff"):
            para = f.split("_")
            existing_results.append((eval(para[3]), eval(para[5]), eval(para[7].replace(".txt", ""))))

    # iterate all combinations using itertools.product
    keys = list(change_para.keys())
    for combo in itertools.product(*(change_para[k] for k in keys)):
        # set each argument in sys.argv (or append if missing)
        for k, val in zip(keys, combo):
            if k in sys.argv:
                idx = sys.argv.index(k)
                sys.argv[idx + 1] = str(val)
            else:
                sys.argv.extend([k, str(val)])

        # read numeric values from sys.argv
        try:
            diff_steps = int(sys.argv[sys.argv.index("-num_diffusion_steps") + 1])
            ddim = int(sys.argv[sys.argv.index("-ddim_steps") + 1])
            lr_diffusion = float(sys.argv[sys.argv.index("-lr_diffusion") + 1])
        except Exception:
            print("Skipping combination due to invalid argument values:", combo)
            continue

        # basic validation
        if diff_steps < ddim:
            print(f"Skipping invalid config: diffusion steps {diff_steps} > ddim steps {ddim}")
            continue

        if (diff_steps, ddim, lr_diffusion) in existing_results:
            print(f"Skipping existing result for {diff_steps}, {ddim}, {lr_diffusion}")
            continue

        print(f"Running config: {diff_steps}, {ddim}, {lr_diffusion}")
        main()

    print(r"""
          ==================================================
            Training completed!
          ==================================================
          """)