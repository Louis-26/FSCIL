"""
DiffusionFSCIL Trainer
极简训练器 - Training-Free增量学习
"""
import os
import torch
import torch.nn as nn
from models.my_vit.Network import MYNET
from models.my_vit import helper
from utils import *
from dataloader.data_utils import *
from models.logger import LOGGER

log = LOGGER.LOGGER


class FSCILTrainer:
    """DiffusionFSCIL训练器"""
    
    def __init__(self, args):
        self.args = args
        
        # 设置保存路径
        self.save_path = os.path.join(
            'checkpoint',
            f'{args.project}_{args.dataset}',
        )
        ensure_path(self.save_path)
        
        log.info("="*60)
        log.info("DiffusionFSCIL Trainer Initialized")
        log.info(f"Project: {args.project}")
        log.info(f"Dataset: {args.dataset}")
        log.info(f"Save path: {self.save_path}")
        log.info("="*60)
        
        # 初始化模型
        log.info("Initializing model...")
        self.model = MYNET(self.args).cuda()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        log.info(f"Total parameters: {total_params/1e6:.2f}M")
        log.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
        log.info(f"Frozen parameters: {(total_params-trainable_params)/1e6:.2f}M")
        
        # 结果记录
        self.train_acc_curve = []
        self.test_acc_curve = []
        self.gAcc_curve = []

    def get_dataloader(self, session):
        """获取数据加载器"""
        if session == 0:
            # Base session (返回4个值，第4个是trainloader_pk)
            trainset, trainloader, testloader, _ = get_base_dataloader(self.args)
        else:
            # Incremental sessions
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        
        return trainset, trainloader, testloader
    
    def train_base(self):
        """训练Base Session（扩散模型）"""
        log.info("\n" + "="*60)
        log.info("SESSION 0 - BASE TRAINING")
        log.info("="*60)
        
        # 获取数据
        trainset, trainloader, testloader = self.get_dataloader(session=0)
        
        # 训练扩散模型
        helper.train_diffusion_base(
            model=self.model,
            dataloader=trainloader,
            args=self.args
        )
        
        # 测试
        acc_dict, acc_list = helper.test_diffusion(
            model=self.model,
            testloader=testloader,
            args=self.args,
            session=0
        )
        
        # 记录结果
        self.test_acc_curve.append(acc_list)
        self.gAcc_curve.append(acc_dict['gAcc'])
        
        log.info(f"Session 0 Results:")
        log.info(f"  - Top1 Acc: {acc_dict['total']:.2f}%")
        log.info(f"  - gAcc: {acc_dict['gAcc']:.2f}%")
        
        # 保存检查点
        self.save_checkpoint('session0_diffusion.pth')
        log.info(f"✅ Session 0 checkpoint saved")
    
    def train_incremental(self, session):
        """增量学习（Training-Free）"""
        log.info("\n" + "="*60)
        log.info(f"SESSION {session} - INCREMENTAL LEARNING (Training-Free)")
        log.info("="*60)
        
        # 获取数据
        trainset, trainloader, testloader = self.get_dataloader(session)
        
        # Training-Free增量学习（只更新文本原型）
        helper.incremental_learning(
            model=self.model,
            trainloader=trainloader,
            args=self.args,
            session=session
        )
        
        # 测试
        acc_dict, acc_list = helper.test_diffusion(
            model=self.model,
            testloader=testloader,
            args=self.args,
            session=session
        )
        
        # 记录结果
        self.test_acc_curve.append(acc_list)
        self.gAcc_curve.append(acc_dict['gAcc'])
        
        log.info(f"Session {session} Results:")
        log.info(f"  - Top1 Acc: {acc_dict['total']:.2f}%")
        log.info(f"  - gAcc: {acc_dict['gAcc']:.2f}%")
        for i, acc in enumerate(acc_list):
            log.info(f"    Task {i}: {acc:.2f}%")
        
        # 保存检查点（虽然没有参数更新，但保存状态）
        self.save_checkpoint(f'session{session}_diffusion.pth')
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        # 复制args但排除不能pickle的module对象
        args_dict = {k: v for k, v in vars(self.args).items() if k != 'Dataset'}
        
        save_dict = {
            'args': args_dict,
            'dataset_name': self.args.dataset,  # 保存数据集名称以便加载时重建
            'model_state_dict': self.model.state_dict(),
            'text_prototypes': self.model.text_prototypes,
            'train_acc_curve': self.train_acc_curve,
            'test_acc_curve': self.test_acc_curve,
            'gAcc_curve': self.gAcc_curve,
        }
        
        save_path = os.path.join(self.save_path, filename)
        torch.save(save_dict, save_path)
        log.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        load_path = os.path.join(self.save_path, filename)
        if not os.path.exists(load_path):
            log.warning(f"Checkpoint not found: {load_path}")
            return False
        
        checkpoint = torch.load(load_path, map_location='cuda')
        
        # 加载模型
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载文本原型
        self.model.text_prototypes = checkpoint.get('text_prototypes', {})
        
        # 加载结果记录
        self.train_acc_curve = checkpoint.get('train_acc_curve', [])
        self.test_acc_curve = checkpoint.get('test_acc_curve', [])
        self.gAcc_curve = checkpoint.get('gAcc_curve', [])
        
        log.info(f"✅ Checkpoint loaded from {load_path}")
        return True
    
    def run(self):
        """完整训练流程"""
        log.info("\n" + "="*60)
        log.info("STARTING DiffusionFSCIL TRAINING")
        log.info("="*60)
        log.info(f"Total sessions: {self.args.sessions}")
        log.info(f"Base classes: {self.args.base_class}")
        log.info(f"Way per session: {self.args.way}")
        log.info(f"Shot: {self.args.shot}")
        log.info("="*60)
        
        # Session 0: Base训练
        self.train_base()
        
        # Sessions 1-N: 增量学习
        for session in range(1, self.args.sessions):
            self.train_incremental(session)
        
        # 最终结果
        log.info("\n" + "="*60)
        log.info("TRAINING COMPLETED")
        log.info("="*60)
        
        log.info("\nAccuracy Summary:")
        for i, (test_acc, gacc) in enumerate(zip(self.test_acc_curve, self.gAcc_curve)):
            log.info(f"Session {i}: Top1={test_acc[-1]:.2f}% | gAcc={gacc:.2f}%")
        
        # 保存最终结果
        result_file = os.path.join(self.save_path, 'final_results_diffusion.txt')
        with open(result_file, 'w') as f:
            f.write(f"FSCIL Results with diffusion- {self.args.dataset}\n")
            f.write("="*60 + "\n")
            for i, (test_acc, gacc) in enumerate(zip(self.test_acc_curve, self.gAcc_curve)):
                f.write(f"Session {i}: Top1={test_acc[-1]:.2f}% | gAcc={gacc:.2f}%\n")
        
        log.info(f"\n✅ Results saved to {result_file}")
        log.info("="*60)
