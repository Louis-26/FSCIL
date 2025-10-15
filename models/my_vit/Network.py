"""
DiffusionFSCIL Network
使用 CLIP + 特征扩散模型 实现Few-Shot Class-Incremental Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from .diffusion_core import FeatureDDPM
from models.logger import LOGGER

log = LOGGER.LOGGER


class MYNET(nn.Module):
    """DiffusionFSCIL主网络"""

    def __init__(self, args, mode=None):
        super().__init__()

        self.args = args
        self.mode = mode
        
        log.info("="*60)
        log.info("Initializing DiffusionFSCIL Network")
        log.info("="*60)
        
        log.info("Loading CLIP ViT-B/16...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device="cuda")
        
        # 固定CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
        self.num_features = 512  # CLIP ViT-B/16 特征维度
        log.info(f"✅ CLIP loaded and frozen (feature_dim={self.num_features})")
        
        self.diffusion_model = FeatureDDPM(
            feature_dim=self.num_features,
            num_classes=args.num_classes,
            num_steps=getattr(args, 'num_diffusion_steps', 1000)
        )
        log.info(f"✅ Diffusion model created ({args.num_diffusion_steps if hasattr(args, 'num_diffusion_steps') else 1000} steps)")
        
        self.text_prototypes = {}      # 文本原型：{class_id: text_feature}
        self.visual_prototypes = {}    # 视觉原型（缓存）：{class_id: generated_feature}
        
        self.class_names = self._get_class_names(args.dataset)
        log.info(f"✅ Class names loaded ({len(self.class_names)} classes)")
        
        log.info("="*60)
    
    def _get_class_names(self, dataset):
        """获取数据集类别名称"""
        if dataset == 'cifar100':
            # CIFAR-100的100个类别（按官方顺序）
            return [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 
                'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 
                'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 
                'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
                'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray',
                'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
        
        elif dataset == 'mini_imagenet':
            # mini-ImageNet的100个类（简化版，实际可能需要完整映射）
            return [f"imagenet_class_{i}" for i in range(100)]
        
        elif dataset == 'cub200':
            # CUB-200的200个鸟类（简化版，实际应该用鸟类名称）
            return [f"bird_species_{i}" for i in range(200)]
        
        else:
            # 默认
            return [f"class_{i}" for i in range(self.args.num_classes)]
    
    @torch.no_grad()
    def encode_image(self, images):
        """
        使用CLIP提取图像特征
        
        Args:
            images: (B, 3, H, W) 图像tensor
        
        Returns:
            features: (B, 512) CLIP特征
        """
        return self.clip_model.encode_image(images).float()
    
    @torch.no_grad()
    def encode_text(self, text_list):
        """
        使用CLIP提取文本特征
        
        Args:
            text_list: List[str] 文本列表
        
        Returns:
            features: (N, 512) 文本特征
        """
        text_tokens = clip.tokenize(text_list).cuda()
        return self.clip_model.encode_text(text_tokens).float()
    
    def generate_text_prototypes(self, class_ids, use_templates=True):
        """
        为指定类别生成文本原型
        
        Args:
            class_ids: List[int] 类别ID列表
            use_templates: bool 是否使用多模板增强
        
        Returns:
            text_features: (C, 512) 文本特征
        """
        text_features = []
        
        for class_id in class_ids:
            if class_id >= len(self.class_names):
                log.warning(f"Class ID {class_id} out of range, using default")
                class_name = f"class_{class_id}"
            else:
                class_name = self.class_names[class_id]
            
            if use_templates:
                # 多模板增强
                templates = [
                    f"a photo of a {class_name}.",
                    f"a blurry photo of a {class_name}.",
                    f"a photo of many {class_name}.",
                    f"a photo of the large {class_name}.",
                    f"a photo of the small {class_name}.",
                ]
            else:
                templates = [f"a photo of a {class_name}."]
            
            # 编码所有模板
            text_feat = self.encode_text(templates)
            
            # 聚合并归一化
            text_feat = text_feat.mean(dim=0)
            text_feat = text_feat / text_feat.norm()
            
            text_features.append(text_feat)
            self.text_prototypes[class_id] = text_feat
        
        return torch.stack(text_features)
    
    @torch.no_grad()
    def generate_prototypes(self, class_ids, use_cache=True, ddim_steps=50):
        """
        使用扩散模型生成视觉原型
        
        Args:
            class_ids: List[int] or Tensor 类别ID
            use_cache: bool 是否使用缓存
            ddim_steps: int DDIM采样步数
        
        Returns:
            prototypes: (C, 512) 生成的原型特征
        """
        if isinstance(class_ids, list):
            class_ids = torch.tensor(class_ids).cuda()
        elif not isinstance(class_ids, torch.Tensor):
            class_ids = torch.tensor([class_ids]).cuda()
        
        if class_ids.device != torch.device('cuda'):
            class_ids = class_ids.cuda()
        
        prototypes = []
        
        for cid in class_ids:
            cid_int = cid.item()
            
            # 检查缓存
            if use_cache and cid_int in self.visual_prototypes:
                prototypes.append(self.visual_prototypes[cid_int])
            else:
                # 生成新原型
                if cid_int not in self.text_prototypes:
                    log.warning(f"Text prototype for class {cid_int} not found, generating...")
                    self.generate_text_prototypes([cid_int])
                
                text_feat = self.text_prototypes[cid_int].unsqueeze(0)
                label = torch.tensor([cid_int]).cuda()
                
                # 使用扩散模型采样（guidance_scale=5.0最优）
                proto = self.diffusion_model.sample(
                    label, 
                    text_feat, 
                    ddim_steps=ddim_steps,
                    guidance_scale=5.0  #  关键：增强条件控制
                )
                
                # 缓存
                self.visual_prototypes[cid_int] = proto.squeeze(0)
                prototypes.append(proto.squeeze(0))
        
        return torch.stack(prototypes)
    
    def clear_cache(self):
        """清除视觉原型缓存"""
        self.visual_prototypes = {}
        log.info("Visual prototype cache cleared")
    
    def forward(self, x, mode='encode'):
        """
        前向传播（保持接口兼容性）
        
        Args:
            x: 输入
            mode: 'encode' - 提取特征
        
        Returns:
            features
        """
        if mode == 'encode' or mode == 'encoder':
            return self.encode_image(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_feature_dim(self):
        """返回特征维度"""
        return self.num_features


if __name__ == '__main__':
    """简单测试"""
    import argparse
    import sys
    sys.path.append('../..')
    from diffusion_core import FeatureDDPM
    
    # 重新定义MYNET避免import问题
    print("✅ Network.py can be imported as module")
    print("   Run test from project root: python -m models.my_vit.Network")
