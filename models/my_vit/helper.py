"""
DiffusionFSCIL Helper Functions
包含扩散模型训练、测试和特征提取等辅助函数
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import accuracy_per_task
from models.logger import LOGGER

log = LOGGER.LOGGER


@torch.no_grad()
def extract_clip_features(model, dataloader):
    """
    提取CLIP特征
    
    Args:
        model: MYNET模型
        dataloader: 数据加载器
    
    Returns:
        all_features: (N, 512) 特征
        all_labels: (N,) 标签
    """
    all_features = []
    all_labels = []
    
    model.eval()
    
    for images, labels in tqdm(dataloader, desc="Extracting CLIP features"):
        images = images.cuda()
        features = model.encode_image(images)
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


def train_diffusion_base(model, dataloader, args):
    """
    训练扩散模型（Session 0）
    
    Args:
        model: MYNET模型
        dataloader: 数据加载器
        args: 参数
    """
    log.info("="*60)
    log.info("Training Diffusion Model (Base Session)")
    log.info("="*60)
    
    # 1. 提取所有CLIP特征
    log.info("Step 1: Extracting CLIP features...")
    all_features, all_labels = extract_clip_features(model, dataloader)
    all_features = all_features.cuda()
    all_labels = all_labels.cuda()
    
    log.info(f"✅ Extracted {all_features.shape[0]} features")
    
    # 2. 生成文本原型
    log.info("Step 2: Generating text prototypes...")
    base_class_ids = list(range(args.base_class))
    model.generate_text_prototypes(base_class_ids, use_templates=True)
    
    log.info(f"✅ Generated text prototypes for {len(base_class_ids)} classes")
    
    # 3. 拟合目标范数参数
    log.info("Step 3: Fitting target norm parameters...")
    model.diffusion_model.fit_target_norm(all_features)
    log.info(f"✅ Target norm fitted (mean={model.diffusion_model.target_norm_mean:.4f}, std={model.diffusion_model.target_norm_std:.4f})")
    
    # 4. 训练扩散模型
    log.info("Step 4: Training diffusion model...")
    
    model.diffusion_model.train()
    
    optimizer = torch.optim.AdamW(
        model.diffusion_model.parameters(),
        lr=getattr(args, 'lr_diffusion', 1e-4),
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs_base,
        eta_min=1e-6
    )
    
    # 创建特征数据集
    feature_dataset = torch.utils.data.TensorDataset(all_features, all_labels)
    feature_loader = torch.utils.data.DataLoader(
        feature_dataset,
        batch_size=getattr(args, 'batch_size_diffusion', 256),
        shuffle=True,
        num_workers=0
    )
    
    # 训练循环
    for epoch in range(args.epochs_base):
        epoch_losses = []
        
        for features, labels in feature_loader:
            features = features.cuda()
            labels = labels.cuda()
            
            # 获取对应的文本特征
            text_feats = torch.stack([model.text_prototypes[l.item()] for l in labels])
            
            # 训练扩散模型（增加CFG dropout到20%）
            loss = model.diffusion_model(features, labels, text_feats, cfg_dropout=0.2)

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.diffusion_model.parameters(), max_norm=1.0)
            
            optimizer.step()

            model.diffusion_model.update_ema()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # 定期输出
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            log.info(f"Epoch [{epoch+1}/{args.epochs_base}] Loss: {avg_loss:.6f} LR: {lr:.6f}")
    
    model.diffusion_model.eval()
    log.info("✅ Diffusion training completed!")
    log.info("="*60)


@torch.no_grad()
def test_diffusion(model, testloader, args, session):
    """
    测试扩散FSCIL模型（所有classes都用扩散生成）
    
    Args:
        model: MYNET模型
        testloader: 测试数据加载器
        args: 参数
        session: 当前session
    
    Returns:
        acc_dict: 准确率字典
        acc_list: 准确率列表
    """
    log.info(f"Testing Session {session}...")
    
    test_class = args.base_class + session * args.way
    model.eval()
    
    # 所有类别都用扩散生成原型
    log.info(f"Generating prototypes for {test_class} classes using Diffusion...")
    class_ids = list(range(test_class))
    
    ddim_steps = getattr(args, 'ddim_steps', 50)
    prototypes = model.generate_prototypes(
        class_ids,
        use_cache=True,
        ddim_steps=ddim_steps
    )
    
    log.info(f"✅ Prototypes generated: {prototypes.shape}")
    
    # 测试
    all_similarities = []
    all_labels = []
    
    for images, labels in tqdm(testloader, desc="Testing"):
        images = images.cuda()
        
        # 提取测试特征
        test_features = model.encode_image(images)
        
        # 计算余弦相似度
        test_norm = F.normalize(test_features, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        similarities = test_norm @ proto_norm.T
        
        all_similarities.append(similarities.cpu())
        all_labels.append(labels.cpu())
    
    all_similarities = torch.cat(all_similarities)  # (N, num_classes)
    all_labels = torch.cat(all_labels)  # (N,)
    
    # 计算准确率
    acc_dict, acc_list = accuracy_per_task(
        all_similarities,  # 传入similarities tensor而不是predictions
        all_labels.numpy(),
        init_task_size=args.base_class,
        task_size=args.way
    )
    
    log.info(f"Accuracy: {acc_dict}")
    
    return acc_dict, acc_list


def incremental_learning(model, trainloader, args, session):
    """
    增量学习（Training-Free）
    
    Args:
        model: MYNET模型
        trainloader: 新类数据加载器
        args: 参数
        session: 当前session
    """
    log.info(f"="*60)
    log.info(f"Incremental Learning - Session {session} (Training-Free)")
    log.info("="*60)
    
    # 计算新类的ID范围
    new_class_start = args.base_class + (session - 1) * args.way
    new_class_end = args.base_class + session * args.way
    new_class_ids = list(range(new_class_start, new_class_end))
    
    log.info(f"New classes: {new_class_start} to {new_class_end-1} ({len(new_class_ids)} classes)")
    
    # 1. 生成新类的文本原型
    log.info("Step 1: Generating text prototypes for new classes...")
    model.generate_text_prototypes(new_class_ids, use_templates=True)
    log.info(f"✅ Text prototypes generated")
    
    # 2. 清除视觉原型缓存（可选：提取新类特征用于验证）
    model.clear_cache()
    log.info(f"✅ Visual prototype cache cleared")
    
    # 3. （可选）提取新类的少量样本特征
    log.info("Step 2: Extracting new class features (for analysis)...")
    new_features, new_labels = extract_clip_features(model, trainloader)
    log.info(f"✅ Extracted {len(new_features)} samples for new classes")
    
    log.info("="*60)
    log.info("✅ Incremental learning completed (no parameters updated)")
    log.info("="*60)



def get_logger_str(losses_dict, prefix=""):
    """格式化损失字典为日志字符串"""
    log_str = f"{prefix}"
    for key, val in losses_dict.items():
        if hasattr(val, 'item'):
            log_str += f" {key}={val.item():.4f}"
        else:
            log_str += f" {key}={val:.4f}"
    return log_str


if __name__ == '__main__':
    """简单测试"""
    print("✅ helper.py can be imported")
    print("   Functions available:")
    print("   - extract_clip_features()")
    print("   - train_diffusion_base()")
    print("   - test_diffusion()")
    print("   - incremental_learning()")
