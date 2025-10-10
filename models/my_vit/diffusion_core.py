"""
Feature Diffusion Model for FSCIL
在特征空间进行扩散建模，用于生成类别原型
Feature Diffusion Model for FSCIL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class EMA:
    """指数移动平均（Exponential Moving Average）"""
    def __init__(self, decay=0.9999):
        self.decay = decay
    
    def update_model_average(self, ema_model, current_model):
        """更新EMA模型的参数"""
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, new_weight = ema_params.data, current_params.data
            ema_params.data = old_weight * self.decay + (1 - self.decay) * new_weight


class TimeEmbedding(nn.Module):
    """正弦位置编码 - 时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class DenoiseNet(nn.Module):
    """
    改进的去噪网络
    使用残差连接和更大容量
    """
    def __init__(self, feat_dim=512, cond_dim=576, time_dim=128, hidden_dim=2048, num_blocks=6):
        super().__init__()
        
        # 时间嵌入
        self.time_emb = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 输入投影
        input_dim = feat_dim + cond_dim + time_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=0.1) for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feat_dim)
        
        # 初始化输出层为接近零（让模型开始时预测接近零的噪声）
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, t, condition):
        """
        Args:
            x: (B, feat_dim) 加噪特征
            t: (B,) 时间步
            condition: (B, cond_dim) 条件（类别+文本）
        Returns:
            predicted_noise: (B, feat_dim)
        """
        t_emb = self.time_emb(t)
        inp = torch.cat([x, condition, t_emb], dim=-1)
        
        h = self.input_proj(inp)
        for block in self.res_blocks:
            h = block(h)
        
        return self.output_proj(h)


class FeatureDDPM(nn.Module):
    """
    特征空间的去噪扩散概率模型
    用于学习和生成类别原型特征
    """
    
    def __init__(self, feature_dim=512, num_classes=100, num_steps=1000):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # 条件维度 = class_embedding(64) + text_feature(512)
        condition_dim = 64 + feature_dim
        
        # 去噪网络
        self.denoiser = DenoiseNet(
            feat_dim=feature_dim,
            cond_dim=condition_dim,
            time_dim=128
        )
        
        # 噪声调度（余弦调度，对特征空间更友好）
        betas = self._cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为buffer（不参与梯度计算）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 预计算常用系数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        #  特征统计参数（用于后处理范数匹配）
        self.register_buffer('target_norm_mean', torch.tensor(10.0))  # CLIP特征的平均范数
        self.register_buffer('target_norm_std', torch.tensor(2.0))    # CLIP特征范数的标准差
        
        # ⭐ EMA模型
        self.ema_model = deepcopy(self.denoiser)
        self.ema = EMA(decay=0.9999)
        self.ema_start = 2000  # EMA从第2000步开始
        self.ema_update_rate = 1  # 每步都更新
        self.training_step = 0
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        余弦噪声调度
        比线性调度对特征空间更稳定
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def fit_target_norm(self, features):
        """
        计算目标特征的范数统计（用于后处理匹配）
        
        Args:
            features: (N, feature_dim) 所有训练特征
        """
        norms = torch.norm(features, dim=-1)
        self.target_norm_mean.copy_(norms.mean())
        self.target_norm_std.copy_(norms.std())
    
    def match_norm(self, x):
        """
        匹配生成特征的范数到目标分布
        
        策略：将生成特征缩放到目标平均范数
        """
        current_norms = torch.norm(x, dim=-1, keepdim=True)
        target_norms = self.target_norm_mean
        x_matched = x * (target_norms / (current_norms + 1e-8))
        return x_matched
    
    def update_ema(self):
        """更新EMA模型"""
        self.training_step += 1
        if self.training_step % self.ema_update_rate == 0:
            if self.training_step < self.ema_start:
                # 前期直接复制
                self.ema_model.load_state_dict(self.denoiser.state_dict())
            else:
                # 后期使用EMA更新
                self.ema.update_model_average(self.ema_model, self.denoiser)
    
    def add_noise(self, x_0, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        
        Args:
            x_0: (B, feature_dim) 原始特征
            t: (B,) 时间步
            noise: (B, feature_dim) 可选，指定噪声
        
        Returns:
            x_t: 加噪后的特征
            noise: 实际使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t, noise
    
    def forward(self, x_0, class_labels, text_features, t=None, cfg_dropout=0.1):
        """
        训练时的前向传播（支持Classifier-Free Guidance）
        
        Args:
            x_0: (B, feature_dim) 干净的CLIP特征
            class_labels: (B,) 类别标签
            text_features: (B, feature_dim) 文本原型特征
            t: (B,) 可选，时间步（不提供则随机采样）
            cfg_dropout: 条件dropout概率（用于CFG训练）
        
        Returns:
            loss: MSE损失
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device).long()
        
        # 前向扩散：加噪
        x_t, noise = self.add_noise(x_0, t)
        
        #  Classifier-Free Guidance: 随机dropout条件
        # 以cfg_dropout的概率将条件设为零（无条件）
        mask = torch.rand(B, 1, device=device) > cfg_dropout  # (B, 1)
        
        class_emb = self.class_embedding(class_labels)
        condition = torch.cat([class_emb, text_features], dim=-1)
        condition = condition * mask  # 应用mask
        
        # 预测噪声
        predicted_noise = self.denoiser(x_t, t, condition)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, class_labels, text_features, ddim_steps=50, eta=0.0, use_ema=True, debug=False, guidance_scale=3.0):
        """
        DDIM采样 with Classifier-Free Guidance（增强条件控制）
        
        Args:
            class_labels: (C,) 要生成的类别ID
            text_features: (C, feature_dim) 对应的文本特征
            ddim_steps: DDIM采样步数（越大越慢但质量越好）
            eta: DDIM的随机性参数（0=确定性，1=DDPM）
            use_ema: 是否使用EMA模型（推理时应该用）
            debug: 是否打印调试信息
            guidance_scale: CFG引导强度（>1增强条件，1=无引导）
        
        Returns:
            generated_features: (C, feature_dim) 生成的原型特征
        """
        device = class_labels.device
        C = len(class_labels)
        
        # 选择采样时间步（均匀间隔）
        skip = self.num_steps // ddim_steps
        timesteps = list(range(0, self.num_steps, skip))
        timesteps = list(reversed(timesteps))
        
        # 从纯噪声开始
        x_t = torch.randn(C, self.feature_dim, device=device)
        
        if debug:
            print(f"\n[DEBUG] DDIM Sampling with CFG (guidance_scale={guidance_scale}):")
            print(f"  Initial noise norm: {x_t.norm(dim=-1).mean():.4f}")
        
        # 构建有条件和无条件输入
        class_emb = self.class_embedding(class_labels)
        cond_full = torch.cat([class_emb, text_features], dim=-1)
        cond_null = torch.zeros_like(cond_full)  # 无条件（全零）
        
        # 选择使用EMA模型还是普通模型
        model = self.ema_model if use_ema else self.denoiser
        
        # DDIM逆向去噪过程
        for i, t_idx in enumerate(timesteps):
            t = torch.full((C,), t_idx, device=device, dtype=torch.long)
            
            #  Classifier-Free Guidance: 同时预测有条件和无条件的噪声
            if guidance_scale != 1.0:
                # 预测无条件噪声
                noise_uncond = model(x_t, t, cond_null)
                # 预测有条件噪声
                noise_cond = model(x_t, t, cond_full)
                # CFG引导
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # 不使用CFG，直接预测
                predicted_noise = model(x_t, t, cond_full)
            
            # 获取alpha值
            alpha_cumprod = self.alphas_cumprod[t_idx]
            
            if i < len(timesteps) - 1:
                alpha_cumprod_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)
            
            # DDIM更新公式
            # 1. 预测x0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            # 2. 方向指向xt
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - eta**2 * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)) * predicted_noise
            
            # 3. 随机噪声项
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * torch.sqrt(1 - alpha_cumprod / alpha_cumprod_prev)
                x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
            
            
            if debug and (i == 0 or i == len(timesteps) // 2 or i == len(timesteps) - 1):
                print(f"  Step {i}/{len(timesteps)-1} (t={t_idx}): norm={x_t.norm(dim=-1).mean():.4f}")
        
        x_t = self.match_norm(x_t)
        
        if debug:
            print(f"  Final (after norm matching): norm={x_t.norm(dim=-1).mean():.4f}")
        
        return x_t
    
    @torch.no_grad()
    def sample_ddpm(self, class_labels, text_features):
        """
        标准DDPM采样（完整1000步，较慢但质量最高）
        
        仅用于对比实验，正常训练使用DDIM即可
        """
        device = class_labels.device
        C = len(class_labels)
        
        x_t = torch.randn(C, self.feature_dim, device=device)
        
        class_emb = self.class_embedding(class_labels)
        condition = torch.cat([class_emb, text_features], dim=-1)
        
        for t_idx in reversed(range(self.num_steps)):
            t = torch.full((C,), t_idx, device=device, dtype=torch.long)
            
            predicted_noise = self.denoiser(x_t, t, condition)
            
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t_idx]
            
            # 预测x0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            # 后验均值
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # 可选的clip
            mean = torch.sqrt(alpha_cumprod_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
            
            # 添加噪声
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                variance = torch.sqrt(self.betas[t_idx])
                x_t = mean + variance * noise
            else:
                x_t = mean
        
        x_t = self.match_norm(x_t)
        
        return x_t


if __name__ == '__main__':
    """简单测试"""
    print("Testing FeatureDDPM...")
    
    model = FeatureDDPM(feature_dim=512, num_classes=10, num_steps=100).cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 测试训练
    x = torch.randn(8, 512).cuda()
    labels = torch.randint(0, 10, (8,)).cuda()
    text_feat = torch.randn(8, 512).cuda()
    
    loss = model(x, labels, text_feat)
    print(f"Training loss: {loss.item():.4f}")
    
    # 测试采样
    model.eval()
    samples = model.sample(torch.tensor([0, 1, 2]).cuda(), torch.randn(3, 512).cuda(), ddim_steps=20)
    print(f"Sampled features: {samples.shape}")
    
    print("✅ FeatureDDPM test passed!")

