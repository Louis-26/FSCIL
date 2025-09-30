import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
import argparse

# 辅助函数 (保留)
def one_hot(indices, depth):
    """
    将索引转换为一次性编码（One-Hot Encoding）。
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]), device=indices.device)
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies

def extract(a, t, x_shape):
    """
    从张量中提取指定时间步的值，并调整到目标形状。
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    """
    指数移动平均（EMA）类。
    """
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class SinusoidalPosEmb(nn.Module):
    """将时间步 t 转换为正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """带有时间和条件注入的残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_emb_dim):
        super().__init__()
        # 时间嵌入和条件嵌入的融合层
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_emb_dim, out_channels)
        )
        
        # 主路径的卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 如果输入输出通道数不同，需要一个额外的卷积来匹配维度以便进行残差连接
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.GroupNorm(8, in_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, time_emb, cond_emb):
        # 处理残差连接
        residual = self.residual_conv(x)
        
        # 第一个卷积块
        x = self.norm1(x) # 现在 norm1 的通道数与 x 的通道数匹配
        x = self.activation(x)
        x = self.conv1(x) # 经过 conv1 后，x 的通道数变为 out_channels
        
        # 将时间和条件信息通过加法注入
        x = x + self.time_mlp(time_emb).unsqueeze(-1) + self.condition_mlp(cond_emb).unsqueeze(-1)
        
        # 第二个卷积块
        x = self.norm2(x) # norm2 的通道数与 x 的通道数匹配
        x = self.activation(x)
        x = self.conv2(x)
        
        # 加上残差
        return x + residual

class DownBlock(nn.Module):
    """U-Net的下采样模块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_emb_dim):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim, condition_emb_dim)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb, cond_emb):
        x = self.res_block(x, time_emb, cond_emb)
        skip_connection = x
        x = self.downsample(x)
        return x, skip_connection


class UpBlock(nn.Module):
    """U-Net的上采样模块 (已修复通道不匹配问题)"""
    def __init__(self, in_channels, out_channels, time_emb_dim, condition_emb_dim):
        super().__init__()
        # 上采样层，它的输出通道数是输入通道数的一半
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        
        concatenated_channels = in_channels + (in_channels // 2)
        self.res_block = ResidualBlock(concatenated_channels, out_channels, time_emb_dim, condition_emb_dim)


    def forward(self, x, skip_connection, time_emb, cond_emb):
        # 1. 对输入 x 进行上采样
        x = self.upsample(x)
        
        # 2. 检查并对齐尺寸
        if x.shape[2] != skip_connection.shape[2]:
            x = F.interpolate(x, size=skip_connection.shape[2], mode='linear')
        
        # 3. 拼接跳跃连接的特征
        x = torch.cat([x, skip_connection], dim=1) # 这里的 x 通道数是 concatenated_channels
        
        # 4. 将拼接后的特征通过残差块进行处理
        x = self.res_block(x, time_emb, cond_emb)
        return x


# --- 2. 重新定义 DMFunc，采用 1D U-Net 架构 ---
class DMFunc_Unet1D(nn.Module):
    """
    【U-Net版】适用于 MEDIC 线性层权重生成的条件噪声预测网络。
    """
    def __init__(self, flat_weight_dim, time_embed_dim=128, condition_dim=512):
        super().__init__()
        
        # --- a. 时间和条件嵌入处理 ---
        # 时间编码层
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        # 条件编码层
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_dim * 4, condition_dim)
        )
        
        # --- b. 1D U-Net 结构 ---
        # 初始投影层，将输入的1通道“图像”投影到基础通道数
        base_dim = 64
        self.init_conv = nn.Conv1d(1, base_dim, kernel_size=1)

        # 定义下采样路径 (Encoder)
        self.down1 = DownBlock(base_dim, base_dim * 2, time_embed_dim, condition_dim)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, time_embed_dim, condition_dim)

        # 中间瓶颈层
        self.mid_block = ResidualBlock(base_dim * 4, base_dim * 4, time_embed_dim, condition_dim)

        # 定义上采样路径 (Decoder)
        self.up1 = UpBlock(base_dim * 4, base_dim * 2, time_embed_dim, condition_dim)
        self.up2 = UpBlock(base_dim * 2, base_dim, time_embed_dim, condition_dim)

        # 最终输出层，将特征映射回1通道
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_dim, base_dim, 1),
            nn.SiLU(),
            nn.Conv1d(base_dim, 1, 1)
        )

    def forward(self, x_noisy_flat, time_steps_batch, condition_vector):
        # 0. 预处理输入形状：U-Net需要 (Batch, Channels, Length)
        #    我们将 (B, P) -> (B, 1, P)
        x = x_noisy_flat.unsqueeze(1)
        
        # 1. 计算时间和条件嵌入
        time_emb = self.time_mlp(time_steps_batch)
        cond_emb = self.condition_mlp(condition_vector)
        
        # 2. U-Net 前向传播
        # 初始卷积
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x, time_emb, cond_emb)
        x, skip2 = self.down2(x, time_emb, cond_emb)
        
        # Bottleneck
        x = self.mid_block(x, time_emb, cond_emb)
        
        # Decoder (使用跳跃连接)
        x = self.up1(x, skip2, time_emb, cond_emb)
        x = self.up2(x, skip1, time_emb, cond_emb)
        
        # 最终输出
        x = self.final_conv(x)
        
        # 3. 返回与输入形状匹配的预测噪声
        return x.squeeze(1) # (B, 1, P) -> (B, P)


class GaussianDiffusion_for_Linear(nn.Module): # 重命名以示区分
    def __init__(
            self,
            # denoise_model: DMFunc_for_MEDIC_Linear,
            denoise_model: DMFunc_Unet1D,
            flat_param_dim: int, # 展平后的总参数维度 (weight + bias)
            betas: np.ndarray,
            loss_type="l2",
            ema_decay=0.9999,
            ema_start=2000,
            ema_update_rate=1,
    ):
        super().__init__()
        self.denoise_model = denoise_model # 修改属性名
        self.ema_model = deepcopy(self.denoise_model)
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.flat_param_dim = flat_param_dim
        self._internal_img_channels = 1 
        self._internal_img_size = (flat_param_dim,)

        if loss_type not in ["l1", "l2"]:
            raise ValueError("未知的损失类型")
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1.0 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.denoise_model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.denoise_model)
   


    def _predict_noise(self, x_t_flat, time_steps_batch, condition_vector, use_ema=True):
        model_to_use = self.ema_model if use_ema else self.denoise_model
        return model_to_use(x_t_flat, time_steps_batch, condition_vector)

    def _remove_noise_step(self, x_t_flat_reshaped, time_steps_batch, condition_vector, use_ema=True):
        x_t_flat = x_t_flat_reshaped.squeeze(1)
        predicted_noise_flat = self._predict_noise(x_t_flat, time_steps_batch, condition_vector, use_ema)
        predicted_noise_reshaped = predicted_noise_flat.unsqueeze(1)

        term1_coeff = extract(self.reciprocal_sqrt_alphas, time_steps_batch, x_t_flat_reshaped.shape)
        term2_coeff = extract(self.remove_noise_coeff, time_steps_batch, x_t_flat_reshaped.shape)
        x_prev_flat_reshaped = term1_coeff * (x_t_flat_reshaped - term2_coeff * predicted_noise_reshaped)
        return x_prev_flat_reshaped

    @torch.no_grad()
    def sample(self, batch_size, device, condition_vector, use_ema=True):
        """
        从噪声生成“干净”的展平参数。
        返回: (batch_size, flat_param_dim)
        """
        x_t_flat_reshaped = torch.randn(batch_size, self._internal_img_channels, *self._internal_img_size, device=device)

        # --- 【调试点 1：追踪采样过程】 ---
        print(f"\n--- 调试点 1：追踪采样过程 [DEBUG] sample(): Starting sample process ---")
        print(f"  - Initial random noise norm: {torch.linalg.norm(x_t_flat_reshaped).item():.4f}")

        for t in range(self.num_timesteps - 1, -1, -1):
            time_steps_batch = torch.tensor([t], device=device).repeat(batch_size)
            x_t_flat_reshaped = self._remove_noise_step(x_t_flat_reshaped, time_steps_batch, condition_vector, use_ema)
        
            # --- 【调试点 2.4 继续】 ---
            if t == self.num_timesteps - 1 or t == self.num_timesteps // 2 or t == 0:
                print(f"  - Denoising step t={t}, current weight norm: {torch.linalg.norm(x_t_flat_reshaped).item():.4f}")
            # --------------------------

        # --- 【调试点 1 继续】 ---
        final_generated_tensor = x_t_flat_reshaped.squeeze(1)
        print(f"  - Final generated flat params norm: {torch.linalg.norm(final_generated_tensor).item():.4f}")

        return x_t_flat_reshaped.squeeze(1) # (batch_size, flat_param_dim)

    def perturb_x(self, x_0_flat_reshaped, time_steps_batch, noise_reshaped):
        term1 = extract(self.sqrt_alphas_cumprod, time_steps_batch, x_0_flat_reshaped.shape) * x_0_flat_reshaped
        term2 = extract(self.sqrt_one_minus_alphas_cumprod, time_steps_batch, x_0_flat_reshaped.shape) * noise_reshaped
        return term1 + term2
    def get_denoising_loss(self, clean_params_flat, time_steps_batch, condition_vector):
        """
        计算训练扩散模型（噪声预测器）的损失，并包含详细的调试信息。
        """
        # --- 准备阶段 ---
        # 1. 准备干净的目标 x0 和要添加的真实噪声 ε
        #    将输入的干净参数增加一个“通道”维度，以匹配内部处理格式
        x_0_flat_reshaped = clean_params_flat.unsqueeze(1) # 形状: (B, 1, P)
        
        #    创建一个与 x0 同形状的随机高斯噪声，这是我们希望模型学习预测的目标
        noise_reshaped = torch.randn_like(x_0_flat_reshaped)

        # 2. 执行前向加噪过程，得到带噪输入 xt
        #    调用 self.perturb_x 方法，根据时间步 t，将 noise_reshaped 添加到 x_0_flat_reshaped 上
        perturbed_x_reshaped = self.perturb_x(x_0_flat_reshaped, time_steps_batch, noise_reshaped)
        
        #    移除“通道”维度，以匹配噪声预测网络的输入格式
        perturbed_x_flat = perturbed_x_reshaped.squeeze(1) # 形状: (B, P)
        
        # --- 噪声预测阶段 ---
        # 3. 调用噪声预测网络（DMFunc），得到预测的噪声 ε_theta
        #    训练时总是使用非 EMA 模型
        estimated_noise_flat = self._predict_noise(perturbed_x_flat, time_steps_batch, condition_vector, use_ema=False)
        
        # --- 准备计算损失 ---
        # 4. 准备真实的目标噪声 ε
        target_noise_flat = noise_reshaped.squeeze(1) # 形状: (B, P)

        # --- 【深度调试】: 检查所有关键张量的状态 ---
        if not hasattr(self, 'debug_loss_counter'): self.debug_loss_counter = 0
        
        # 每隔一定步骤打印一次，避免日志刷屏
        if self.debug_loss_counter % 200 == 0:
            # 为了更清晰地调试，我们只看批次中的第一个样本 (index 0)
            idx = 0
            
            # 获取第一个样本对应的时间步
            time_step_sample = time_steps_batch[idx].item()

            # 提取第一个样本的各个张量
            x0_sample = clean_params_flat[idx]
            noise_sample = target_noise_flat[idx]
            xt_sample = perturbed_x_flat[idx]
            est_noise_sample = estimated_noise_flat[idx]
            cond_sample = condition_vector[idx]

            print(f"\n--- [DEBUG] get_denoising_loss (Call: {self.debug_loss_counter}, Sample Index: {idx}, TimeStep: {time_step_sample}) ---")
            
            # # 1. 检查输入是否有效 (非 NaN/Inf)
            # print(f"  - Is Valid: x0={torch.all(torch.isfinite(x0_sample))}, noise={torch.all(torch.isfinite(noise_sample))}, xt={torch.all(torch.isfinite(xt_sample))}, cond={torch.all(torch.isfinite(cond_sample))}")
            
            # # 2. 检查数值范围和统计量
            # print(f"  - Norms: Clean(x0)={torch.linalg.norm(x0_sample):.2f}, Noise(ε)={torch.linalg.norm(noise_sample):.2f}, Noisy(xt)={torch.linalg.norm(xt_sample):.2f}, Condition={torch.linalg.norm(cond_sample):.2f}")
            
            # 3. 检查模型预测
            print(f"  - Prediction: Estimated_Noise_Norm={torch.linalg.norm(est_noise_sample).item():.4f}, Is_Valid={torch.all(torch.isfinite(est_noise_sample))}")

            print(f"  - Target Noise Norm: {torch.linalg.norm(target_noise_flat).item():.4f}")
            print(f"  - Estimated Noise Norm: {torch.linalg.norm(estimated_noise_flat).item():.4f}")
            # 4. 比较预测和目标的相似度与差异
            #    余弦相似度可以衡量方向是否一致。如果模型学到了东西，它应该大于0。
            #    MAE 提供了比 MSE 更直观的平均误差。
            cos_sim = F.cosine_similarity(est_noise_sample, noise_sample, dim=0).item()
            mae = F.l1_loss(est_noise_sample, noise_sample).item()
            print(f"  - Comparison: CosineSimilarity={cos_sim:.4f}, MAE={mae:.4f}")
            print("-------------------------------------------------------------------------")

        self.debug_loss_counter += 1
        # --- (结束) 深度调试 ---

        # 计算最终的损失函数
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise_flat, target_noise_flat)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise_flat, target_noise_flat)
        else:
            raise NotImplementedError()
        return loss


    # 这个 forward 方法是用来训练扩散模型本身的
    def forward(self, clean_params_flat_batch, condition_vector):
        b_size = clean_params_flat_batch.shape[0]
        device = clean_params_flat_batch.device
        time_steps_batch = torch.randint(0, self.num_timesteps, (b_size,), device=device).long()
        return self.get_denoising_loss(clean_params_flat_batch, time_steps_batch, condition_vector)

# 创建扩散模型实例 ---
def create_diffusion_for_linear_layer(
    in_features: int,
    out_features: int,
    condition_dim: int,
    num_timesteps=1000, # MEDIC中可以尝试用更少的时间步，如100-200
    schedule_type="linear",
    loss_type="l2",
    ema_decay=0.9999,
    ema_start=100, # 调整EMA开始时机
    time_embed_dim_for_dmfunc=128, # 减小一点以匹配简化的DMFunc
    hidden_dim_scale_factor_for_dmfunc=0.5 # 减小一点以匹配简化的DMFunc
    ):
    flat_param_dim = (out_features * in_features) + out_features # weight_dim + bias_dim
    print(f"--- 扩散模型参数维度：{flat_param_dim} ---")
    
    # 实例化新的 U-Net 模型
    denoise_model = DMFunc_Unet1D(
        flat_weight_dim=flat_param_dim, # 注意：U-Net内部不直接使用这个，但保留接口一致性
        time_embed_dim=time_embed_dim_for_dmfunc,
        condition_dim=condition_dim
    )

    if schedule_type == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    elif schedule_type == "linear":
        schedule_low = 1e-4
        schedule_high = 0.02 # 对于权重生成，beta_end可能需要更小
        betas = generate_linear_schedule(num_timesteps, schedule_low, schedule_high)
    else:
        raise ValueError(f"未知的调度类型: {schedule_type}")

    diffusion_model_instance = GaussianDiffusion_for_Linear(
        denoise_model=denoise_model,
        flat_param_dim=flat_param_dim,
        betas=betas,
        loss_type=loss_type,
        ema_decay=ema_decay,
        ema_start=ema_start,
    )
    return diffusion_model_instance


def generate_cosine_schedule(T, s=0.008):
    def f(t, T_val):
        return (np.cos((t / T_val + s) / (1 + s) * np.pi / 2)) ** 2
    alphas_cumprod = []
    f0 = f(0, T)
    for t_val in range(T + 1):
        alphas_cumprod.append(f(t_val, T) / f0)
    
    betas = []
    for t_val in range(1, T + 1):
        beta = min(1 - alphas_cumprod[t_val] / alphas_cumprod[t_val - 1], 0.999)
        # 确保 beta > 0，避免 log(0) 或除以0的问题
        # 在DDPM中，beta通常被限制在一个小范围内，例如不小于1e-6
        beta = max(beta, 1e-6) 
        betas.append(beta)
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

