# reference: https://github.com/thuml/OpenDG-DAML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torch.nn.parameter import Parameter
import math

from model.diff_grad import create_diffusion_for_linear_layer

Parameter.fast = None

# 自定义线性层，支持快速权重更新
# --- 新的线性层，包含扩散模型 ---
class Linear_fw_Diffusion(nn.Linear):
    def __init__(self, in_features, out_features, condition_dim, diffusion_config=None, bias=True, device=None, dtype=None):
        # 先调用父类初始化，这样 self.weight 和 self.bias 会被创建
        super(Linear_fw_Diffusion, self).__init__(in_features, out_features, bias, device, dtype)
        
        self.in_features = in_features
        self.out_features = out_features
        self.condition_dim = condition_dim
        self.use_bias = bias

        # # 初始化快速权重属性
        # self.weight.fast = None
        # if self.bias is not None:
        #     self.bias.fast = None
        
        # 创建并存储扩散模型实例
        # diffusion_config 是一个字典，包含创建扩散模型的参数
        # 例如: {'num_timesteps': 200, 'schedule_type': 'cosine', ...}
        config = diffusion_config if diffusion_config is not None else {}
        self.diffusion_model = create_diffusion_for_linear_layer(
            in_features=in_features,
            out_features=out_features,
            condition_dim=condition_dim,
            **config # 解包配置参数
        )
        self.target_w_shape = (out_features, in_features)
        self.target_b_shape = (out_features,) if bias else None


    def generate_and_set_weights(self, condition_vector: torch.Tensor, device: torch.device, use_ema=True):
        """
        使用扩散模型生成权重和偏置，并设置到 self.weight.fast 和 self.bias.fast。
        condition_vector: (batch_size_cond, condition_dim)
                        如果 batch_size_cond > 1, 表示为多个输入样本生成多套权重 (不常见于单层)
                        通常对于一个线性层实例，我们一次只关心一套权重，所以 batch_size_cond 应该是1
        """
        # print(f"--- [DEBUG] 条件向量 In generate_and_set_weights ---")
        # print(f"  - Condition vector norm: {torch.linalg.norm(condition_vector).item():.4f}")
        if condition_vector.ndim == 1:
            condition_vector = condition_vector.unsqueeze(0) # (1, condition_dim)
        
        batch_size_to_sample = condition_vector.shape[0]

        # 从扩散模型采样展平的参数
        # self.diffusion_model.sample 返回 (batch_size_to_sample, flat_param_dim)
        flat_params_generated = self.diffusion_model.sample(
            batch_size=batch_size_to_sample,
            device=device,
            condition_vector=condition_vector.to(device), # 确保条件向量在正确设备上
            use_ema=use_ema
        )


        if batch_size_to_sample != 1:
            # 这个场景比较复杂，因为一个nn.Linear实例通常只有一套权重。
            # 如果要为批次中的每个样本生成不同权重，Linear_fw_Diffusion的forward需要大改，
            # 或者外部循环处理。目前假设一次只处理一套权重。
            print("警告: generate_and_set_weights 期望为单个实例生成权重 (batch_size=1 for sampling)")
            # 取第一套生成的权重
            flat_params_single = flat_params_generated[0]
        else:
            flat_params_single = flat_params_generated.squeeze(0) # (flat_param_dim,)

        # 反向展平 (Unflatten)
        num_weight_params = self.out_features * self.in_features
        
        generated_w = flat_params_single[:num_weight_params].view(self.target_w_shape).detach()

        self.weight.data = generated_w

        if self.use_bias:
            generated_b = flat_params_single[num_weight_params:].view(self.target_b_shape).detach()
            if self.bias is not None:
                self.bias.data = generated_b

        # self.weight.fast = generated_w # 设置到 .fast 属性
        # # self.weight.data = generated_w # 或者直接修改 .data

        # if self.use_bias:
        #     generated_b = flat_params_single[num_weight_params:].view(self.target_b_shape).detach()
        #     if self.bias is not None:
        #         self.bias.fast = generated_b
        #         # self.bias.data = generated_b
        # else:
        #     if self.bias is not None:
        #         self.bias.fast = None # 如果不使用偏置，但父类创建了，则清空

    def train_diffusion_step(self, target_clean_weight: torch.Tensor, target_clean_bias: torch.Tensor, condition_vector: torch.Tensor, optimizer_diffusion):
        """
        执行一步扩散模型的训练。
        target_clean_weight: (out_features, in_features) - 理想的干净权重。
        target_clean_bias: (out_features,) - 理想的干净偏置。
        condition_vector: (condition_dim,) or (1, condition_dim) - MEDIC 条件。
        optimizer_diffusion: 专门用于优化扩散模型参数的优化器。
        """

        if condition_vector.ndim == 1:
            condition_vector = condition_vector.unsqueeze(0) # (1, condition_dim)

        # 1. 展平目标权重和偏置
        flat_target_w = target_clean_weight.flatten()
        if self.use_bias and target_clean_bias is not None:
            flat_target_b = target_clean_bias.flatten()
            clean_params_flat = torch.cat([flat_target_w, flat_target_b]).unsqueeze(0) # (1, flat_param_dim)
        else:
            clean_params_flat = flat_target_w.unsqueeze(0) # (1, flat_param_dim)
        # print(f"  -ooo Clean params flat shape: {clean_params_flat.shape}")

        torch.cuda.empty_cache()  # 释放所有未被占用的缓存块
        # 2. 计算去噪损失
        # diffusion_model.forward 期望 (batch_size, flat_param_dim)
        denoising_loss = self.diffusion_model(clean_params_flat, condition_vector)

        # 3. 反向传播和优化
        optimizer_diffusion.zero_grad()
        denoising_loss.backward()

        # 加入梯度裁剪】 ---
        # 作用：在优化器更新参数之前，对所有扩散模型参数的梯度进行裁剪。
        # torch.nn.utils.clip_grad_norm_ 会计算所有梯度的总范数，
        # 如果这个总范数超过了 max_norm (这里设为1.0)，它会按比例缩小所有梯度，
        # 使得总范数等于 max_norm。这可以有效防止梯度爆炸。
        torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
        # --- (结束) 核心修改点 ---
        optimizer_diffusion.step()
        
        # 4. 更新EMA模型
        self.diffusion_model.update_ema()
        
        return denoising_loss.item()

# 自定义卷积层，支持快速权重更新
class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        # 调用父类初始化
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)

    def forward(self, x):
        # 如果没有偏置项
        if self.bias is None:
            # 如果存在快速权重，使用快速权重进行卷积
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                # 否则使用普通权重进行卷积
                out = super(Conv2d_fw, self).forward(x)
        else:
            # 如果存在快速权重和快速偏置，使用它们进行卷积
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                # 否则使用普通权重和偏置进行卷积
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        # 计算指数平均因子
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # 在训练模式下更新跟踪的batch数量
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        """ 决定是否使用mini-batch统计数据进行归一化，而不是使用缓冲区。
        在训练模式下使用mini-batch统计数据，在评估模式下当缓冲区为None时也使用mini-batch统计数据。
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """缓冲区仅在需要跟踪且处于训练模式时更新。因此它们只需要在更新发生时传递（即在训练模式下跟踪时），
        或者当使用缓冲区统计数据进行归一化时传递（即在评估模式下缓冲区不为None时）。
        """

        # 如果存在快速权重和快速偏置，使用它们进行批归一化
        if self.weight.fast is not None and self.bias.fast is not None:
            return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight.fast, self.bias.fast, bn_training, exponential_average_factor, self.eps)
        else:
            # 否则使用普通权重和偏置进行批归一化
            return F.batch_norm(
                input,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 扩展因子，用于调整输出通道数
    expansion = 4

    def __init__(
        self,
        inplanes,  # 输入通道数
        planes,    # 中间层通道数
        stride=1,  # 卷积步长
        downsample=None,  # 下采样层
    ):
        super(Bottleneck, self).__init__()
        # 1x1卷积，用于降维
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        # 3x3卷积，用于特征提取
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)  # 批归一化层
        # 1x1卷积，用于升维
        self.conv3 = Conv2d_fw(in_channels=planes, out_channels=planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d_fw(planes*self.expansion)  # 批归一化层
        self.downsample = downsample  # 下采样层
        self.stride = stride  # 卷积步长

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x

        # 第一层：1x1卷积 + 批归一化 + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层：3x3卷积 + 批归一化 + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三层：1x1卷积 + 批归一化
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果需要下采样，对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接
        out += residual
        # 最终ReLU激活
        out = self.relu(out)

        return out



class ResNetFast(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFast, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_fw(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = Linear_fw_Diffusion(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_fw(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_fw(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d_fw(64, 64, kernel_size=3, stride=1, padding=1)

        self._out_features = 256

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)

        return x


class MutiClassifier(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512, condition_dim = None, diffusion_config=None):
        super(MutiClassifier, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.classifier = Linear_fw_Diffusion(feature_dim, self.num_classes, condition_dim, diffusion_config)
        self.b_classifier = Linear_fw_Diffusion(feature_dim, self.num_classes*2, condition_dim, diffusion_config)

        # self.classifier = Linear_fw_Diffusion(feature_dim, self.num_classes)
        # self.b_classifier = Linear_fw_Diffusion(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.classifier.weight, .1)
        nn.init.constant_(self.classifier.bias, 0.)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)
        x1 = self.classifier(x)
        x2 = self.b_classifier(x)
        return x1, x2


class MutiClassifier_(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier_, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.b_classifier = Linear_fw_Diffusion(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        x = x.view(x.size(0), 2, -1)
        x = x[:, 1, :]
            
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)   
        x2 = self.b_classifier(x)
        x1 = x2.view(x.size(0), 2, -1)
        x1 = x1[:, 1, :]
        return x1, x2

# ========== 新增：原型分类器 ===========
class ProtoClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
    def forward(self, x):
        # x: (batch, feature_dim)
        # 计算每个样本到每个 prototype 的距离（欧氏距离）
        dists = torch.cdist(x, self.prototypes)  # (batch, num_classes)
        return -dists

# ========== 新增：多任务原型分类器，集成扩散模型 ===========
class MutiProtoClassifier(nn.Module):
    def __init__(self, net, num_classes, feature_dim, condition_dim, diffusion_config=None):
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.proto_classifier = ProtoClassifier(feature_dim, num_classes)
        # 扩散模型，输入/输出为 (num_classes, feature_dim)
        self.diffusion_model = create_diffusion_for_linear_layer(
            in_features=feature_dim,
            out_features=num_classes,
            condition_dim=condition_dim,
            **(diffusion_config or {})
        )
    def forward(self, x):
        x = self.net(x)
        return self.proto_classifier(x)
    # 生成并设置 prototype
    def generate_and_set_prototypes(self, condition_vector, device, use_ema=True):
        flat_proto = self.diffusion_model.sample(
            batch_size=1,
            device=device,
            condition_vector=condition_vector.to(device),
            use_ema=use_ema
        ).squeeze(0)
        self.proto_classifier.prototypes.data = flat_proto.view(self.num_classes, -1)
    # 训练扩散模型
    def train_diffusion_step(self, target_proto, condition_vector, optimizer_diffusion):
        flat_target = target_proto.flatten().unsqueeze(0)
        loss = self.diffusion_model(flat_target, condition_vector.unsqueeze(0))
        optimizer_diffusion.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
        optimizer_diffusion.step()
        self.diffusion_model.update_ema()
        return loss.item()

# ========== 新增：特征空间扩散模型 ===========
class FeatureDiffusion(nn.Module):
    def __init__(self, feature_dim, num_timesteps=100, hidden_dim=512, condition_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        self.condition_dim = condition_dim
        # MLP输入维度根据是否有condition调整
        input_dim = feature_dim + 1 + (condition_dim if condition_dim else 0)
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, noisy_feature, t, condition=None):
        # noisy_feature: [B, D], t: [B] or int, condition: [B, C] or None
        if isinstance(t, int):
            t = torch.full((noisy_feature.size(0), 1), t, device=noisy_feature.device, dtype=noisy_feature.dtype)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        t_norm = t.float() / self.num_timesteps
        inp = [noisy_feature, t_norm]
        if condition is not None:
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            if condition.shape[0] != noisy_feature.shape[0]:
                condition = condition.expand(noisy_feature.shape[0], -1)
            inp.append(condition)
        inp = torch.cat(inp, dim=1)
        return self.denoiser(inp)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        beta = 0.02 + 0.98 * t.float() / self.num_timesteps
        beta = beta.view(-1, 1)
        return (1 - beta).sqrt() * x_start + beta.sqrt() * noise

# ========== 新增：主网络，集成特征扩散 ===========
class FeatureDiffusionNet(nn.Module):
    def __init__(self, backbone, classifier, feature_dim, diffusion_steps=100, hidden_dim=512, multi_scale=False, num_classes=None):
        super().__init__()
        self.backbone = backbone
        self.multi_scale = multi_scale
        if multi_scale:
            self.feature_dims = [128, 256, 512]
            self.feature_diffusions = nn.ModuleList([
                FeatureDiffusion(dim, num_timesteps=diffusion_steps, hidden_dim=hidden_dim)
                for dim in self.feature_dims
            ])
            self.diffusion_steps = diffusion_steps
            assert num_classes is not None, 'num_classes must be provided for multi_scale.'
            self.classifier = nn.Linear(sum(self.feature_dims), num_classes)
            # 新增：每层一个可学习权重，初始化为1
            self.scale_weights = nn.Parameter(torch.ones(len(self.feature_dims)))
        else:
            self.feature_diffusion = FeatureDiffusion(feature_dim, num_timesteps=diffusion_steps, hidden_dim=hidden_dim)
            self.diffusion_steps = diffusion_steps
        self.classifier = classifier

    def forward(self, x, t=None, noise=None, return_feature=False):
        if self.multi_scale:
            feats = self.backbone(x, return_multi=True)  # list of [B, D]
            B = feats[0].shape[0]
            if t is None:
                t = torch.randint(0, self.diffusion_steps, (B,), device=feats[0].device)
            if noise is None:
                noise = [torch.randn_like(f) for f in feats]
            noisy_feats = [fd.q_sample(f, t, n) for fd, f, n in zip(self.feature_diffusions, feats, noise)]
            denoised_feats = [fd(nf, t) for fd, nf in zip(self.feature_diffusions, noisy_feats)]
            # 新增：加权融合
            scaled_feats = [w * f for w, f in zip(self.scale_weights, denoised_feats)]
            concat_feat = torch.cat(scaled_feats, dim=1)
            logits = self.classifier(concat_feat)
            if return_feature:
                return logits, feats, noisy_feats, denoised_feats, t, noise
            return logits
        else:
            feat = self.backbone(x)
            B, D = feat.shape
            if t is None:
                t = torch.randint(0, self.diffusion_steps, (B,), device=feat.device)
            if noise is None:
                noise = torch.randn_like(feat)
            noisy_feat = self.feature_diffusion.q_sample(feat, t, noise)
            denoised_feat = self.feature_diffusion(noisy_feat, t)
            logits = self.classifier(denoised_feat)
            if return_feature:
                return logits, feat, noisy_feat, denoised_feat, t, noise
            return logits

def resnet18_fast(pretrained=True, progress=True): # 重新添加 pretrained 参数以控制是否加载权重
    """ResNet-18 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr (新API中由Weights对象处理)
    """
    # 假设 ResNetFast 是您自定义的 ResNet 类
    # ResNetFast, BasicBlock 需要在此作用域内可用
    model = ResNetFast(BasicBlock, [2, 2, 2, 2]) # 假设 ResNetFast 的构造函数与标准 ResNet 类似

    if pretrained:
        # 获取预训练权重
        weights = ResNet18_Weights.IMAGENET1K_V1 # 或者选择 ResNet18_Weights.DEFAULT 获取最新推荐权重
        state_dict = weights.get_state_dict(progress=progress)

        # 加载权重，strict=False 允许模型只加载匹配的键
        # 这对于自定义模型 ResNetFast 尤其重要
        model.load_state_dict(state_dict, strict=False)

    if hasattr(model, 'fc'): # 检查fc层是否存在再删除
        del model.fc

    return model


def resnet50_fast(pretrained=True, progress=True): # 重新添加 pretrained 参数
    """ResNet-50 model from
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

    Parameters:
        - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
        - **progress** (bool): If True, displays a progress bar of the download to stderr
    """
    # ResNetFast 自定义的 ResNet 类
    model = ResNetFast(Bottleneck, [3, 4, 6, 3]) # 假设 ResNetFast 的构造函数与标准 ResNet 类似

    if pretrained:
        # 获取预训练权重
        weights = ResNet50_Weights.IMAGENET1K_V2 # 或者选择 .IMAGENET1K_V1 或 .DEFAULT
        state_dict = weights.get_state_dict(progress=progress)

        # 加载权重，strict=False
        model.load_state_dict(state_dict, strict=False)

    if hasattr(model, 'fc'): # 检查fc层是否存在再删除
        del model.fc

    return model

# def resnet18_fast(progress=True):
#     """ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Parameters:
#         - **pretrained** (bool): If True, returns a model pre-trained on ImageNet
#         - **progress** (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = ResNetFast(BasicBlock, [2, 2, 2, 2])
#     state_dict = load_state_dict_from_url(model_urls['resnet18'],
#                                           progress=progress)
#     model.load_state_dict(state_dict, strict=False)
#     del model.fc

#     return model


# def resnet50_fast(progress=True):
#     model = ResNetFast(Bottleneck, [3, 4, 6, 3])
#     state_dict = load_state_dict_from_url(model_urls['resnet50'],
#                                           progress=progress)
#     model.load_state_dict(state_dict, strict=False)
#     del model.fc

#     return model

    


