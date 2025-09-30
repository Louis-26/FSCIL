import torch
import torch.nn as nn

def generate_condition_vector_from_current_split(
    feature_extractor: nn.Module,
    current_split_data: torch.Tensor,    # 当前 MEDIC 元任务分裂的输入数据 (N, C, H, W)
    current_split_labels: torch.Tensor,  # 当前 MEDIC 元任务分裂的标签 (N,)
    # num_total_known_classes: int,      # 数据集中已知的总类别数 (用于参考，但这里主要关注分裂中出现的类)
    feature_dim: int,                    # 特征提取器的输出维度
    device: torch.device
) -> torch.Tensor:
    """
    为当前的 MEDIC 元任务分裂生成条件向量。
    条件向量是该分裂中每个出现类别的平均特征嵌入的平均值。

    参数:
    - feature_extractor (nn.Module): 特征提取网络 (例如 ResNet 骨干)。
    - current_split_data (torch.Tensor): 当前分裂的输入图像数据。
    - current_split_labels (torch.Tensor): 当前分裂的图像标签。
    - feature_dim (int): feature_extractor 输出的特征维度。
    - device (torch.device): 计算设备 (例如 'cuda' 或 'cpu')。

    返回:
    - torch.Tensor: 计算得到的条件向量，形状为 (1, feature_dim)。
                      如果当前分裂中没有样本，则返回零向量。
    """
    if current_split_data.nelement() == 0: # 检查输入数据是否为空
        print("警告: 当前分裂数据为空，返回零条件向量。")
        return torch.zeros(1, feature_dim, device=device)

    feature_extractor.eval() # 确保特征提取器处于评估模式
    with torch.no_grad(): # 不需要计算梯度
        # 1. 提取特征
        # feature_extractor 就是骨干网络
        features = feature_extractor(current_split_data) # (N, feature_dim)

    # 2. 识别当前分裂中出现的唯一类别
    present_classes = torch.unique(current_split_labels)

    if present_classes.nelement() == 0: # 如果没有唯一的类别（理论上不应发生）
        print("警告: 当前分裂中没有识别到类别，返回零条件向量。")
        return torch.zeros(1, feature_dim, device=device)

    # 3. 计算每个出现类别的平均特征嵌入 (原型)
    present_class_prototypes = []
    for cls_idx in present_classes:
        # 找到属于当前类别的所有特征
        class_mask = (current_split_labels == cls_idx)
        if not class_mask.any(): # 如果某个在unique中找到的类实际上没有样本（理论上不应发生）
            continue
        class_features = features[class_mask]
        
        # 计算该类别的平均特征 (原型)
        class_prototype = torch.mean(class_features, dim=0) # (feature_dim,)
        present_class_prototypes.append(class_prototype)

    if not present_class_prototypes: # 如果没有收集到任何原型
        print("警告: 未能为当前分裂中的任何类别计算原型，返回零条件向量。")
        return torch.zeros(1, feature_dim, device=device)

    # 4. 将收集到的类别原型堆叠起来
    # (num_present_classes, feature_dim)
    stacked_prototypes = torch.stack(present_class_prototypes) 

    # 5. 计算这些“出现类别原型”的平均值，作为最终的条件向量
    # (feature_dim,)
    condition_vector = torch.mean(stacked_prototypes, dim=0) 
    
    # 6. 增加一个批次维度，以匹配扩散模型通常的输入格式 (1, feature_dim)
    return condition_vector.unsqueeze(0)


# **代码解释和说明：**

# 1.  **函数签名**：
#     * `feature_extractor (nn.Module)`: 这是您的骨干网络，例如 MEDIC 代码中的 `net.net`（如 `resnet50_fast()`）。它负责将输入的图像转换成特征向量。
#     * `current_split_data (torch.Tensor)`: 这是来自特定 MEDIC 元任务分裂（例如，一个元训练集 $(\mathcal{S_{F_{1}}},\mathcal{S_{G_{2}}})$）的图像数据。
#     * `current_split_labels (torch.Tensor)`: 对应的标签。
#     * `feature_dim (int)`: 您的 `feature_extractor` 输出的特征向量的维度。例如，ResNet50 通常是 2048。
#     * `device (torch.device)`: 计算设备。

# 2.  **处理流程**：
#     * **空数据检查**：首先检查输入数据是否为空，如果为空则返回一个零向量作为条件，并打印警告。
#     * **特征提取**：使用 `feature_extractor` 从 `current_split_data` 中提取特征。这里假设 `feature_extractor` 直接输出特征。在 `eval()` 模式和 `torch.no_grad()` 上下文中执行以提高效率并避免不必要的梯度计算。
#     * **识别出现类别**：通过 `torch.unique(current_split_labels)` 找到当前数据分裂中实际出现的所有类别。
#     * **计算类别原型**：遍历这些出现的类别，对每个类别：
#         * 筛选出属于该类别的所有特征。
#         * 计算这些特征的平均值，得到该类别的原型 (prototype)。
#     * **处理无原型情况**：如果在当前分裂中未能为任何类别计算出原型（例如，分裂数据虽非空但标签处理后无有效类别），则返回零向量。
#     * **堆叠原型**：将所有计算得到的类别原型堆叠成一个张量。
#     * **计算最终条件向量**：对堆叠起来的类别原型再次取平均值，得到一个单一的向量。这个向量代表了当前数据分裂中所有出现类别的“中心趋势”。
#     * **调整形状**：使用 `unsqueeze(0)` 将其形状从 `(feature_dim,)` 变为 `(1, feature_dim)`，以匹配扩散模型通常期望的批处理输入格式（即使批大小为1）。

