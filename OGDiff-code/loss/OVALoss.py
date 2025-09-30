import torch 
from torch import nn
from torch.nn import functional as F

class OVALoss(nn.Module):
    def __init__(self):
        super(OVALoss, self).__init__()

    def forward(self, input, label):
        # 检查输入张量的维度是否为3，且第二维度为2
        assert len(input.size()) == 3
        assert input.size(1) == 2

        # 对输入进行softmax归一化处理
        input = F.softmax(input, 1)
        
        # 创建与输入相同大小的零张量，用于存储正类标签
        label_p = torch.zeros((input.size(0),
                           input.size(2))).long().cuda()
        
        # 生成索引范围，用于后续的标签赋值
        label_range = torch.range(0, input.size(0) - 1).long()
        
        # 将正类标签位置设为1
        label_p[label_range, label] = 1
        
        # 负类标签为1减去正类标签
        label_n = 1 - label_p
        
        # 计算正类损失：对正类预测值取对数并加权平均
        open_loss_pos = torch.mean(torch.sum(-torch.log(input[:, 1, :]
                                                    + 1e-8) * label_p, 1))
        
        # 计算负类损失：对负类预测值取对数并取最大值
        open_loss_neg = torch.mean(torch.max(-torch.log(input[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
        
        # 返回正负类损失的平均值
        return 0.5*(open_loss_pos + open_loss_neg)
