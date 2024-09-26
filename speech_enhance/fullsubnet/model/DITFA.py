from re import A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepInteractiveTemporalFrequencyAttentionModule(nn.Module):
    def __init__(self, in_channels):
        """
        深度交互时空注意力模块的初始化函数

        参数：
        in_channels (int)：输入特征的通道数
        """
        super(DeepInteractiveTemporalFrequencyAttentionModule, self).__init__()
        self.in_channels = in_channels

        # 定义投影卷积
        self.proj_conv_mask = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.proj_conv_map = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

    def forward(self, x_mask, x_map):
        """
        前向传播函数

        参数：
        x_mask (torch.Tensor)：掩蔽分支的输入特征，形状为 (B, C, L, F)  2, 1, 10, 20
        x_map (torch.Tensor)：映射分支的输入特征，形状为 (B, C, L, F)

        返回：
        output (torch.Tensor)：注意力模块的输出特征
        """
        # 从掩蔽分支到映射分支的注意力
        Q_T_mask = self.proj_conv_mask(x_mask).permute(0, 3, 2, 1)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, F, L)[2, 20, 10, 1]
        Q_F_mask = self.proj_conv_mask(x_mask).permute(0, 2, 3, 1)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, L, F)[2, 10, 20, 1]

        K_T_map = self.proj_conv_map(x_map).permute(0, 3, 1, 2)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, F, L)[2, 20, 1, 10]
        K_F_map = self.proj_conv_map(x_map).permute(0, 2, 1, 3)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, L, F)[2, 10, 1, 20]
        V_T_map = self.proj_conv_map(x_map).permute(0, 3, 2, 1)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, F, L)[2, 20, 10, 1]
        V_F_map = self.proj_conv_map(x_map).permute(0, 2, 3, 1)  # 变换维度为 (B, C/2, L, F) -> (B, C/2, L, F)[2, 10, 20, 1]

        # 计算频率注意力分数矩阵

        A_F = F.softmax(torch.matmul(Q_F_mask, K_F_map) / torch.sqrt(torch.tensor(self.in_channels // 2)), dim=-1) #[2, 10, 20, 20]
        # print(A_F.shape)
        # print(V_F_map.shape)
        # print(V_F_map.shape)
        # A_F1 = torch.einsum('blff,blfc->blfc', A_F, V_F_map)  # 进行张量乘法操作
        A_F2 = torch.matmul(A_F, V_F_map)
        # print(torch.equal(A_F1, A_F2))
        # print(A_F2.shape)
        A_F = A_F2.permute(0, 3, 1, 2)  # 变换维度为 (B, L, F, C/2) -> (B, C/2, L, F)

        # 计算时间注意力分数矩阵，并应用掩码
        # mask = self.get_mask(Q_T_mask.size(-2), 2)  # 获取掩码
        # print(mask)
        
        '''
        例如imgs为
        tensor([[[182., 242.,  11.],
                [163.,  92., 183.],
                [222.,  54.,  86.]],
                [[157., 139., 254.],
                [158., 148.,  46.],
                [  1.,  13.,  56.]]])
        mask为
        tensor([[ True, False, False],
        [False,  True, False],
        [False, False,  True]])
        
        imgs_masked = torch.masked_fill(input=imgs, mask=~mask, value=0) # 这里mask取反：true表示被“遮住的”，也就是被“遮住的”保持不变，其它变为value的值
        imas_masked为
        tensor([[[182.,   0.,   0.],
         [  0.,  92.,   0.],
         [  0.,   0.,  86.]],
        [[157.,   0.,   0.],
         [  0., 148.,   0.],
         [  0.,   0.,  56.]]])

        '''
        Q_T_matmul_K_T_map = torch.matmul(Q_T_mask, K_T_map)
        # masked_fill = torch.masked_fill(torch.matmul(Q_T_mask, K_T_map), ~mask, float('-inf'))  # 应用掩码填充
        # print(masked_fill)
        # print(F.softmax(masked_fill, dim=-1))
        A_T = F.softmax((Q_T_matmul_K_T_map) / torch.sqrt(torch.tensor(self.in_channels // 2)), dim=-1) 

        A_T = torch.einsum('bfll,bflc->bflc', A_T, V_T_map)  # 进行张量乘法操作

        A_T = A_T.permute(0, 3, 2, 1)  # 变换维度为 (B, F, L, C/2) -> (B, C/2, L, F)

        # 连接时空注意力输出
        # output = torch.cat([A_F, A_T], dim=1)  
        output = A_F + A_T
        return output

    def get_mask(self, frame_length, max_temporal_length=2):
        """
        获取掩码矩阵的函数

        参数：
        frame_length (int)：帧长度
        max_temporal_length (int)：最大时间长度（以秒为单位对应的帧数），默认为2

        返回：
        mask (torch.Tensor)：掩码矩阵
        """
        # 创建一个全为1的矩阵
        mask = np.ones((frame_length, frame_length))
        # 将上三角部分（不包括对角线）设置为0
        mask[np.triu_indices(frame_length, k=1)] = 0
        # 将超过最大时间长度的下三角部分设置为0
        mask[np.tril_indices(frame_length, k=-max_temporal_length - 1)] = 0
        mask = torch.tensor(mask).to(self.proj_conv_mask.weight.device)
        # 将掩码转换为布尔类型
        return mask.bool()

# 测试示例
if __name__ == '__main__':
    # 创建注意力模块实例
    attention_module = DeepInteractiveTemporalFrequencyAttentionModule(in_channels=1)

    # 模拟输入特征
    x_mask = torch.randn(2, 1, 10, 20)  # 掩蔽分支特征，假设批次大小为2，通道数为128，帧数为10，频率维度为20
    x_map = torch.randn(2, 1, 10, 20)  # 映射分支特征

    # 进行注意力计算
    output = attention_module(x_mask, x_map)

    # 输出结果形状
    print("Output shape:", output.shape)
    
#     # 示例用法
#     frame_length = 10  # 帧长度
#     max_temporal_length = 2  # 最大时间长度（以秒为单位对应的帧数）

    
#    # 创建一个全为1的矩阵
#     # 创建一个全为0的矩阵
#     # 创建一个全为1的矩阵
#     mask = np.ones((frame_length, frame_length))
#     # 将上三角部分（不包括对角线）设置为负无穷
#     mask[np.triu_indices(frame_length, k=1)] = float('-inf')
#     # 将超过最大时间长度的下三角部分设置为负无穷
#     mask[np.tril_indices(frame_length, k=-max_temporal_length - 1)] = float('-inf')
#     mask = torch.tensor(mask)
#     print(mask)
#     print(F.softmax(mask, dim=-1))