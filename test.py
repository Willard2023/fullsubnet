
from curses import unget_wch
import torch
import torch.nn.functional as F

from speech_enhance.fullsubnet.model.fullsubnet import ungrid

def block(input_tensor, b, W):
    C, M, N = input_tensor.shape
    temp_tensor = torch.zeros(C,M,N)
    for c in range(C):
        for m in range(M):
            for n in range(N-1):
                temp_w = W/b
                index_m = b * (m // temp_w) + (n // b)
                index_n = b * (m % temp_w) + (n % b)
                if index_n >= N:
                    continue
                temp_tensor[c,m,n] = input_tensor[c,int(index_m),int(index_n)]
    return temp_tensor

def unblock(input_tensor, b, W):
    C, M, N = input_tensor.shape
    temp_tensor = torch.zeros(C,M,N)
    for c in range(C):
        for m in range(M):
            for n in range(N-1):
                temp_w = W/b
                index_m = b * (m // temp_w) + (n // b)
                index_n = b * (m % temp_w) + (n % b)
                if index_n >= N:
                    continue
                temp_tensor[c,int(index_m),int(index_n)] = input_tensor[c,m,n]
    return temp_tensor

def grid(input_tensor, b, W):
    C, N, M = input_tensor.shape
    temp_tensor = torch.zeros(C,N,M)
    for c in range(C):
        for n in range(N-1):
            for m in range(M):
                temp_w = W/b
                index_n = b * (m // temp_w) + (n // b)
                index_m = b * (m % temp_w) + (n % b)
                # print(index_n, index_m)
                if index_n >= n:
                    continue
                temp_tensor[c,n,m] = input_tensor[c,int(index_n),int(index_m)]
    return temp_tensor

def ungrid(input_tensor, b, W):
    C, N, M = input_tensor.shape
    temp_tensor = torch.zeros(C,N,M)
    for c in range(C):
        for n in range(N):
            for m in range(M-1):
                temp_w = W/b
                index_m = b * (m // temp_w) + (n // b)
                index_n = b * (m % temp_w) + (n % b)
                if index_n >= N:
                    continue
                temp_tensor[c,int(index_n),int(index_m)] = input_tensor[c,n,m]
    return temp_tensor


def overlap():
    pass
if __name__ == "__main__":
    # 假设输入张量的形状是 (C, L, W)
    C, L, W = 10, 193, 257  # 示例维度
    b = 8  # 块大小
    input_tensor = torch.randn(C, L, W)  # 创建一个随机张量

    # 计算需要补零的数量
    total_elements = L * W
    pad_elements = (b**2 - (total_elements % b**2)) % b**2  # 计算补零个数
    padded_tensor = F.pad(input_tensor.view(C, -1), (0, pad_elements))  # 先展开为 (C, L*W) 进行补零

    # Reshape 成 (C, LW/b^2, b^2)
    output_tensor = padded_tensor.view(C, (total_elements + pad_elements) // b**2, b**2)

    print(output_tensor.shape)  # 输出应该是 (C, 776, 64)
    
    output_tensor = block(output_tensor, b, W)
    print(output_tensor.shape)  # 输出应该是 (C, 776, 64)
    
    output_tensor = unblock(output_tensor, b, W)
    print(output_tensor.shape)  # 输出应该是 (C, 776, 64)
    output_tensor = output_tensor.permute(0, 2, 1)
    output_tensor = grid(output_tensor, b, W)
    print(output_tensor.shape)  # 输出应该是 (C, 193, 257)
    
    output_tensor = ungrid(output_tensor, b, W)
    print(output_tensor.shape)  # 输出应该是 (C, 776, 64)
        # 1. Reshape 成原始形状 (C, L*W)
    flattened_tensor = output_tensor.view(C, -1)

    # 2. 移除补零部分
    original_size_tensor = flattened_tensor[:, :-pad_elements] if pad_elements > 0 else flattened_tensor

    # 3. Reshape 成原来的形状 (C, L, W)
    output_tensor = original_size_tensor.view(C, L, W)
    print(output_tensor.shape)  # 输出应该是 (C, 193, 257)
