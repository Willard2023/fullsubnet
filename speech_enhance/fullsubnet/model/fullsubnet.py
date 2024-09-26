import torch

def block(X_B, b, W):
    """
    实现 block 操作

    参数：
    - X_B (torch.Tensor)：输入张量
    - b (int)：块的大小
    - W (int)：输入张量的宽度

    返回：
    - X_B_prime (torch.Tensor)：经过 block 操作后的张量
    """
    W_prime = W // b
    X_B_prime = torch.zeros((X_B.shape[0], X_B.shape[1] * X_B.shape[2] // (b ** 2), b ** 2), device=X_B.device)
    for c in range(X_B.shape[0]):
        for m in range(X_B.shape[1] * X_B.shape[2] // (b ** 2)):
            for n in range(b ** 2):
                row = b * (m // W_prime) + (n // b)
                col = b * (m % W_prime) + (n % b)
                if row < X_B.shape[1] and col < W:
                    X_B_prime[c, m, n] = X_B[c, row, col]
    return X_B_prime

def grid(X_G, b, W):
    """
    实现 grid 操作

    参数：
    - X_G (torch.Tensor)：输入张量
    - b (int)：网格的大小
    - W (int)：输入张量的宽度

    返回：
    - X_G_prime (torch.Tensor)：经过 grid 操作后的张量
    """
    W_prime = W // b
    X_G_prime = torch.zeros((X_G.shape[0], b ** 2, X_G.shape[1] * X_G.shape[2] // (b ** 2)), device=X_G.device)
    for c in range(X_G.shape[0]):
        for n in range(b ** 2):
            for m in range(X_G.shape[1] * X_G.shape[2] // (b ** 2)):
                row = b * (m // W_prime) + (n // b)
                col = b * (m % W_prime) + (n % b)
                if row < X_G.shape[1] and col < W:
                    X_G_prime[c, n, m] = X_G[c, row, col]
    return X_G_prime

def overlap(X_O, b, W, gamma):
    """
    实现 overlap 操作

    参数：
    - X_O (torch.Tensor)：输入张量
    - b (int)：块的大小
    - W (int)：输入张量的宽度
    - gamma (float)：重叠率

    返回：
    - X_O_prime (torch.Tensor)：经过 overlap 操作后的张量
    """
    W_prime = W // b
    padding = (int(gamma * b / 2), int(gamma * b / 2))
    X_O_padded = torch.nn.functional.pad(X_O, padding, 'constant', 0)
    X_O_prime = torch.zeros((X_O.shape[0], X_O.shape[1] * X_O.shape[2] // (b ** 2), int((1 + gamma) ** 2 * b ** 2)), device=X_O.device)
    for c in range(X_O.shape[0]):
        for m in range(X_O.shape[1] * X_O.shape[2] // (b ** 2)):
            for p in range(int((1 + gamma) ** 2 * b ** 2)):
                row = b * (m // W_prime) + (p // ((1 + gamma) * b)) - gamma * b // 2
                col = b * (m % W_prime) + (p % ((1 + gamma) * b)) - gamma * b // 2
                if row >= 0 and row < X_O_padded.shape[1] and col >= 0 and col < X_O_padded.shape[2]:
                    X_O_prime[c, m, p] = X_O_padded[c, int(row), int(col)]
    return X_O_prime

def unblock(X_B_prime, b, W):
    """
    block 操作的逆操作

    参数：
    - X_B_prime (torch.Tensor)：经过 block 操作后的张量
    - b (int)：块的大小
    - W (int)：原始张量的宽度

    返回：
    - X_B (torch.Tensor)：恢复后的张量
    """
    W_prime = W // b
    X_B = torch.zeros((X_B_prime.shape[0], X_B_prime.shape[1] * b, X_B_prime.shape[2] * b), device=X_B_prime.device)
    for c in range(X_B_prime.shape[0]):
        for m in range(X_B_prime.shape[1]):
            for n in range(X_B_prime.shape[2]):
                row = b * m + (n // b)
                col = b * (m % W_prime) + (n % b)
                if row < X_B.shape[1] and col < X_B.shape[2]:
                    X_B[c, row, col] = X_B_prime[c, m, n]
    return X_B

def ungrid(X_G_prime, b, W):
    """
    grid 操作的逆操作

    参数：
    - X_G_prime (torch.Tensor)：经过 grid 操作后的张量
    - b (int)：网格的大小
    - W (int)：原始张量的宽度

    返回：
    - X_G (torch.Tensor)：恢复后的张量
    """
    W_prime = W // b
    X_G = torch.zeros((X_G_prime.shape[0], X_G_prime.shape[2] * b, X_G_prime.shape[1] * b), device=X_G_prime.device)
    for c in range(X_G_prime.shape[0]):
        for n in range(X_G_prime.shape[1]):
            for m in range(X_G_prime.shape[2]):
                row = b * (m // W_prime) + (n // b)
                col = b * (m % W_prime) + (n % b)
                if row < X_G.shape[1] and col < W:
                    X_G[c, row, col] = X_G_prime[c, n, m]
    return X_G

def unoverlap(X_O_prime, b, W, gamma):
    """
    overlap 操作的逆操作

    参数：
    - X_O_prime (torch.Tensor)：经过 overlap 操作后的张量
    - b (int)：块的大小
    - W (int)：原始张量的宽度
    - gamma (float)：重叠率

    返回：
    - X_O (torch.Tensor)：恢复后的张量
    """
    W_prime = W // b
    padding = (int(gamma * b / 2), int(gamma * b / 2))
    X_O = torch.nn.functional.pad(X_O_prime, (-padding[0], -padding[1], -padding[0], -padding[1]), 'constant', 0)
    for c in range(X_O_prime.shape[0]):
        for m in range(X_O_prime.shape[1]):
            for p in range(X_O_prime.shape[2]):
                row = b * (m // W_prime) + (p // ((1 + gamma) * b)) - gamma * b // 2
                col = b * (m % W_prime) + (p % ((1 + gamma) * b)) - gamma * b // 2
                X_O[c, int(row), int(col)] = X_O_prime[c, m, p]
    return X_O

if __name__ == '__main__':

    # 测试数据
    X_in = torch.rand(3, 4, 5)  # 假设输入张量 X_in 的形状为 (C, L, W)
    b = 2
    gamma = 0.5

    # 执行操作
    X_B = X_in.clone()
    X_G = X_in.clone()
    X_O = X_in.clone()

    X_B_prime = block(X_B, b, X_in.shape[2])
    X_G_prime = grid(X_G, b, X_in.shape[2])
    # X_O_prime = overlap(X_O, b, X_in.shape[2], gamma)

    # 恢复操作
    X_B_restored = unblock(X_B_prime, b, X_in.shape[2])
    X_G_restored = ungrid(X_G_prime, b, X_in.shape[2])
    # X_O_restored = unoverlap(X_O_prime, b, X_in.shape[2], gamma)

    # 打印结果
    print("原始张量 X_B:")
    print(X_B.shape)
    print("block 操作后 X_B_prime:")
    print(X_B_prime.shape)
    print("恢复后的 X_B_restored:")
    print(X_B_restored.shape)

    print("原始张量 X_G:")
    print(X_G.shape)
    print("grid 操作后 X_G_prime:")
    print(X_G_prime.shape)
    print("恢复后的 X_G_restored:")
    print(X_G_restored.shape)

    # print("原始张量 X_O:")
    # print(X_O.shape)
    # print("overlap 操作后 X_O_prime:")
    # print(X_O_prime.shape)
    # print("恢复后的 X_O_restored:")
    # print(X_O_restored.shape)