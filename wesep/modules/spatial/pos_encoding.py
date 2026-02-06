import torch
import torch.nn as nn
import math

class CycPosEncoding(nn.Module):
    """
    Cyclic Positional Encoding (Cyc-pos)
    
    Reference: 
    "End-to-End DOA-Guided Speech Extraction in Noisy Multi-Talker Scenarios" (DSENet)
    Eq.(2): Encodes an angle phi into a D-dimensional vector using sin/cos functions.
    """
    def __init__(self, embed_dim, alpha=1.0):
        """
        Args:
            embed_dim (int): 输出的特征维度 D (必须是偶数)。
            alpha (float): 缩放因子 (scaling factor)，论文中提及用于调整周期敏感度 。
                           默认为 1.0。
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
            
        self.embed_dim = embed_dim
        self.alpha = alpha
        
        # 预计算分母项 (div_term)，逻辑与 Transformer 的 PE 一致
        # div_term = 10000^(2j/D)
        # log(div_term) = (2j/D) * log(10000)
        # 1/div_term = exp(- (2j/D) * log(10000))
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # 将 div_term 注册为 buffer，这样它会随模型保存和移动设备 (cpu/gpu)，但不是可学习参数
        self.register_buffer('div_term', div_term)

    def forward(self, angle):
        """
        Args:
            angle (Tensor): 输入角度张量，形状可以是任意形状 (..., )。
                            例如 (B, T) 或 (B, 1)。
                            建议输入单位为弧度或度 (需与 alpha 配合，DSENet未明确单位，通常配合 alpha 使用)。

        Returns:
            Tensor: 编码后的张量，形状为 (..., embed_dim)。
                    例如输入 (B, T) -> 输出 (B, T, D)。
        """
        # 1. 应用缩放因子 alpha 
        x = angle * self.alpha 
        
        # 2. 增加最后一个维度以便广播: (..., 1) * (D/2, ) -> (..., D/2)
        # x.unsqueeze(-1) shape: (..., 1)
        # self.div_term shape: (D/2, )
        phase = x.unsqueeze(-1) * self.div_term
        
        # 3. 初始化输出张量
        # output shape: (..., D)
        output = torch.zeros(*angle.shape, self.embed_dim, device=angle.device, dtype=angle.dtype)
        
        # 4. 填充偶数维 (sin) 和 奇数维 (cos) [cite: 468-469]
        output[..., 0::2] = torch.sin(phase)
        output[..., 1::2] = torch.cos(phase)
        
        return output

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设 Batch=2, Time=100, Embedding Dim=40 (DSENet设置 [cite: 532])
    encoder = CycPosEncoding(embed_dim=40, alpha=1.0)
    
    # 模拟输入角度 (例如方位角)
    dummy_angles = torch.randn(2, 100) 
    
    # 前向计算
    encoded_angles = encoder(dummy_angles)
    
    print(f"Input shape: {dummy_angles.shape}")   # torch.Size([2, 100])
    print(f"Output shape: {encoded_angles.shape}") # torch.Size([2, 100, 40])