import torch
import torch.nn as nn


class GameTimeModel(nn.Module):
    def __init__(self):
        super(GameTimeModel, self).__init__()
        # 输入特征：年龄(标准化), 自控力, 睡眠质量, 压力, 是否周末
        self.linear = nn.Linear(in_features=5, out_features=1)

        # 预设权重和偏置（对应：年龄,自控力,睡眠,压力,周末）
        self.linear.weight.data = torch.tensor([[0.3, 0.5, -0.4, 0.2, 0.3]])
        self.linear.bias.data = torch.tensor([1.5])  # 基础时长

    def forward(self, x):
        # 线性计算：y = xW^T + b
        raw_output = self.linear(x)
        # 限制在0.5~4小时之间
        return torch.clamp(raw_output, min=0.5, max=4.0)


def normalize_input(age, self_control, sleep, stress, is_weekend):
    """将原始输入标准化到模型需要的范围"""
    return torch.tensor([
        age / 100,  # 年龄假设0-100岁→0-1
        self_control / 10,  # 自控力1-10→0.1-1.0
        (5 - sleep) / 5,  # 睡眠质量1-5→0.8-0.0（反向）
        stress / 5,  # 压力1-5→0.2-1.0
        float(is_weekend)  # 周末0/1
    ], dtype=torch.float32)


if __name__ == "__main__":
    # 初始化模型
    model = GameTimeModel()

    # 模拟用户输入（原始值）
    user_input = normalize_input(
        age=55,
        self_control=7,
        sleep=2,  # 睡眠质量3/5
        stress=2,
        is_weekend=True
    )

    # 预测
    recommended_time = model(user_input)
    print(f"推荐游戏时长: {recommended_time.item():.1f} 小时")

    # 显示模型参数
    print("\n模型参数：")
    print("权重:", model.linear.weight.detach().numpy()[0])
    print("偏置:", model.linear.bias.item())