import torch
import torch.nn as nn
import torch.nn.functional as F


class DietClassifier(nn.Module):
    def __init__(self):
        super(DietClassifier, self).__init__()
        # 输入特征：辣椒摄入量, 酸菜频率, 腌制品量, 炖菜频率, 米饭占比
        self.linear = nn.Linear(5, 3)  # 5特征→3省份

        # 预设权重（对应：安徽,辽宁,四川）
        self.linear.weight.data = torch.tensor([
            [0.3, -0.8, 1.2],  # 辣椒权重
            [-0.5, 1.5, -0.2],  # 酸菜权重
            [1.1, 0.2, 0.4],  # 腌制品权重
            [0.7, 1.3, -0.5],  # 炖菜权重
            [0.6, -0.4, 0.8]  # 米饭权重
        ], dtype=torch.float32).T

        self.linear.bias.data = torch.zeros(3)  # 偏置设为0

    def forward(self, x):
        logits = self.linear(x)
        return F.softmax(logits, dim=0)  # 转换为概率


# 省份标签
PROVINCES = ["安徽", "辽宁", "四川"]


def normalize_input(spicy, sauerkraut, pickled, stew, rice):
    """简单标准化（实际应用需更精细处理）"""
    return torch.tensor([
        spicy / 100,  # 辣椒摄入量假设0-100g
        sauerkraut / 7,  # 酸菜频率（次/周）
        pickled / 10,  # 腌制品量（份/周）
        stew / 7,  # 炖菜频率（次/周）
        rice / 100  # 米饭占比（%）
    ], dtype=torch.float32)


if __name__ == "__main__":
    model = DietClassifier()

    # 模拟用户输入（原始值）
    user_input = normalize_input(
        spicy=0,  # 辣椒80g/天
        sauerkraut=7,  # 酸菜1次/周
        pickled=3,  # 腌制品3份/周
        stew=7,  # 炖菜2次/周
        rice=90  # 主食90%是米饭
    )

    # 预测
    prob = model(user_input)

    # 输出结果
    print("省份预测概率：")
    for province, p in zip(PROVINCES, prob):
        print(f"{province}: {p.item() * 100:.1f}%")

    # 显示最可能省份
    pred_idx = torch.argmax(prob).item()
    print(f"\n预测结果：{PROVINCES[pred_idx]}")