import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

def silu(x):
    return x * sigmoid(x)

# 生成x轴数据
x = np.linspace(-3, 3, 500)

# 计算各激活函数值
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)
y_silu = silu(x)

# 创建图形和子图
plt.figure(figsize=(14, 8))

# 绘制所有曲线在一个图中
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue', linewidth=2)
plt.plot(x, y_relu, label='ReLU', color='red', linewidth=2)
plt.plot(x, y_leaky_relu, label='Leaky ReLU (α=0.1)', color='purple', linewidth=2)
plt.plot(x, y_tanh, label='Tanh', color='green', linewidth=2)
plt.plot(x, y_silu, label='SiLU/Swish', color='orange', linewidth=2)

# 添加标题和标签
plt.title('Common Activation Functions', fontsize=16, pad=20)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

# 设置坐标轴范围
plt.xlim(-3, 3)
plt.ylim(-1.2, 1.5)

# 添加特殊点标记
plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
plt.axvline(0, color='black', linewidth=0.5, linestyle='-')

plt.tight_layout()
plt.show()

# 可选：分开绘制每个函数的子图
plt.figure(figsize=(14, 8))
functions = [
    ('Sigmoid', y_sigmoid, 'blue'),
    ('ReLU', y_relu, 'red'),
    ('Leaky ReLU', y_leaky_relu, 'purple'),
    ('Tanh', y_tanh, 'green'),
    ('SiLU/Swish', y_silu, 'orange')
]

for i, (name, y, color) in enumerate(functions, 1):
    plt.subplot(2, 3, i)
    plt.plot(x, y, color=color, linewidth=2)
    plt.title(name, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(-3, 3)
    if name in ['ReLU', 'Leaky ReLU']:
        plt.ylim(-0.5, 3)
    else:
        plt.ylim(-1.2, 1.5)

plt.tight_layout()
plt.suptitle('Activation Functions (Individual Plots)', y=1.02, fontsize=16)
plt.show()