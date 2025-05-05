# -*- coding: utf-8 -*-
import jieba
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 准备数据（假设已下载红楼梦文本为hongloumeng.txt）
with open('../dataset/honglou.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. 中文分词处理
print("正在进行中文分词...")
sentences = []
for line in text.split('\n'):
    if line.strip():
        words = [w for w in jieba.cut(line) if w.strip() and len(w) > 1]  # 过滤单字
        sentences.append(words)

# 3. 训练CBOW模型（sg=0表示使用CBOW算法）
print("训练CBOW模型中...")
model = Word2Vec(sentences, vector_size=100, window=5, min_count=10, sg=0, epochs=20, workers=4)

# 4. 准备主要人物列表
main_characters = ['宝玉', '黛玉', '宝钗', '王熙凤', '贾母', '贾政',
                   '王夫人', '袭人', '晴雯', '探春', '湘云', '妙玉']

# 5. 提取人物向量
print("提取人物向量...")
char_vectors = []
valid_chars = []
for char in main_characters:
    if char in model.wv:
        char_vectors.append(model.wv[char])
        valid_chars.append(char)
    else:
        print(f"警告: 人物'{char}'不在词汇表中")

char_vectors = np.array(char_vectors)

# 6. 使用PCA降维到3维
print("降维处理...")
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(char_vectors)

# 7. 创建3D可视化
print("创建可视化...")
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制点和标签
ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], s=100, alpha=0.8)
for i, char in enumerate(valid_chars):
    ax.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2]+0.02, char,
            fontsize=14, ha='center', va='bottom')

# 绘制连线（基于相似度）
threshold = 0.65  # 相似度阈值
print("绘制人物关系连线...")
for i in range(len(valid_chars)):
    for j in range(i+1, len(valid_chars)):
        similarity = model.wv.similarity(valid_chars[i], valid_chars[j])
        if similarity > threshold:
            ax.plot([vectors_3d[i, 0], vectors_3d[j, 0]],
                    [vectors_3d[i, 1], vectors_3d[j, 1]],
                    [vectors_3d[i, 2], vectors_3d[j, 2]],
                    'gray', alpha=0.4, linewidth=1.5)
            # 显示相似度值
            mid_x = (vectors_3d[i, 0] + vectors_3d[j, 0])/2
            mid_y = (vectors_3d[i, 1] + vectors_3d[j, 1])/2
            mid_z = (vectors_3d[i, 2] + vectors_3d[j, 2])/2
            ax.text(mid_x, mid_y, mid_z, f"{similarity:.2f}",
                   fontsize=9, color='blue')

# 设置图表属性
ax.set_title('《红楼梦》主要人物关系3D可视化（CBOW模型）', fontsize=18, pad=20)
ax.set_xlabel('语义维度1', fontsize=12)
ax.set_ylabel('语义维度2', fontsize=12)
ax.set_zlabel('语义维度3', fontsize=12)

# 添加网格和背景色使更清晰
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle='--', alpha=0.6)

print("显示可视化结果...")
plt.tight_layout()
plt.show()