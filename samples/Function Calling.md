# Function Calling
## 1.结论
>### 1.1、Function Calling能力是微调出来的，不是预训练后醍醐灌顶悟出来的；
>### 1.2、Function需不需要Call是大模型识别的，Call谁是靠翻牌子的（递清单)；
>### 1.3、Function是外部提供的，不是大模型炼化过程中长出来的三头六臂；
>### 1.4、Call是应用触发的，不是大模型绕开部署环境私下搞定的；
>### 1.5、Function的形参是在微调过程中定义并学习的，实参是大模型调用时提炼加工的；
>### 1.6、Function Calling不只是一事一议的，还可以是多任务并行的；
>### 1.7、Function Calling也可以是串行的，而且上一个返回参数可以作为下一个输入参数。

## 2.Function Calling定义
>#### Function Calling 是大模型与外部工具或 API 交互的一种方式，允许模型在生成文本的过程中，动态调用预定义的函数（或工具）来获取实时信息或执行特定任务。

## 3.为什么需要Function Calling
>### 3.1 突破大模型的“知识冻结”问题
>### 3.2 弥补大模型的“计算/逻辑缺陷”
>### 3.3 连接现实世界的“动作执行”
>### 3.4 降低幻觉（Hallucination）风险

## 4.Function Calling调用时序
提示词：请绘制一个网页，在网页上展示function calling的调用时序图

## 5.微调数据集展示

## 6.极简Function Calling代码实现

## 7.Function Calling vs. Tool Calls

## 8.并行Function Calling

## 9.大模型整合Function Calling返回消息
