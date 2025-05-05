class LowerCaseConverter:
    def __or__(self, other):
        if isinstance(other, str):
            return other.lower()
        else:
            raise TypeError("输入必须是字符串")

lower_case = LowerCaseConverter()  # 创建一个实例

# 使用方式
result = lower_case | "Hello World"
print(result)  # 输出: "hello world"