class Demo:
    def __new__(cls):
        print("__new__ 被调用（创建实例）")
        return super().__new__(cls)

    def __init__(self):
        print("__init__ 被调用（初始化实例）")

    def __call__(self):
        print("__call__ 被调用（实例像函数一样调用）")


# 演示
obj = Demo()  # 先调用__new__，再调用__init__
obj()  # 调用__call__