#生成一个函数，用yield关键字输出一个小于5的序列
from typing import Any, Generator


def generate_sequence():
    # for i in range(5):
    #     yield i
    #     print( f'generate - {i}')
    i = 0
    while True:
        yield i
        i += 1

def getGen():
    # gen = generate_sequence()
    # return gen
    for i in generate_sequence():
        value = yield i
        print(f"receive - {value}")
    #以生成器表达式返回一个小于5的序列
    # return (i for i in range(5))


#调用函数，打印序列
sequence: Generator[int, Any, None] = getGen()
next(sequence)
for num in sequence.send(100):
    print(num)