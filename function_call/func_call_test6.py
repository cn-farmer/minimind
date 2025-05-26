import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 模拟工具函数
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_weather(location):
    """获取指定城市的天气（模拟）"""
    weather_data = {
        "location": location,
        "temperature": "25°C",
        "condition": "晴天",
        "humidity": "60%",
        "wind": "10 km/h"
    }
    return json.dumps(weather_data, ensure_ascii=False)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [{"role": "user", "content": "请查询当前时间，杭州天气怎么样"}]

# 第一步：模型生成工具调用请求
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True,
    tool_choice="auto"
)

# 提取工具调用请求
tool_calls = completion.choices[0].message.tool_calls
if tool_calls:
    messages.append(completion.choices[0].message)

    # 第二步：并行执行工具调用
    tool_responses = []
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        if func_name == "get_current_time":
            result = get_current_time()
        elif func_name == "get_current_weather":
            params = json.loads(tool_call.function.arguments)
            result = get_current_weather(params["location"])
        else:
            result = "工具不存在"

        tool_responses.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": result
        })

    # 将工具调用结果添加到消息中
    messages.extend(tool_responses)

    # 第三步：将工具调用结果返回给模型，获取最终回复
    final_completion = client.chat.completions.create(
        model="qwen-plus",
        tools=tools,
        messages=messages
    )

    print(final_completion.choices[0].message.content)
else:
    print(completion.choices[0].message.content)