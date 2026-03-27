import os

from universal_llm import LLMConfig, UniversalLLM


# 本示例演示如何调用 OpenAI 兼容接口。
# 当前以 DeepSeek 为例，运行前请先配置环境变量 DEEPSEEK_API_KEY。
config = LLMConfig(
    provider="openai_compatible",
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.2,
)

client = UniversalLLM(config)

history = [
    ("human", "你是谁？"),
    ("ai", "我是一个 AI 助手。"),
]

print(client.chat("请简要介绍一下 Transformer。", history=history))

# 如果你要测试 OpenAI 官方接口，可以改成下面的写法：
# config = LLMConfig(
#     provider="openai",
#     model="gpt-4o-mini",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.2,
# )
# client = UniversalLLM(config)
# print(client.chat("请简要介绍一下 Transformer。"))
