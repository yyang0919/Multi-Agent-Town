from universal_llm import LLMConfig, UniversalLLM


# 本示例演示如何调用本地 vLLM 服务。
# 运行前请先确保 vLLM 已启动，并监听在 http://127.0.0.1:8000/v1
vllm_config = LLMConfig(
    provider="vllm",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    temperature=0.2,
    extra_body={"top_k": 20},
)

client = UniversalLLM(vllm_config)

history = [
    ("human", "你是谁？"),
    ("ai", "我是一个本地部署的大模型助手。"),
]

print(client.chat("请简要介绍一下 Transformer。", history=history))

# 流式输出示例
# for token in client.stream("请用三点介绍 LangChain 的作用"):
#     print(token, end="", flush=True)
