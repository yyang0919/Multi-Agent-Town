from universal_llm import LLMConfig, UniversalLLM


config = LLMConfig.for_vllm(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    temperature=0.2,
    max_tokens=512,
    extra_body={"top_k": 20},
)

client = UniversalLLM(config)
history = [
    ("human", "你是谁？"),
    ("assistant", "我是一个由本地 vLLM 服务提供能力的模型。"),
]

response = client.generate("请简要介绍一下 Transformer。", history=history)
print("普通调用结果：")
print(response.text)
print("\n元信息：")
print(response.response_metadata)
