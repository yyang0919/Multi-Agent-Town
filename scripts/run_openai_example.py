from universal_llm import LLMConfig, UniversalLLM


deepseek_config = LLMConfig.for_openai_compatible(
    provider="deepseek",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
    temperature=0.2,
    max_tokens=512,
)

deepseek_client = UniversalLLM(deepseek_config)
deepseek_reply = deepseek_client.generate("请简要介绍一下 Transformer。")

print("DeepSeek 响应：")
print(deepseek_reply.text)
print("\nDeepSeek usage：")
print(deepseek_reply.usage)
