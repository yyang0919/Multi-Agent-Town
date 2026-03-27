from universal_llm import LLMConfig, UniversalLLM


client = UniversalLLM(
    LLMConfig.for_vllm(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        base_url="http://127.0.0.1:8000/v1",
    )
)

print(client.chat("请用一句话介绍 LangChain。"))
