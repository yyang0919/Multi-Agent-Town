# Universal LLM

基于 **LangChain** 的通用 LLM 调用项目，支持：

- LangChain 原生 provider
- OpenAI-compatible provider
- 本地 **vLLM** 模型服务
- 普通对话、流式输出、异步调用

---

## 项目结构

```text
.
├─ src/
│  └─ universal_llm/
├─ scripts/
├─ examples/
├─ tests/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ README_langchain_vllm.md
```

---

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
pip install -e .
```

### 2. 启动本地 vLLM

请参考详细部署文档：

**[README_langchain_vllm.md](./README_langchain_vllm.md)**

### 3. 运行示例

```powershell
python scripts/run_vllm_example.py
python scripts/run_openai_example.py
python examples/basic_chat.py
python examples/stream_chat.py
```

### 4. 运行测试

```powershell
python tests/test_config.py
```

---

## 支持的 provider

### LangChain 原生 provider

- `openai`
- `anthropic`
- `google_genai`
- `ollama`

### OpenAI-compatible provider

- `deepseek`
- `openai_compatible`
- `vllm`
- `lm_studio`
- `local_openai`

---

## 核心设计

### `src/universal_llm/config.py`
- 配置类
- provider 注册信息
- 异常类
- LangChain 运行时加载

### `src/universal_llm/factory.py`
- 创建底层 chat model

### `src/universal_llm/messages.py`
- 统一消息结构
- role 标准化
- 文本提取

### `src/universal_llm/client.py`
- `UniversalLLM` 主调用接口

---

## 详细文档

详细部署说明、工作流程、vLLM Docker 搭建步骤、脚本运行方式，请查看：

**[README_langchain_vllm.md](./README_langchain_vllm.md)**

---

## 该项目特点：

- 统一接口设计
- 本地模型与云端模型统一调用
- vLLM 本地部署流程
- 便于扩展到 Agent / RAG / Tool
