# 基于 LangChain 的通用 LLM 调用接口设计与实现

## 一、项目简介

本项目实现了一个基于 **LangChain** 的通用大模型调用接口，目标是在同一套代码框架下同时支持：

1. LangChain 官方已支持的主流模型供应商；
2. 兼容 OpenAI API 的模型服务；
3. 通过 **vLLM** 调用本地部署模型；
4. 使用 **Qwen/Qwen2.5-1.5B-Instruct** 作为本地模型示例。

本项目适合作为课程作业、实验报告或大模型接口封装练习。

---

## 二、项目目标

本作业需要完成的核心内容如下：

- 使用 LangChain 编写一个通用的 LLM 调用接口；
- 能够在统一接口下切换不同模型供应商；
- 支持本地 vLLM 服务；
- 支持本地部署并调用 Qwen 模型。

---

## 三、项目文件说明

当前项目中与任务直接相关的文件如下：

- `universal_llm.py`：通用 LLM 调用接口核心代码；
- `example_usage.py`：本地 vLLM 调用示例；
- `openai_example.py`：OpenAI-compatible 接口调用示例；
- `requirements.txt`：本地 Python 客户端依赖；
- `README_langchain_vllm.md`：项目说明文档。

说明：  
本项目 **不再使用单独的 Python 启动脚本来启动 vLLM**，而是直接通过 **命令行 / Docker 命令** 启动本地 vLLM 服务。

---

## 四、总体设计思路

为了避免为每一个模型供应商单独编写不同的调用代码，本项目采用统一抽象方式，将模型服务划分为两类处理：

### 1. LangChain 官方 provider

对于 LangChain 官方已支持的供应商，使用：

```python
init_chat_model(...)
```

例如：

- OpenAI
- Anthropic
- Google
- Groq
- Ollama

### 2. OpenAI-compatible 接口

对于兼容 OpenAI API 的服务，使用：

```python
ChatOpenAI(...)
```

例如：

- vLLM
- DeepSeek
- Moonshot
- LM Studio
- SiliconFlow

### 3. 设计优势

这种设计具有以下优点：

- 结构清晰；
- 调用方式统一；
- 云端模型与本地模型可以共用一套接口；
- 方便后续扩展更多供应商。

---

## 五、核心代码设计

## 1. 配置类 `LLMConfig`

在 `universal_llm.py` 中，定义了统一配置类：

```python
@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
```

该类统一保存模型调用所需参数，包括：

- `provider`：模型供应商名称；
- `model`：模型名称；
- `api_key`：访问密钥；
- `base_url`：服务地址；
- `temperature`：生成温度；
- `max_tokens`：最大输出长度；
- `timeout`：超时时间；
- `model_kwargs`：额外模型参数；
- `extra_body`：兼容 OpenAI 接口时附加的请求体参数。

---

## 2. 通用调用类 `UniversalLLM`

`UniversalLLM` 是本项目的核心类，用于根据配置自动选择调用方式并统一执行请求。

### （1）模型构建逻辑

初始化时：

```python
self.llm = self._build_llm()
```

如果配置中存在 `base_url`，则说明目标服务是 OpenAI-compatible 接口，使用：

```python
ChatOpenAI(...)
```

如果配置中不存在 `base_url`，则说明使用 LangChain 官方 provider，使用：

```python
init_chat_model(...)
```

### （2）消息构造

`build_messages()` 负责统一构造消息列表，支持：

- system prompt；
- 历史对话；
- 当前用户输入。

例如：

```python
[
    ("system", "你是一个有帮助的 AI 助手。"),
    ("human", "你是谁？"),
    ("ai", "我是一个 AI 助手。"),
    ("human", "请介绍一下 Transformer。")
]
```

### （3）主要接口方法

#### `chat()`

普通文本问答：

```python
answer = client.chat("请简要介绍一下 Transformer。")
```

#### `stream()`

流式输出：

```python
for token in client.stream("请用三点介绍 LangChain"):
    print(token, end="", flush=True)
```

#### `get_llm()`

返回底层 LangChain 模型对象，便于后续扩展到 Chain、Agent、RAG 等更复杂场景。

---

## 六、本地 vLLM 实现方式说明

本项目采用的方式是：

### **通过命令行直接启动 vLLM 服务**

也就是说，本项目并不是通过 Python 脚本自动启动本地模型，而是通过命令行执行 Docker 命令启动 vLLM 容器，然后再由 Python 代码去调用这个本地服务。

这也是本项目的一项重要实现思路：

- **模型服务层**：通过 Docker 命令运行 vLLM；
- **调用层**：通过 LangChain 接口访问本地 vLLM 服务。

---

## 七、从启动本地 vLLM 到成功调用的完整实现步骤

下面给出完整的操作流程。

### 第 1 步：打开 Docker Desktop

重启电脑后，需要先确保 Docker Desktop 已经启动。  
可以在 PowerShell 中检查：

```powershell
docker version
```

如果能够正常显示版本信息，说明 Docker 可以使用。

---

### 第 2 步：进入项目目录

```powershell
cd 文件目录
```

---

### 第 3 步：检查 vLLM 镜像是否存在

```powershell
docker images
```

如果输出中存在：

```text
vllm-cpu
```

说明之前构建好的镜像仍然存在，无需重新构建。

如果不存在，则需要重新构建 vLLM 镜像。

---

### 第 4 步：配置 Hugging Face Token

为了让容器能够下载模型，需要配置 Hugging Face Token。建议先在 PowerShell 中设置环境变量：

```powershell
$env:HUGGING_FACE_HUB_TOKEN="你的HF_TOKEN"
```

说明：

- `HUGGING_FACE_HUB_TOKEN` 用于从 Hugging Face 下载模型；
- 这个 token 不应该直接写死在代码里；
- 如果 token 已泄露，应立即撤销并重新生成。

---

### 第 5 步：通过命令行启动本地 vLLM 服务

本项目采用如下命令启动本地 vLLM：

```powershell
docker run -d --name vllm-qwen `
  -p 8000:8000 `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  --env "HUGGING_FACE_HUB_TOKEN=$env:HUGGING_FACE_HUB_TOKEN" `
  vllm-cpu `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --dtype float16
```

该命令的含义如下：

- `-d`：后台运行容器；
- `--name vllm-qwen`：给容器命名，方便后续查看与管理；
- `-p 8000:8000`：将容器内服务映射到本机 8000 端口；
- `-v ...huggingface...`：挂载本地 Hugging Face 缓存，避免重复下载模型；
- `--env ...`：把 Hugging Face Token 传入容器；
- `vllm-cpu`：使用之前构建好的 vLLM 镜像；
- `--model Qwen/Qwen2.5-1.5B-Instruct`：指定部署模型；
- `--dtype float16`：设置数据类型，降低资源占用。

---

### 第 6 步：查看 vLLM 是否启动成功

运行：

```powershell
docker logs -f vllm-qwen
```

如果看到类似输出：

```text
Started server process
Waiting for application startup.
Application startup complete.
```

说明本地 vLLM 服务已经启动成功。

按 `Ctrl + C` 可以退出日志查看，但不会停止容器。

---

### 第 7 步：验证本地服务是否可访问

运行：

```powershell
curl http://127.0.0.1:8000/v1/models
```

如果返回模型信息，说明本地接口已经可以访问。

---

### 第 8 步：运行 Python 调用代码

本地 vLLM 启动成功后，运行：

```powershell
python example_usage.py
```

该文件中配置如下：

```python
vllm_config = LLMConfig(
    provider="vllm",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    temperature=0.2,
    extra_body={"top_k": 20},
)
```

这表示 Python 程序会通过 LangChain 连接本地地址：

```text
http://127.0.0.1:8000/v1
```

从而完成对本地 Qwen 模型的调用。

---

### 第 9 步：停止服务

如果需要停止本地 vLLM 服务，可以运行：

```powershell
docker stop vllm-qwen
```

如果还想删除这个容器，可以运行：

```powershell
docker rm vllm-qwen
```

---

## 八、本地调用示例说明

`example_usage.py` 用于演示本地 vLLM 调用：

```python
from universal_llm import LLMConfig, UniversalLLM

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
```

说明：

- `history` 用于手动传入历史记录；
- 当前实现支持多轮上下文，但不会自动记忆；
- `extra_body` 用于给 vLLM 传入额外参数，如 `top_k`。

---

## 九、OpenAI-compatible 接口示例说明

`openai_example.py` 展示了如何调用 OpenAI-compatible 服务。当前代码以 DeepSeek 为例：

```python
config = LLMConfig(
    provider="openai_compatible",
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.2,
)
```

运行前需要先设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
python openai_example.py
```

如果需要切换为 OpenAI 官方接口，则只需：

- 将 `provider` 改为 `openai`；
- 删除 `base_url`；
- 将 `model` 改为 OpenAI 模型名；
- 使用 `OPENAI_API_KEY`。

---

## 十、环境依赖安装

安装本地依赖：

```powershell
pip install -r requirements.txt
```

当前主要依赖包括：

- `langchain`
- `langchain-core`
- `langchain-openai`
- `python-dotenv`

如果需要接入更多 LangChain 官方 provider，还可以按需安装：

- `langchain-anthropic`
- `langchain-google-genai`
- `langchain-groq`
- `langchain-ollama`

---

## 十一、项目优点

本项目具有以下优点：

1. **统一接口**：本地模型与云端模型共用统一调用方式；
2. **结构清晰**：模型部署与接口调用分层明确；
3. **易于扩展**：新增模型供应商时通常只需要修改配置；
4. **支持多轮对话**：可传入历史消息；
5. **支持流式输出**：适合进一步扩展为聊天系统；
6. **实现路径明确**：命令行负责部署，Python 负责调用，便于理解整体流程。

---

## 十二、当前不足与可改进点

当前项目还存在以下可以继续完善的地方：

1. “支持所有供应商”更准确的表述应为“支持主流 LangChain provider 与 OpenAI-compatible 服务”；
2. 历史记录目前需要手动维护，尚未实现自动记忆；
3. 可进一步增加异常处理，例如服务未启动、配额不足、端口冲突等提示；
4. 后续还可以扩展到 RAG、Agent、多模型路由等功能。

---

## 十三、安全说明

不要将以下敏感信息直接写在代码中：

- OpenAI API Key
- Hugging Face Token
- DeepSeek API Key
- 其他平台密钥

推荐使用环境变量或 `.env` 文件，例如：

```powershell
$env:DEEPSEEK_API_KEY="你的API Key"
python openai_example.py
```

如果密钥已经泄露，应立即到对应平台后台撤销并重新生成。

---

## 十四、总结

本项目实现了一个基于 LangChain 的通用 LLM 调用接口，并通过“LangChain 官方 provider”与“OpenAI-compatible 接口”两种方式实现了多供应商统一接入。

在本项目中，本地 vLLM 服务并不是通过 Python 脚本启动，而是通过 **命令行 / Docker 命令** 直接部署；随后再由 `universal_llm.py` 提供的统一接口完成模型调用。

因此，本项目已经能够较好地满足课程作业中“利用 LangChain 写一个通用的 LLM 调用接口，并支持通过 vLLM 调用本地 Qwen 模型”的要求。
