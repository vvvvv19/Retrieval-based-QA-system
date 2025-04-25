# 基于BERT和FAISS的中文问答系统

## 📘 项目简介

本项目构建了一个基于 **BERT + FAISS** 的中文语义问答系统，支持从结构化问答数据库中高效检索最相关的答案。

该系统采用了 **DPR（Dense Passage Retrieval）** 框架，分为两个主要模块：

1. 预处理模块：将问答数据转换为BERT嵌入，并构建FAISS索引。
2. 问答模块：用户输入问题后，系统通过BERT和FAISS返回最相关答案。

---

## 📂 文件结构

| 文件名               | 描述                       |
|----------------------|----------------------------|
| `preprocess_data.py` | 文本预处理和索引构建模块   |
| `QA_system.py`       | 问答交互模块               |
| `datasets/test_datasets.jsonl` | 输入问答数据集（JSONL 格式）|
| `question_embeddings.npy` | 预处理后的问题向量       |
| `answer_embeddings.npy`   | 预处理后的答案向量       |
| `questions.npy`           | 所有问题文本             |
| `answers.npy`             | 所有答案文本             |
| `faiss_index.index`       | FAISS 向量检索索引       |

---

## 🚀 使用方法

### 📌 1. 环境依赖

请确保已安装以下依赖：

```bash
pip install transformers torch faiss-cpu numpy
```

### 🧪 2. 预处理数据（运行一次）

```bash
python preprocess_data.py
```

- 加载中文BERT模型（`bert-base-chinese`）
- 读取数据集 `datasets/test_datasets.jsonl`，生成向量
- 构建 FAISS 索引
- 保存所有嵌入向量和索引文件

### 🧠 3. 启动问答系统

```bash
python QA_system.py
```

输入问题后系统将返回最相关的答案。例如：

```
请输入您的问题（输入'q'退出）：今天天气怎么样？

最相关的答案：
答案 1: 今天天气晴朗，适合出行。 相似度: 0.89
答案 2: 今天天气不错，可以考虑出去走走。 相似度: 0.87
...
```

---

## 🧩 模块说明

### 🔹 `preprocess_data.py`

- 使用 BERT 将每个问题/答案编码为稠密向量。
- 采用均值池化（`last_hidden_state.mean(dim=1)`）作为文本表示。
- 构建内积索引 `faiss.IndexFlatIP`。
- 保存所有嵌入向量和索引文件。

### 🔹 `QA_system.py`

- 加载保存的嵌入文件及FAISS索引。
- 用户输入问题后使用BERT获取其嵌入，并进行归一化处理。
- 使用 `index.search()` 返回前5个最相似的问题及对应答案。

---

## 📚 数据格式示例

输入数据为 `.jsonl` 格式，每行一个字典，包含如下字段：

```json
{
  "questions": "你叫什么名字？",
  "answers": "我是一个问答机器人。"
}
```

---

## 🔧 未来可扩展方向

- 使用更先进的中文预训练模型（如 MacBERT、RoFormer）
- 引入排序模型（RankNet、LambdaMART）
- 支持答案生成（结合RAG或LLM）
- 添加Web接口或API服务支持
- 多轮问答与上下文跟踪功能

---

## 🧠 1. 题目与背景介绍

本项目致力于构建一个基于检索的中文问答系统。该类系统广泛应用于智能客服、搜索引擎、知识问答等场景。

### 🛠 系统流程

1. **文本检索阶段**  
   - **传统方法**：如 TF-IDF、BM25，基于关键词出现频率进行打分。
   - **深度学习方法**：如 **DPR（Dense Passage Retrieval）**，通过 BERT 等模型将问题和文档转化为稠密向量，进行向量匹配。

2. **候选答案排序**  
   - 使用语义相似度度量，如 BERT 模型的向量余弦相似度。
   - 也可引入排序模型如 RankNet、LambdaMART 进一步优化排序效果。

3. **答案抽取/生成**
   - 可基于 BERT/RoBERTa/ALBERT 进行抽取式问答。
   - 或使用 GPT、DeepSick 等生成式大模型，基于候选句生成回答。

---

## 🧪 2. 系统模块与原理

本系统采用 DPR 思路，基于中文 BERT 模型和 FAISS 向量搜索引擎，分为以下两个模块：

### 🔧 2.1 文本预处理模块

- 使用 `bert-base-chinese` 将 JSONL 格式问答数据编码为向量。
- 对每个问题和答案，提取 `last_hidden_state` 并取均值作为嵌入。
- 通过 FAISS 的 `IndexFlatIP` 构建基于内积的向量索引。
- 输出保存：
  - 嵌入向量文件：`question_embeddings.npy`, `answer_embeddings.npy`
  - 原始文本文件：`questions.npy`, `answers.npy`
  - 检索索引：`faiss_index.index`

**输入数据格式（JSONL）**：

```json
{"questions": "你叫什么名字？", "answers": "我是一个问答机器人。"}
```

**运行结果**：
- 成功生成 BERT 嵌入向量。
- 构建并保存 FAISS 检索索引。
- 可用于后续高效的语义问答检索任务。

---

### 🤖 2.2 问答交互模块

- 加载保存的模型、嵌入向量、FAISS 索引。
- 用户输入查询后，使用 BERT 编码并归一化为单位向量。
- 使用 FAISS 的 `index.search()` 返回前 5 个最相似问题及其对应答案。
- 显示答案文本及与用户问题的相似度（内积值越高表示越相关）。

**相关函数说明**：
- `get_bert_embedding(text)`：获取输入文本的平均池化嵌入。
- `faiss.IndexFlatIP`：构建基于内积的向量检索索引。
- `index.search(query_vec, k)`：返回前 `k` 个最相似结果的索引及相似度。

---

