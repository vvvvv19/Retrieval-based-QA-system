import json
import torch
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 获取BERT向量的函数
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)# 使用BERT分词器对输入的文本进行分词处理，转换为BERT模型需要的格式
    with torch.no_grad():# 使用torch.no_grad()来禁止计算梯度，减少内存占用
        outputs = model(**inputs)# 将处理过的输入文本传入BERT模型
    return outputs.last_hidden_state.mean(dim=1).numpy()# 提取BERT模型的最后一层隐藏状态（last_hidden_state），并对每个文本的向量进行平均池化，最终返回一个NumPy数组

# 创建FAISS索引
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # 向量的维度
    index = faiss.IndexFlatIP(d)  # 使用内积来计算相似度
    index.add(embeddings)  # 将所有问题的嵌入添加到索引中
    return index

if __name__ == "__main__":
    # 加载JSONL数据集
    data = []
    with open('datasets/test_datasets.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))  # 逐行解析每个JSON对象

    question_embeddings = []
    answer_embeddings = []
    questions = []
    answers = []

    for entry in data:
        question = entry['questions']
        answer = entry['answers']

        # 获取问题和答案的BERT向量
        question_embedding = get_bert_embedding(question)
        answer_embedding = get_bert_embedding(answer)
        # 添加嵌入向量到该列表里
        question_embeddings.append(question_embedding)
        answer_embeddings.append(answer_embedding)
        # 添加文本到该列表里
        questions.append(question)
        answers.append(answer)

    # 转换为NumPy数组
    question_embeddings = np.vstack(question_embeddings)
    answer_embeddings = np.vstack(answer_embeddings)

    # 创建FAISS索引
    index = create_faiss_index(question_embeddings)

    # 保存相关文件
    np.save('question_embeddings.npy', question_embeddings)
    np.save('answer_embeddings.npy', answer_embeddings)
    np.save('questions.npy', questions)
    np.save('answers.npy', answers)
    faiss.write_index(index, 'faiss_index.index')

    print("预处理完成，文件已保存。")
