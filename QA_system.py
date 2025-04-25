import torch
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 获取BERT嵌入的函数
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# 加载预处理文件
def load_preprocessed_data():
    question_embeddings = np.load('question_embeddings.npy')
    answer_embeddings = np.load('answer_embeddings.npy')
    questions = np.load('questions.npy', allow_pickle=True)
    answers = np.load('answers.npy', allow_pickle=True)
    # 加载已经保存的 FAISS 索引
    index = faiss.read_index('faiss_index.index')
    return question_embeddings, answer_embeddings, questions, answers, index

# 获取用户输入问题的BERT嵌入
def get_user_query_embedding(user_query):
    return get_bert_embedding(user_query)

# 主程序块
if __name__ == "__main__":
    # 加载预处理的文件
    question_embeddings, answer_embeddings, questions, answers, index = load_preprocessed_data()

    # 将问题嵌入归一化，以便使用余弦相似度
    faiss.normalize_L2(question_embeddings)  # 对问题嵌入进行L2归一化

    # 获取用户输入的问题
    while True:
        # 获取用户输入的问题
        user_query = input("请输入您的问题（输入'q'退出）：")

        # 如果用户输入'q'，则退出循环
        if user_query.lower() == 'q':
            print("退出程序。")
            break

        # 获取用户问题的BERT嵌入
        user_query_embedding = get_user_query_embedding(user_query)

        # 对用户输入的问题嵌入进行归一化
        faiss.normalize_L2(user_query_embedding)

        # 使用加载的 FAISS 索引检索最相关的几个问题
        distances, indices = index.search(user_query_embedding, 5)  # 返回最相似的前5个问题

        # 获取最相关的答案
        relevant_answers = [answers[i] for i in indices[0]]
        relevant_distances = distances[0]  # 这些是内积（即余弦相似度）

        # 输出最相关的前5个答案
        print("\n最相关的答案：")
        for i in range(min(5, len(relevant_answers))):  # 确保输出最多3个答案
            print(f"答案 {i + 1}: {relevant_answers[i]}, 相似度: {relevant_distances[i]}")  # 输出内积作为相似度
