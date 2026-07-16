import os
import json
import numpy as np
from dashscope import Generation, TextEmbedding
import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

FAISS_INDEX_PATH = "faiss_index"
KNOWLEDGE_FILE = "knowledge/credit_risk_manual.txt"

knowledge_base = """
银行信贷风控知识手册

一、信贷审批流程
1. 客户申请：客户提交贷款申请，提供个人信息和资产证明
2. 资料审核：审核客户身份真实性、收入证明、征信记录
3. 风险评估：通过风控模型评估客户信用风险等级
4. 额度审批：根据风险等级确定授信额度和利率
5. 合同签署：客户确认贷款合同条款并签署
6. 放款执行：完成放款并开始计息

二、风控指标体系
1. 资产负债率：负债总额/资产总额，警戒线70%
2. 逾期率：逾期贷款余额/贷款总余额
3. 不良贷款率：不良贷款余额/贷款总余额
4. 客户信用评分：基于历史行为的综合评分
5. 贷款集中度：单一客户贷款占比

三、风险等级划分
- A级（低风险）：信用良好，还款能力强
- B级（中低风险）：信用较好，还款能力一般
- C级（中风险）：信用一般，存在一定风险
- D级（高风险）：信用较差，逾期风险高

四、授信策略
1. 低收入高负债客户：降低授信额度，提高利率
2. 高收入低负债客户：可适当提高授信额度
3. 有逾期记录客户：限制或拒绝授信
4. 优质客户：提供优惠利率和更高额度

五、数据分析方法
1. 特征工程：提取客户行为特征构建模型
2. 模型训练：使用机器学习算法训练风控模型
3. 模型评估：通过准确率、召回率评估模型效果
4. 模型迭代：根据业务反馈持续优化模型

六、合规要求
1. 数据隐私保护：严格遵守个人信息保护法
2. 反洗钱要求：识别可疑交易行为
3. 合规审查：确保授信流程符合监管规定

七、常见问题解答
Q: 为什么收入低的客户逾期风险更高？
A: 收入低意味着还款能力有限，一旦遇到突发情况更容易出现资金缺口。

Q: 理财购买次数与逾期率有什么关系？
A: 理财购买次数多通常说明客户有一定资金管理意识和风险承受能力，逾期率相对较低。

Q: 如何识别潜在的高风险客户？
A: 通过特征分析，关注低收入、高负债、频繁借贷的客户群体。

Q: 风控模型的准确率应该达到多少？
A: 根据业务需求，一般要求准确率达到80%以上，关键是平衡误拒率和漏判率。

八、银行理财产品知识
1. 保本理财：本金保障，收益相对较低，适合风险厌恶型客户
2. 混合基金：部分投资股票，部分投资债券，风险收益适中
3. 股票基金：主要投资股票市场，潜在收益高但风险也高
4. 定期存款：银行存款，无风险，收益稳定
5. 货币基金：流动性强，风险低，收益略高于活期存款

九、客户分层管理
1. 高净值客户：提供专属理财顾问和定制化服务
2. 普通客户：提供标准化产品和线上服务
3. 潜在客户：通过营销活动转化为正式客户
4. 风险客户：加强监控，必要时采取催收措施
"""

conversation_history = []

def get_embedding(text):
    response = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text
    )
    return response.output["embeddings"][0]["embedding"]

def build_knowledge_base():
    if not os.path.exists("knowledge"):
        os.makedirs("knowledge")
    
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        f.write(knowledge_base)
    
    chunks = []
    sections = knowledge_base.split("\n\n")
    for section in sections:
        section = section.strip()
        if len(section) > 0:
            if len(section) > 500:
                sub_chunks = [section[i:i+500] for i in range(0, len(section), 500)]
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)
    
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)
    
    embeddings = []
    for chunk in chunks:
        try:
            emb = get_embedding(chunk)
            embeddings.append(emb)
        except Exception as e:
            print(f"生成向量失败: {e}")
    
    if embeddings:
        index = np.array(embeddings)
        np.save(os.path.join(FAISS_INDEX_PATH, "index.npy"), index)
        
        with open(os.path.join(FAISS_INDEX_PATH, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    return True

def load_knowledge_base():
    index_path = os.path.join(FAISS_INDEX_PATH, "index.npy")
    chunks_path = os.path.join(FAISS_INDEX_PATH, "chunks.json")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        build_knowledge_base()
    
    index = np.load(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    return index, chunks

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_similar(question, top_k=3):
    index, chunks = load_knowledge_base()
    question_emb = get_embedding(question)
    
    similarities = []
    for i, chunk_emb in enumerate(index):
        sim = cosine_similarity(question_emb, chunk_emb)
        similarities.append((i, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, sim in similarities[:top_k]]
    
    return [chunks[idx] for idx in top_indices]

def query_knowledge_base(question, history=None):
    try:
        if history:
            history_text = "\n".join([f"用户: {h['user']}\n助手: {h['bot']}" for h in history[-3:]])
        else:
            history_text = ""
        
        prompt = f"""你是一位专业的银行信贷风控专家，精通信贷审批流程、风险评估、授信策略等领域。请基于以下知识库内容，用自然、专业的语言回答用户问题。

知识库内容：
{knowledge_base}

历史对话：
{history_text}

用户问题：{question}

回答要求：
1. 优先使用知识库中的信息进行回答，确保准确性；
2. 回答要自然流畅，用段落式表达，避免生硬的列表形式；
3. 根据问题类型采用不同语气：流程类问题详细步骤说明，分析类问题深入分析原因和影响，策略类问题给出具体建议，一般咨询友好专业解答；
4. 如果知识库中没有相关内容，可以基于专业知识回答。"""
        
        response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.output.text
        
        return {
            "answer": answer,
            "sources": []
        }
    except Exception as e:
        return {
            "answer": f"抱歉，暂时无法回答这个问题。错误信息：{str(e)}",
            "sources": []
        }

if __name__ == "__main__":
    build_knowledge_base()
    print("知识库构建完成")
    
    questions = [
        "银行信贷审批流程是什么？",
        "如何识别高风险客户？",
        "理财购买次数与逾期率有什么关系？",
        "风控模型的准确率应该达到多少？"
    ]
    
    for q in questions:
        print(f"\n问题：{q}")
        answer = query_knowledge_base(q)
        print(f"回答：{answer['answer']}")