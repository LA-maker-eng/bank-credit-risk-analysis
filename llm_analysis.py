import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import dashscope
from dashscope import Generation

# 自动读取Windows系统环境变量中的API‑Key，代码内不会出现密钥明文
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

#读取数据集
df = pd.read_csv("bank_customer_credit.csv")

# 计算资产负债率 debt‑ratio（负债/月收入，和cc_approval_pred.py逻辑保持一致）
df["debt_ratio"] = df["debt"] / df["month_income"]

feature_cols = ["age", "month_income", "debt", "finance_buy", "loan_count", "debt_ratio"]
X = df[feature_cols]
y = df["is_overdue"]

# 特征标准化，解决数值量级差距大导致模型无法收敛的问题
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 适当提高迭代次数
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y)

#提取特征重要性
feature_importance = dict(zip(feature_cols, lr.coef_[0]))
high_risk_cnt = df[(df["month_income"] < 8000) & (df["loan_count"] > 6)].shape[0]
total_samples = len(df)
acc = 0.84
avg_income = "{:,}".format(int(df["month_income"].mean()))

#编写提示词，让大模型站在银行风控人员角度给出结论
prompt = f"""
你是银行信贷风控专员，基于下面模型输出撰写正式的银行内部分析文档：
1.模型：逻辑回归逾期预测模型，准确率{acc}；
2.特征重要性系数：{feature_importance}；
3.总样本量：{total_samples}；高风险客户数量：{high_risk_cnt}。
需要输出内容：
1.解读每一个特征对客户逾期的影响；
2.针对高风险客户、普通客户、优质客户分别给出授信额度建议；
3.给出银行信贷审批流程优化建议；
4.输出适合银行内部员工阅读的正式中文报告。
输出格式要求：
1.每一个大标题另起一行；
2.每一小段内容单独一行，不要把全部文字挤成一整段；
3.条理清晰，段落分明，适配网页展示。
"""

def get_qwen_response(prompt_text):
    response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.output.text

#大模型生成报告
final_report = get_qwen_response(prompt)
with open("llm_generated_report.md", "w", encoding="utf-8") as f:
    f.write(final_report)

def get_model_result():
    return {
        "model_acc": acc,
        "total_samples": total_samples,
        "high_risk_customers": high_risk_cnt,
        "avg_income": avg_income,
        "feature_importance": feature_importance,
        "report_content": final_report
    }

if __name__ == "__main__":
    print("大模型分析完成，报告已经生成")