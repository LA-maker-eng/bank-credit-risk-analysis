import numpy as np
import pandas as pd

np.random.seed(42)  # 设置随机种子保证结果固定
samples = 20000  # 对应简历里面2万条数据

customer_id = np.arange(1, samples + 1)
age = np.random.randint(22, 60, samples)          # 年龄22‑60岁
month_income = np.random.randint(3000, 35000, samples) # 月收入3000‑35000
debt = np.random.randint(0, 25000, samples)       # 负债总额
finance_buy = np.random.randint(0, 12, samples)   # 理财购买次数
loan_count = np.random.randint(0, 15, samples)    # 历史贷款次数

# 设定规则：收入低、贷款次数多更容易逾期，用来贴合后面模型结论
prob = 0.02
prob += np.where(month_income < 8000, 0.25, 0)
prob += np.where(loan_count > 6, 0.3, 0)
prob = np.clip(prob,0,0.85)
is_overdue = np.random.binomial(1, prob, samples)

df = pd.DataFrame({
    "customer_id":customer_id,
    "age":age,
    "month_income":month_income,
    "debt":debt,
    "finance_buy":finance_buy,
    "loan_count":loan_count,
    "is_overdue":is_overdue
})

df.to_csv("bank_customer_credit.csv", index = False, encoding="utf‑8")
print("数据集生成完成，一共20000条客户数据")