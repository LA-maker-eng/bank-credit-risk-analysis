import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("bank_customer_credit.csv")
df["debt_ratio"] = df["debt"] / df["month_income"]

# 1.不同收入区间逾期率柱状图
df["income_level"] = pd.cut(df["month_income"], bins=[0,8000,15000,25000,40000],
                           labels=["低收入","中等收入","中高收入","高收入"])
income_overdue = df.groupby("income_level")["is_overdue"].mean()
income_overdue.plot(kind="bar",title="不同收入区间逾期率")
plt.savefig("收入逾期统计图.png")

#2.理财购买次数‑逾期率折线图
finance_overdue = df.groupby("finance_buy")["is_overdue"].mean()
finance_overdue.plot(kind="line",title="理财购买次数和逾期率关系")
plt.savefig("理财逾期统计图.png")

#3.特征相关性热力图
plt.figure(figsize=(10,6))
sns.heatmap(df[["age","month_income","loan_count","debt_ratio","is_overdue"]].corr(),annot=True)
plt.title("特征相关性热力图")
plt.savefig("特征热力图.png")