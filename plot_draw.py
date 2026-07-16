import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

df = pd.read_csv("bank_customer_credit.csv")
df["debt_ratio"] = df["debt"] / df["month_income"]

plt.figure(figsize=(10, 6))
df["income_level"] = pd.cut(df["month_income"], bins=[0,8000,15000,25000,40000],
                           labels=["低收入(0-8k)","中等收入(8-15k)","中高收入(15-25k)","高收入(25k+)"])
income_overdue = df.groupby("income_level")["is_overdue"].mean() * 100
income_overdue.plot(kind="bar", color=["#cc0000", "#ff9900", "#009944", "#005599"], edgecolor="white", linewidth=2)
plt.title("不同收入区间客户逾期率", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("收入区间", fontsize=12)
plt.ylabel("逾期率(%)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
for i, v in enumerate(income_overdue):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("static/收入逾期统计图.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
finance_overdue = df.groupby("finance_buy")["is_overdue"].mean() * 100
finance_overdue.plot(kind="line", marker="o", markersize=8, linewidth=3, color="#005599")
plt.title("理财购买次数与逾期率关系", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("理财购买次数", fontsize=12)
plt.ylabel("逾期率(%)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
for i, v in enumerate(finance_overdue):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("static/理财逾期统计图.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
corr_data = df[["age","month_income","loan_count","debt_ratio","is_overdue"]].corr()
sns.heatmap(corr_data, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, 
            annot_kws={"fontsize": 11, "fontweight": "bold"},
            cbar_kws={"label": "相关系数"},
            linewidths=0.5, linecolor="white")
plt.title("信贷特征相关性热力图", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("static/特征热力图.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
age_bins = pd.cut(df["age"], bins=[18,25,35,45,55,100],
                  labels=["18-25岁","25-35岁","35-45岁","45-55岁","55岁以上"])
age_overdue = df.groupby(age_bins)["is_overdue"].mean() * 100
age_overdue.plot(kind="bar", color=["#005599", "#009944", "#ff9900", "#cc6600", "#cc0000"], edgecolor="white")
plt.title("不同年龄段客户逾期率", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("年龄段", fontsize=12)
plt.ylabel("逾期率(%)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
for i, v in enumerate(age_overdue):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("static/年龄逾期对比.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
loan_count_dist = df["loan_count"].value_counts().sort_index()
loan_count_dist.plot(kind="bar", color="#009944", edgecolor="white")
plt.title("客户贷款次数分布", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("贷款次数", fontsize=12)
plt.ylabel("客户数量", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("static/贷款次数分布.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
df["debt_level"] = pd.cut(df["debt_ratio"], bins=[0,0.5,1,2,5,100],
                          labels=["低负债(<50%)","中低负债(50-100%)","中高负债(100-200%)","高负债(200-500%)","极高负债(>500%)"])
debt_overdue = df.groupby("debt_level")["is_overdue"].mean() * 100
debt_overdue.plot(kind="bar", color=["#009944", "#005599", "#ff9900", "#cc6600", "#cc0000"], edgecolor="white")
plt.title("不同负债水平客户逾期率", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("负债水平", fontsize=12)
plt.ylabel("逾期率(%)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
for i, v in enumerate(debt_overdue):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("static/负债逾期对比.png", bbox_inches="tight")
plt.close()

print("所有图表生成完成！")