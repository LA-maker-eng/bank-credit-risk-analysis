# 银行信贷风控智能分析平台

基于 Flask + Python + SQL + LangChain + FAISS RAG 构建的银行信贷风控分析系统，集成大数据分析、机器学习建模、大语言模型智能问答等能力。

## 功能模块

### 数据仪表盘
- 客户样本统计
- 风控模型准确率展示
- 高风险客户识别
- 交互式可视化图表（Chart.js）

### SQL多表查询分析
- 客户信息表（customers）
- 贷款记录表（loan_records）
- 理财产品表（financial_products）
- 客户理财购买表（customer_finance）
- 支持自定义SQL查询

### 智能问答系统（LangChain + FAISS RAG）
- 基于知识库的风控知识问答
- 信贷审批流程查询
- 风控指标体系解读
- 授信策略建议

### 金融接口数据校验
- 客户信息格式校验
- 批量数据校验报告
- 资产负债率预警
- 异常数据检测

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | Flask 3.1 |
| 数据处理 | Pandas 3.0, NumPy 2.5 |
| 机器学习 | Scikit-learn 1.9 |
| 可视化 | Matplotlib, Seaborn, Chart.js |
| 数据库 | SQLite |
| 大模型 | 通义千问（DashScope） |
| RAG | LangChain + FAISS |

## 环境要求

- Python 3.11+
- Windows/Linux/macOS
- 通义千问 API Key（环境变量 DASHSCOPE_API_KEY）

## 安装与运行

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量（Windows PowerShell）
$env:DASHSCOPE_API_KEY = "your-api-key"

# 运行应用
python app.py
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页仪表盘 |
| `/api/stats` | GET | 统计数据 |
| `/api/multi_table` | GET | 多表关联分析 |
| `/api/finance_products` | GET | 理财产品数据 |
| `/api/rag_query` | POST | RAG智能问答 |
| `/api/validate` | POST | 数据校验 |
| `/api/validate_batch` | POST | 批量数据校验 |
| `/api/query` | POST | SQL查询 |

## 项目结构

```
bank-credit-risk-analysis/
├── app.py                    # Flask应用入口
├── cc_approval_pred.py       # 风控模型训练
├── db_module.py              # SQL数据库模块
├── rag_module.py             # RAG知识库模块
├── data_validation.py        # 数据校验模块
├── llm_analysis.py           # 大模型分析
├── plot_draw.py              # 可视化图表绘制
├── bank_customer_credit.csv  # 客户信贷数据集
├── bank_credit.db            # SQLite数据库（自动生成）
├── requirements.txt          # 依赖列表
├── templates/
│   └── index.html            # 前端界面
└── static/                   # 静态资源
```

## 核心功能演示

### SQL查询示例
```sql
-- 查询不同性别客户逾期率
SELECT gender, COUNT(*) as total, 
       SUM(is_overdue) as overdue_count,
       ROUND(SUM(is_overdue)*100.0/COUNT(*), 2) as overdue_rate
FROM customers 
GROUP BY gender 
ORDER BY overdue_rate DESC;

-- 多表关联分析
SELECT c.name, c.month_income, lr.loan_type, lr.loan_amount
FROM customers c
LEFT JOIN loan_records lr ON c.customer_id = lr.customer_id
WHERE c.is_overdue = 1
ORDER BY c.month_income DESC;
```

### 智能问答示例
- Q: 银行信贷审批流程是什么？
- Q: 如何识别高风险客户？
- Q: 理财购买次数与逾期率有什么关系？

### 数据校验示例
```json
{
  "name": "张三",
  "age": 28,
  "month_income": 15000,
  "debt": 5000,
  "loan_count": 3,
  "phone": "13812345678"
}
```

## 岗位能力匹配

本项目体现以下数字金融实习生核心能力：

1. **银行数字金融业务参与**：完整的信贷风控业务流程实现
2. **数据展示素材构建**：专业级数据仪表盘和可视化展示
3. **应用系统开发**：面向银行内部员工的Web应用系统
4. **大数据分析**：构建数据分析模型，输出分析结果
5. **大模型技术应用**：提供智能化支持和优化方案