import sqlite3
import pandas as pd
import os

DB_PATH = "bank_credit.db"

def init_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gender TEXT,
            age INTEGER,
            marital_status TEXT,
            education_level TEXT,
            employment_status TEXT,
            employment_length INTEGER,
            dwelling TEXT,
            has_car INTEGER,
            has_property INTEGER,
            has_phone INTEGER,
            has_email INTEGER,
            children_count INTEGER,
            family_member_count INTEGER,
            account_age INTEGER,
            month_income REAL,
            debt REAL,
            debt_ratio REAL,
            finance_buy INTEGER,
            loan_count INTEGER,
            is_overdue INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE loan_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            loan_amount REAL,
            loan_date TEXT,
            loan_type TEXT,
            interest_rate REAL,
            repayment_status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE financial_products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            product_type TEXT,
            expected_return REAL,
            risk_level TEXT,
            min_amount REAL,
            term_days INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE customer_finance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product_id INTEGER,
            purchase_amount REAL,
            purchase_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES financial_products(product_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def import_csv_to_db(csv_path="bank_customer_credit.csv"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_csv(csv_path)
    df["debt_ratio"] = df["debt"] / df["month_income"]
    
    df.to_sql("customers", conn, if_exists="append", index=False)
    
    loan_types = ["信用贷款", "房贷", "车贷", "经营贷", "消费贷"]
    loan_data = []
    for _, row in df.iterrows():
        for i in range(min(int(row["loan_count"]), 5)):
            loan_data.append({
                "customer_id": i + 1,
                "loan_amount": row["month_income"] * (2 + i),
                "loan_date": f"2024-{i%12+1}-{(i%28)+1}",
                "loan_type": loan_types[i % len(loan_types)],
                "interest_rate": 0.03 + i * 0.01,
                "repayment_status": "正常" if row["is_overdue"] == 0 else "逾期"
            })
    
    pd.DataFrame(loan_data).to_sql("loan_records", conn, if_exists="append", index=False)
    
    finance_products = [
        {"product_name": "稳健盈", "product_type": "保本理财", "expected_return": 0.028, "risk_level": "低风险", "min_amount": 10000, "term_days": 90},
        {"product_name": "成长宝", "product_type": "混合基金", "expected_return": 0.052, "risk_level": "中风险", "min_amount": 5000, "term_days": 180},
        {"product_name": "进取通", "product_type": "股票基金", "expected_return": 0.085, "risk_level": "高风险", "min_amount": 10000, "term_days": 365},
        {"product_name": "安心享", "product_type": "定期存款", "expected_return": 0.0225, "risk_level": "无风险", "min_amount": 50, "term_days": 365},
        {"product_name": "灵活宝", "product_type": "货币基金", "expected_return": 0.025, "risk_level": "低风险", "min_amount": 1, "term_days": 1}
    ]
    pd.DataFrame(finance_products).to_sql("financial_products", conn, if_exists="append", index=False)
    
    customer_finance_data = []
    import random
    for customer_id in range(1, min(len(df), 500) + 1):
        if random.random() < 0.3:
            product_id = random.randint(1, 5)
            customer_finance_data.append({
                "customer_id": customer_id,
                "product_id": product_id,
                "purchase_amount": random.randint(1000, 100000),
                "purchase_date": f"2024-{random.randint(1,12)}-{random.randint(1,28)}"
            })
    
    if customer_finance_data:
        pd.DataFrame(customer_finance_data).to_sql("customer_finance", conn, if_exists="append", index=False)
    
    conn.commit()
    conn.close()

def execute_query(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_customer_overdue_stats():
    query = '''
        SELECT 
            c.gender,
            COUNT(*) as total,
            SUM(c.is_overdue) as overdue_count,
            ROUND(SUM(c.is_overdue)*100.0/COUNT(*), 2) as overdue_rate
        FROM customers c
        GROUP BY c.gender
        ORDER BY overdue_rate DESC
    '''
    return execute_query(query)

def get_multi_table_analysis():
    query = '''
        SELECT 
            c.customer_id,
            c.name,
            c.month_income,
            c.is_overdue,
            COUNT(lr.record_id) as total_loans,
            SUM(lr.loan_amount) as total_loan_amount,
            fp.product_name,
            cf.purchase_amount
        FROM customers c
        LEFT JOIN loan_records lr ON c.customer_id = lr.customer_id
        LEFT JOIN customer_finance cf ON c.customer_id = cf.customer_id
        LEFT JOIN financial_products fp ON cf.product_id = fp.product_id
        GROUP BY c.customer_id, c.name, c.month_income, c.is_overdue, fp.product_name, cf.purchase_amount
        ORDER BY c.month_income DESC
        LIMIT 50
    '''
    return execute_query(query)

def get_high_risk_customers():
    query = '''
        SELECT 
            customer_id,
            name,
            month_income,
            debt_ratio,
            loan_count,
            is_overdue
        FROM customers
        WHERE month_income < 8000 AND loan_count > 6
        ORDER BY debt_ratio DESC
    '''
    return execute_query(query)

def get_finance_product_popularity():
    query = '''
        SELECT 
            fp.product_name,
            fp.product_type,
            fp.risk_level,
            COUNT(cf.id) as purchase_count,
            ROUND(SUM(cf.purchase_amount), 2) as total_amount
        FROM financial_products fp
        LEFT JOIN customer_finance cf ON fp.product_id = cf.product_id
        GROUP BY fp.product_name, fp.product_type, fp.risk_level
        ORDER BY total_amount DESC
    '''
    return execute_query(query)

def get_loan_type_distribution():
    query = '''
        SELECT 
            loan_type,
            COUNT(*) as count,
            ROUND(AVG(loan_amount), 2) as avg_amount,
            SUM(CASE WHEN repayment_status = '逾期' THEN 1 ELSE 0 END) as overdue_count
        FROM loan_records
        GROUP BY loan_type
        ORDER BY count DESC
    '''
    return execute_query(query)

if __name__ == "__main__":
    init_database()
    import_csv_to_db()
    print("数据库初始化完成")
    
    print("\n客户逾期统计：")
    print(get_customer_overdue_stats())
    
    print("\n多表关联分析：")
    print(get_multi_table_analysis().head())
    
    print("\n高风险客户：")
    print(get_high_risk_customers().head())
    
    print("\n理财产品热度：")
    print(get_finance_product_popularity())
    
    print("\n贷款类型分布：")
    print(get_loan_type_distribution())