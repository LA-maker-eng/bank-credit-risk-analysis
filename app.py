from flask import Flask, render_template, request, jsonify
from llm_analysis import get_model_result
from db_module import (
    init_database, import_csv_to_db, get_customer_overdue_stats,
    get_multi_table_analysis, get_high_risk_customers,
    get_finance_product_popularity, get_loan_type_distribution, execute_query
)
from rag_module import build_knowledge_base, query_knowledge_base
from data_validation import validate_customer_data, validate_batch_data, generate_validation_report
import pandas as pd
import os

app = Flask(__name__)

global_data = get_model_result()

if not os.path.exists("bank_credit.db"):
    init_database()
    import_csv_to_db()

if not os.path.exists("faiss_index"):
    build_knowledge_base()

@app.route("/")
def index():
    return render_template("index.html", data=global_data)

@app.route("/api/stats")
def get_stats():
    overdue_stats = get_customer_overdue_stats()
    high_risk = get_high_risk_customers()
    loan_dist = get_loan_type_distribution()
    
    return jsonify({
        "overdue_stats": overdue_stats.to_dict(orient="records"),
        "high_risk_count": len(high_risk),
        "loan_distribution": loan_dist.to_dict(orient="records")
    })

@app.route("/api/multi_table")
def get_multi_table():
    data = get_multi_table_analysis()
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/finance_products")
def get_finance_products():
    data = get_finance_product_popularity()
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/rag_query", methods=["POST"])
def rag_query():
    app.logger.info(f"请求方法: {request.method}")
    app.logger.info(f"Content-Type: {request.content_type}")
    raw_data = request.get_data().decode('utf-8')
    app.logger.info(f"原始数据: {raw_data}")
    
    if not request.is_json:
        app.logger.error("请求不是JSON格式")
        return jsonify({"error": "请求不是JSON格式"}), 400
    
    data = request.json
    app.logger.info(f"解析后的JSON: {data}")
    
    question = data.get("question", "")
    history = data.get("history", [])
    app.logger.info(f"收到问题: {question}")
    app.logger.info(f"问题长度: {len(question)}")
    
    if not question:
        app.logger.error("问题为空")
        return jsonify({"error": "请输入问题"}), 400
    
    try:
        result = query_knowledge_base(question, history)
        app.logger.info(f"回答长度: {len(result['answer'])}")
        app.logger.info(f"回答前50字: {result['answer'][:50]}...")
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        app.logger.error(f"错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/validate", methods=["POST"])
def validate_api():
    data = request.json.get("data", {})
    result = validate_customer_data(data)
    return jsonify(result)

@app.route("/api/validate_batch", methods=["POST"])
def validate_batch_api():
    records = request.json.get("records", [])
    if not records:
        return jsonify({"error": "请提供待校验数据"}), 400
    
    df = pd.DataFrame(records)
    result = validate_batch_data(df)
    result["report"] = generate_validation_report(result)
    return jsonify(result)

@app.route("/api/query", methods=["POST"])
def sql_query():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "请输入SQL查询语句"}), 400
    
    try:
        result = execute_query(query)
        return jsonify({
            "columns": result.columns.tolist(),
            "data": result.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search_customers", methods=["POST"])
def search_customers():
    keyword = request.json.get("keyword", "")
    
    try:
        if keyword:
            query = f'''
                SELECT 
                    c.customer_id,
                    c.name,
                    c.gender,
                    c.age,
                    c.education_level,
                    c.month_income,
                    c.debt,
                    c.debt_ratio,
                    c.loan_count,
                    c.is_overdue,
                    COUNT(lr.record_id) as loan_record_count
                FROM customers c
                LEFT JOIN loan_records lr ON c.customer_id = lr.customer_id
                WHERE c.name LIKE '%{keyword}%' OR c.customer_id = '{keyword}'
                GROUP BY c.customer_id, c.name, c.gender, c.age, c.education_level, c.month_income, c.debt, c.debt_ratio, c.loan_count, c.is_overdue
                ORDER BY c.month_income DESC
                LIMIT 50
            '''
        else:
            query = f'''
                SELECT 
                    c.customer_id,
                    c.name,
                    c.gender,
                    c.age,
                    c.education_level,
                    c.month_income,
                    c.debt,
                    c.debt_ratio,
                    c.loan_count,
                    c.is_overdue,
                    COUNT(lr.record_id) as loan_record_count
                FROM customers c
                LEFT JOIN loan_records lr ON c.customer_id = lr.customer_id
                GROUP BY c.customer_id, c.name, c.gender, c.age, c.education_level, c.month_income, c.debt, c.debt_ratio, c.loan_count, c.is_overdue
                ORDER BY c.month_income DESC
                LIMIT 100
            '''
        result = execute_query(query)
        return jsonify({
            "columns": result.columns.tolist(),
            "data": result.to_dict(orient="records"),
            "total": len(result)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/customer_detail/<int:customer_id>")
def customer_detail(customer_id):
    try:
        query = f'''
            SELECT 
                c.*,
                lr.loan_amount,
                lr.loan_type,
                lr.interest_rate,
                lr.repayment_status,
                fp.product_name,
                cf.purchase_amount
            FROM customers c
            LEFT JOIN loan_records lr ON c.customer_id = lr.customer_id
            LEFT JOIN customer_finance cf ON c.customer_id = cf.customer_id
            LEFT JOIN financial_products fp ON cf.product_id = fp.product_id
            WHERE c.customer_id = {customer_id}
        '''
        result = execute_query(query)
        return jsonify(result.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/risk_warnings")
def risk_warnings():
    try:
        query = '''
            SELECT 
                c.customer_id,
                c.name,
                c.month_income,
                c.debt,
                c.debt_ratio,
                c.loan_count,
                c.is_overdue,
                CASE 
                    WHEN c.debt_ratio > 3 THEN '高风险-负债过高'
                    WHEN c.month_income < 3000 AND c.loan_count > 3 THEN '高风险-低收入高负债'
                    WHEN c.is_overdue = 1 THEN '高风险-已逾期'
                    WHEN c.debt_ratio > 2 THEN '中风险-负债偏高'
                    ELSE '低风险'
                END as risk_level
            FROM customers c
            WHERE c.is_overdue = 1 OR c.debt_ratio > 2 OR (c.month_income < 3000 AND c.loan_count > 3)
            ORDER BY c.debt_ratio DESC
            LIMIT 100
        '''
        result = execute_query(query)
        return jsonify({
            "columns": result.columns.tolist(),
            "data": result.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)